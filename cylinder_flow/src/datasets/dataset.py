import numpy as np
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import h5py
from typing import Union, List, Tuple

from .data import Graph
from .transform import Periodic, Dirichlet, Neumann, \
    MyDistance, MyCartesian, DirichletInlet, MaskFace, NodeTypeInfo
from src.utils.utils import add_noise, add_noise_to_single_variable
from src.utils.padding import graph_padding
from src.models.voronoi_laplace import compute_discrete_laplace


class PDEGraphDataset(InMemoryDataset):
    def __init__(self, root, raw_files, processed_file, dataset_start,
                 dataset_used, time_start, time_used, window_size, dtype,
                 training=False):
        self.raw_files = raw_files
        self.processed_file = processed_file
        self.laplace_file = 'laplace.pt'
        self.d_file = 'd_vector.pt'
        self.root = root
        self.training = training

        self.dataset_start = dataset_start
        self.dataset_used = dataset_used
        self.time_start = time_start
        self.time_used = time_used
        self.window_size = window_size
        self.dtype = dtype

        self.periodic_trans = None
        self.dirichlet_trans = Dirichlet()
        self.inlet_trans = DirichletInlet()
        self.neumann_trans = None
        self.node_type_trans = NodeTypeInfo()
        self.mask_face_trans = MaskFace()
        self.graph_trans = T.Compose([
            T.Delaunay(),
            self.mask_face_trans,
            T.FaceToEdge(remove_faces=False),
            MyDistance(norm=True),
            MyCartesian(norm=True),
        ])
        transform = []
        if self.dirichlet_trans is not None:
            transform.append(self.dirichlet_trans)
        if self.inlet_trans is not None:
            transform.append(self.inlet_trans)
        if self.periodic_trans is not None:
            transform.append(self.periodic_trans)
        if self.neumann_trans is not None:
            transform.append(self.neumann_trans)
        transform.append(self.node_type_trans)
        transform.append(self.graph_trans)

        super(PDEGraphDataset, self).__init__(
            root=root,
            transform=None,
            pre_transform=T.Compose(transform),
            pre_filter=None
        )
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self.raw_files

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [self.processed_file, self.laplace_file, self.d_file]

    def download(self):
        pass

    def process(self):
        data_list = []
        file_handler = h5py.File(osp.join(self.root, self.raw_files))
        coarse_pos = file_handler['pos'][:]  # (n, 2)
        mesh = file_handler['mesh'][:]  # (n, 3)
        r = file_handler.attrs['r']
        mu = file_handler.attrs['mu']
        rho = file_handler.attrs['rho']
        node_type = file_handler['node_type']
        inlet_index, cylinder_index, outlet_index, inner_index = \
            node_type['inlet'][:], node_type['cylinder'][:], \
            node_type['outlet'][:], node_type['inner'][:]
        self.dirichlet_trans.set_index(cylinder_index)
        self.inlet_trans.set_index(inlet_index)
        self.node_type_trans.set_type_dict(node_type)
        self.mask_face_trans.set_cylinder_index(cylinder_index)
        for i in range(self.dataset_start, self.dataset_used):
            # (t, n_f, d)
            g = file_handler[str(i)]
            U = g['U']
            dt = g.attrs['dt']
            u_m = g.attrs['u_m']

            # dimensionless
            U = U / u_m
            pos = coarse_pos / (2 * r)
            dt = dt / (2 * r / u_m)

            # to tensor
            U_t = torch.tensor(U, dtype=self.dtype)  # (t, n, d)
            pos_t = torch.tensor(pos, dtype=self.dtype)
            truth_index = torch.arange(pos.shape[0], dtype=torch.long)  # (n,)
            # (n, 1)
            u_m_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * u_m
            dt_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * dt
            r_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * r
            mu_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * mu
            rho_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * rho

            for idx in torch.arange(self.time_start,
                                    self.time_start + self.time_used,
                                    step=self.window_size):
                # [t, n, c] -> [n, t, c]
                if idx + self.window_size > self.time_start + self.time_used:
                    break
                y = U_t[idx:idx + self.window_size].transpose(0, 1)
                if self.training:
                    y[:, 0, :] = add_noise(y[:, 0, :], percentage=0.03)
                data_list.append(Graph(pos=pos_t.clone(), y=y.clone(),
                                       truth_index=truth_index.clone(),
                                       dt=dt_t.clone(), u_m=u_m_t.clone(),
                                       r=r_t.clone(), mu=mu_t.clone(),
                                       rho=rho_t.clone()))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if osp.exists(self.processed_paths[1]):
            laplace_matrix = torch.load(
                self.processed_paths[1], weights_only=False)
            d_vector = torch.load(self.processed_paths[2], weights_only=False)
        else:
            laplace_matrix, d_vector = compute_discrete_laplace(data_list[0])
            laplace_matrix = laplace_matrix.clone()
            d_vector = d_vector.unsqueeze(dim=-1).clone()
            torch.save(laplace_matrix, self.processed_paths[1])
            torch.save(d_vector, self.processed_paths[2])
        for data in data_list:
            data.laplace_matrix = laplace_matrix
            data.d_vector = d_vector
            data.dirichlet_value = torch.zeros((data.dirichlet_index.shape[0],
                                                data.y.shape[2]))
            data.inlet_value = self.inlet_velocity(
                data.inlet_index, 1.)
            graph_padding(data, clone=True)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def inlet_velocity(inlet_index, u_m):
        u = u_m * torch.ones(inlet_index.shape[0])
        v = torch.zeros_like(u)

        return torch.stack((u, v), dim=-1)  # (m, 2)

    @staticmethod
    def dimensional(U_pred, U_gt, pos, u_m, D):
        """

        Args:
            U_pred (np.ndarray): shape (bn, 2)
            U_gt:
            pos:
            u_m (np.ndarray): shape (bn, 1) :
            D (float):

        Returns:

        """
        U_pred = U_pred * u_m
        U_gt = U_gt * u_m
        pos = pos * D

        return U_pred, U_gt, pos


class WaveGraphDataset(InMemoryDataset):
    def __init__(self, root, raw_files, processed_file, dataset_start,
                 dataset_used, time_start, time_used, window_size, dtype,
                 training=False):
        self.raw_files = raw_files
        self.processed_file = processed_file
        self.laplace_file = 'laplace.pt'
        self.d_file = 'd_vector.pt'
        self.root = root
        self.training = training

        self.dataset_start = dataset_start
        self.dataset_used = dataset_used
        self.time_start = time_start
        self.time_used = time_used
        self.window_size = window_size
        self.dtype = dtype

        # 波方程特定参数
        self.wave_speeds = []  # 存储每个数据集的波速
        self.time_steps = []   # 存储每个数据集的时间步长

        # 边界条件变换器
        self.dirichlet_trans = Dirichlet()
        self.neumann_trans = None  # 暂时不使用Neumann边界条件
        self.node_type_trans = NodeTypeInfo()

        # 图变换
        transform = []
        if self.dirichlet_trans is not None:
            transform.append(self.dirichlet_trans)
        if self.neumann_trans is not None:
            transform.append(self.neumann_trans)
        transform.append(self.node_type_trans)
        transform.append(T.Compose([
            T.Delaunay(),
            T.FaceToEdge(remove_faces=False),
            MyDistance(norm=True),
            MyCartesian(norm=True),
        ]))

        super(WaveGraphDataset, self).__init__(
            root=root,
            transform=None,
            pre_transform=T.Compose(transform),
            pre_filter=None
        )
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self.raw_files

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [self.processed_file, self.laplace_file, self.d_file]

    def download(self):
        pass

    def process(self):
        # 优化内存：分批处理数据
        file_handler = h5py.File(osp.join(self.root, self.raw_files))

        # 读取图结构信息（所有数据集共享）
        pos = file_handler['pos'][:]  # (n, 2)
        mesh = file_handler['mesh'][:]  # (n_faces, 3)

        # 读取边界信息
        node_type = file_handler['node_type']
        boundary_index, inner_index = node_type['boundary'][:], node_type['inner'][:]
        self.dirichlet_trans.set_index(boundary_index)

        # 转换基本张量（只转换一次）
        pos_t = torch.tensor(pos, dtype=self.dtype)
        truth_index = torch.arange(pos.shape[0], dtype=torch.long)

        # 先创建第一个样本用于计算拉普拉斯矩阵
        g = file_handler[str(self.dataset_start)]
        U = g['U']
        c = g.attrs['c']
        dt = g.attrs['dt']

        # 创建第一个样本（仅用于拉普拉斯矩阵计算）
        U_t = torch.tensor(np.array(U), dtype=self.dtype)
        y_sample = U_t[0:0 + self.window_size].transpose(0, 1)
        c_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * c
        dt_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * dt

        sample_graph = Graph(
            pos=pos_t.clone(),
            y=y_sample.clone(),
            truth_index=truth_index.clone(),
            c=c_t.clone(),
            dt=dt_t.clone(),
        )

        # 应用变换
        if self.pre_transform is not None:
            sample_graph = self.pre_transform(sample_graph)

        # 计算拉普拉斯矩阵
        if osp.exists(self.processed_paths[1]):
            laplace_matrix = torch.load(
                self.processed_paths[1], weights_only=False)
            d_vector = torch.load(self.processed_paths[2], weights_only=False)
        else:
            print("计算拉普拉斯矩阵...")
            laplace_matrix, d_vector = compute_discrete_laplace(sample_graph)
            laplace_matrix = laplace_matrix.clone()
            d_vector = d_vector.unsqueeze(dim=-1).clone()
            torch.save(laplace_matrix, self.processed_paths[1])
            torch.save(d_vector, self.processed_paths[2])
            print("拉普拉斯矩阵计算完成并保存")

        # 释放第一个样本的内存
        del sample_graph, U_t, y_sample, c_t, dt_t
        import gc
        gc.collect()

        # 分批处理：每次处理一个数据集
        data_list = []
        for i in range(self.dataset_start, self.dataset_used):
            print(f"处理数据集 {i+1}/{self.dataset_used}")
            g = file_handler[str(i)]
            U = g['U']  # (timesteps, n_nodes, 1)
            c = g.attrs['c']
            dt = g.attrs['dt']

            # 存储物理参数
            self.wave_speeds.append(c)
            self.time_steps.append(dt)

            # 转换为张量
            U_t = torch.tensor(np.array(U), dtype=self.dtype)  # (t, n, 1)
            c_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * c
            dt_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * dt

            # 生成滑动窗口样本（分批）
            num_windows = (self.time_used) // self.window_size
            print(f"  生成 {num_windows} 个时间窗口样本...")

            for window_idx, start_idx in enumerate(range(self.time_start,
                                                           self.time_start + self.time_used,
                                                           self.window_size)):
                if start_idx + self.window_size > self.time_start + self.time_used:
                    break

                y = U_t[start_idx:start_idx + self.window_size].transpose(0, 1)
                if self.training:
                    y[:, 0, :] = add_noise_to_single_variable(
                        y[:, 0, :], percentage=0.03)

                # 创建图数据
                data = Graph(
                    pos=pos_t.clone(),
                    y=y.clone(),
                    truth_index=truth_index.clone(),
                    c=c_t.clone(),
                    dt=dt_t.clone(),
                )

                # 添加拉普拉斯矩阵和边界条件
                data.laplace_matrix = laplace_matrix
                data.d_vector = d_vector
                if hasattr(data, 'dirichlet_index'):
                    data.dirichlet_value = torch.zeros(
                        (data.dirichlet_index.shape[0], data.y.shape[2])
                    )

                # 应用预变换
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # padding
                graph_padding(data, clone=True)

                data_list.append(data)

                # 定期清理内存
                if window_idx % 100 == 0:
                    gc.collect()
                    print(f"    处理进度: {window_idx}/{num_windows}")

            # 每处理完一个数据集后清理内存
            del U_t, c_t, dt_t
            gc.collect()
            print(f"  数据集 {i+1} 处理完成，当前样本数: {len(data_list)}")

        file_handler.close()

        # 应用预滤波
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        print(f"最终样本数: {len(data_list)}，开始保存...")

        # 合并所有样本并保存
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("数据保存完成")

    def get_dataset_parameters(self):
        """
        获取每个数据集的物理参数

        Returns:
            list: 每个元素是包含c和dt的字典
        """
        params = []
        for i in range(len(self.wave_speeds)):
            params.append({
                'c': self.wave_speeds[i],
                'dt': self.time_steps[i]
            })
        return params

    @staticmethod
    def wave_dimensional(U_pred, U_gt, pos, c, dt, L):
        """
        将无量纲化结果转换回有量纲形式

        Args:
            U_pred: 预测的波场
            U_gt: 真实的波场
            pos: 位置坐标
            c: 波速
            dt: 时间步长
            L: 特征长度

        Returns:
            有量纲的预测值、真实值、位置、时间
        """
        # 这里需要根据具体的无量纲化方法进行调整
        # 示例实现：
        U_pred_dim = U_pred  # 根据实际的无量纲化方法调整
        U_gt_dim = U_gt
        pos_dim = pos * L

        return U_pred_dim, U_gt_dim, pos_dim
