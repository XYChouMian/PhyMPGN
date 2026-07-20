import numpy as np
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import h5py
from typing import Union, List, Tuple

from .data import Graph
from .transform import Periodic, Dirichlet, Neumann, \
    MyDistance, MyCartesian, DirichletInlet, MaskFace, NodeTypeInfo, WaveNodeTypeInfo
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
        self.node_type_trans = WaveNodeTypeInfo()

        # 图变换
        transform = []
        transform.append(self.dirichlet_trans)
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
        data_list = []
        file_handler = h5py.File(osp.join(self.root, self.raw_files))

        # 读取图结构信息（所有数据集共享）
        pos = file_handler['pos'][:]  # (n, 2)
        # mesh = file_handler['mesh'][:]  # (n_faces, 3)

        # 读取边界信息
        node_type = file_handler['node_type']
        boundary_index, inner_index = node_type['boundary'][:], node_type['inner'][:]
        self.dirichlet_trans.set_index(boundary_index)

        # 为每个数据集生成样本
        for i in range(self.dataset_start, self.dataset_used):
            # 读取波场数据
            g = file_handler[str(i)]
            U = g['U']  # (timesteps, n_nodes, 1)

            # 假设每个数据集的波速和dt存储在属性中
            # 如果没有，需要从数据中推断或使用默认值
            c = g.attrs['c']
            dt = g.attrs['dt']

            # 存储当前数据集的物理参数
            self.wave_speeds.append(c)
            self.time_steps.append(dt)

            # 转换为张量
            U_t = torch.tensor(np.array(U), dtype=self.dtype)  # (t, n, 1)
            pos_t = torch.tensor(pos, dtype=self.dtype)
            truth_index = torch.arange(pos.shape[0], dtype=torch.long)  # (n,)

            # 创建物理参数张量（每个节点都有相同的参数值）
            c_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * c
            dt_t = torch.ones((pos.shape[0], 1), dtype=torch.float32) * dt

            # 生成滑动窗口样本
            # 提取时间窗口数据: [t, n, 1] -> [n, t, 1]
            for idx in torch.arange(self.time_start,
                                    self.time_start + self.time_used,
                                    step=self.window_size):
                if idx + self.window_size > self.time_start + self.time_used:
                    break
                y = U_t[idx:idx + self.window_size].transpose(0, 1)
                if self.training:  # 训练时添加噪声
                    y[:, 0, :] = add_noise_to_single_variable(
                        y[:, 0, :], percentage=0.03)
                data_list.append(Graph(
                    pos=pos_t,
                    y=y,
                    truth_index=truth_index,
                    c=c_t,
                    dt=dt_t,
                ))

        # 应用预滤波和预变换
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 计算或加载拉普拉斯矩阵
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

        # 为每个样本添加拉普拉斯矩阵和边界条件值
        for data in data_list:
            data.laplace_matrix = laplace_matrix
            data.d_vector = d_vector

            # 设置 Dirichlet 边界条件值
            data.dirichlet_value = torch.zeros(
                (data.dirichlet_index.shape[0], data.y.shape[2])
            )

            graph_padding(data, clone=True)

        # 合并所有样本
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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


class BLGraphDataset(InMemoryDataset):
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

        # 边界条件变换器
        # wall → Dirichlet (U=V=0), inlet → DirichletInlet (来流速度)
        self.dirichlet_trans = Dirichlet()
        self.inlet_trans = DirichletInlet()
        self.node_type_trans = NodeTypeInfo()

        # 图变换：使用 Delaunay 三角化生成图结构
        self.graph_trans = T.Compose([
            T.Delaunay(),
            T.FaceToEdge(remove_faces=False),
            MyDistance(norm=True),
            MyCartesian(norm=True),
        ])

        transform = []
        transform.append(self.dirichlet_trans)
        transform.append(self.inlet_trans)
        transform.append(self.node_type_trans)
        transform.append(self.graph_trans)

        super(BLGraphDataset, self).__init__(
            root=root,
            transform=None,
            pre_transform=T.Compose(transform),
            pre_filter=None
        )
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False)
        # super().__init__() 执行完毕意味着 process() 已完成，laplace.pt 已存在
        # 加载 laplace_matrix 作为 Dataset 属性，所有样本共享同一引用
        # 避免 collate 时每个样本都复制一份 8050×8050 矩阵
        self.laplace_matrix = torch.load(
            self.processed_paths[1], weights_only=False)
        self.d_vector = torch.load(
            self.processed_paths[2], weights_only=False)

    def get(self, idx):
        """获取单个样本，动态添加共享的 laplace_matrix（避免 collate 时复制）"""
        data = super().get(idx)
        data.laplace_matrix = self.laplace_matrix
        data.d_vector = self.d_vector
        
        Uinf
        return data

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

        # 读取图结构信息
        pos = file_handler['pos'][:]  # (n, 2)

        # 读取物理参数
        Uinf = file_handler.attrs['Uinf']
        delta99 = file_handler.attrs['delta99']
        Uinf_scalar = torch.tensor(Uinf, dtype=torch.float32)
        delta99_scalar = torch.tensor(delta99, dtype=torch.float32)

        # 读取节点类型并设置边界条件
        node_type = file_handler['node_type']
        wall_index, inlet_index = \
            node_type['wall'][:], node_type['inlet'][:]
        self.dirichlet_trans.set_index(wall_index)
        self.inlet_trans.set_index(inlet_index)
        self.node_type_trans.set_type_dict(node_type)

        # 无量纲化特征尺度
        pos_dimless = pos / delta99

        for i in range(self.dataset_start, self.dataset_used):
            g = file_handler[str(i)]
            U = g['U'][:]  # (t, n, 2)

            # 无量纲化
            U = U / Uinf

            # 转换为张量
            U_t = torch.tensor(U, dtype=self.dtype)  # (t, n, 2)
            pos_t = torch.tensor(pos_dimless, dtype=self.dtype)
            truth_index = torch.arange(pos.shape[0], dtype=torch.long)

            # 物理参数张量（每个节点相同值）
            n_nodes = pos.shape[0]
            Uinf_t = torch.ones((n_nodes, 1), dtype=torch.float32) * Uinf
            delta99_t = torch.ones((n_nodes, 1), dtype=torch.float32) * delta99

            for idx in torch.arange(self.time_start,
                                    self.time_start + self.time_used,
                                    step=self.window_size):
                if idx + self.window_size > self.time_start + self.time_used:
                    break
                # [t, n, c] -> [n, t, c]
                y = U_t[idx:idx + self.window_size].transpose(0, 1)
                if self.training:
                    y[:, 0, :] = add_noise(y[:, 0, :], percentage=0.03)

                # 读取 inlet 时变速度序列：[t_window, n_inlet, 2] -> [n_inlet, t_window, 2]
                inlet_U_profile = U_t[idx:idx + self.window_size, inlet_index, :].transpose(0, 1)

                data_list.append(Graph(pos=pos_t.clone(),
                                       y=y.clone(),
                                       truth_index=truth_index.clone(),
                                       Uinf=Uinf_scalar.clone(),
                                       delta99=delta99_scalar.clone(),
                                       inlet_value=inlet_U_profile.clone()))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 计算或加载拉普拉斯矩阵
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

        # 设置边界条件值（不设置 laplace_matrix 和 d_vector，由 get() 动态添加）
        for data in data_list:
            # wall: U=V=0
            data.dirichlet_value = torch.zeros((data.dirichlet_index.shape[0],
                                                data.y.shape[2]))
            # inlet: 无量纲来流速度 (1, 0)
            if not hasattr(data, 'inlet_value'):
                print("inlet_value is None")
                data.inlet_value = self.inlet_velocity(data.inlet_index, 1.)
            graph_padding(data, clone=True)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def inlet_velocity(inlet_index, u_inf, pos=None, timesteps=1):
        """
        inlet 速度剖面（时不变）

        Args:
            inlet_index: inlet 节点索引
            u_inf: 无量纲来流速度
            pos: 节点位置 (n, 2)，可选。如果提供，可基于位置计算边界层剖面
            timesteps: 时间步数，用于时变边界条件（BLGraphDataset 使用实际数据，不调用此方法）

        Returns:
            inlet 速度，形状 (n_inlet, 2) 或 (n_inlet, timesteps, 2)（如 timesteps > 1）

        注意：BLGraphDataset 在 Graph 中存储了从实际数据读取的时变 inlet 序列 (n_inlet, t, 2)，
              此静态方法主要用于 PDEGraphDataset 等需要常数来流的情况。
        """
        if pos is None:
            # 简单均匀来流 (u_inf, 0)
            u = u_inf * torch.ones(inlet_index.shape[0])
            v = torch.zeros_like(u)
        else:
            # 基于位置计算边界层剖面（如 Blasius 剖面）
            inlet_pos = pos[inlet_index]
            u = u_inf * torch.ones(inlet_index.shape[0])
            v = torch.zeros_like(u)
            # TODO: 如需要，可在此处实现基于 y 的边界层剖面计算

        inlet_val = torch.stack((u, v), dim=-1)  # (n_inlet, 2)

        # 如果需要时变边界条件，广播到时间维度
        if timesteps > 1:
            inlet_val = inlet_val.unsqueeze(1).repeat(1, timesteps, 1)  # (n_inlet, t, 2)

        return inlet_val

    @staticmethod
    def dimensional(U_pred, U_gt, pos, Uinf, delta99):
        """将无量纲结果转换回有量纲形式"""
        U_pred = U_pred * Uinf
        U_gt = U_gt * Uinf
        pos = pos * delta99
        return U_pred, U_gt, pos


class BLGraphDataset1(InMemoryDataset):
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

        # 边界条件变换器
        self.dirichlet_trans = Dirichlet()
        self.inlet_trans = DirichletInlet()
        self.node_type_trans = NodeTypeInfo()

        # 图变换：使用 Delaunay 三角化生成图结构
        self.graph_trans = T.Compose([
            T.Delaunay(),
            T.FaceToEdge(remove_faces=False),
            MyDistance(norm=True),
            MyCartesian(norm=True),
        ])

        transform = []
        transform.append(self.dirichlet_trans)
        transform.append(self.inlet_trans)
        transform.append(self.node_type_trans)
        transform.append(self.graph_trans)

        super(BLGraphDataset, self).__init__(
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
        """处理数据：laplace_matrix 单独存储，不放入样本中，避免 collate 时复制"""
        import gc
        import psutil
        import time as _time

        def mem_info(tag=""):
            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024
            print(f"  [MEM] {tag}: {mem:.1f} MB")

        t_start = _time.time()
        print("=" * 60)
        print("[START] BLGraphDataset.process()")
        mem_info("开始处理")

        file_handler = h5py.File(osp.join(self.root, self.raw_files))

        # 读取图结构信息
        pos = file_handler['pos'][:]
        print(f"[INFO] 节点数量: {pos.shape[0]}")

        # 读取物理参数
        Uinf = file_handler.attrs['Uinf']
        delta99 = file_handler.attrs['delta99']

        # 读取节点类型并设置边界条件
        node_type = file_handler['node_type']
        wall_index, inlet_index = \
            node_type['wall'][:], node_type['inlet'][:]
        self.dirichlet_trans.set_index(wall_index)
        self.inlet_trans.set_index(inlet_index)
        self.node_type_trans.set_type_dict(node_type)

        # 无量纲化特征尺度
        pos_dimless = pos / delta99

        # 转换为张量（只转换一次）
        pos_t = torch.tensor(pos_dimless, dtype=self.dtype)
        truth_index = torch.arange(pos.shape[0], dtype=torch.long)

        # 物理参数存储为标量（大幅节省内存）
        Uinf_scalar = torch.tensor(Uinf, dtype=torch.float32)
        delta99_scalar = torch.tensor(delta99, dtype=torch.float32)
        mem_info("读取共享数据后")

        # ===== 第1步：处理第一个样本，用于计算拉普拉斯矩阵 =====
        print("-" * 60)
        print("[STEP 1] 预处理第一个样本以计算拉普拉斯矩阵")
        first_sample = self._create_single_sample(
            file_handler, self.dataset_start, self.time_start,
            pos_t, truth_index, Uinf, Uinf_scalar, delta99_scalar, inlet_index
        )
        if self.pre_filter is not None and not self.pre_filter(first_sample):
            first_sample = None
        elif self.pre_transform is not None:
            first_sample = self.pre_transform(first_sample)

        # 诊断：打印第一个样本各字段大小
        total_size = 0
        print("\n[DIAG] 第一个样本各字段大小:")
        for key, value in first_sample:
            if isinstance(value, torch.Tensor):
                size_mb = value.element_size() * value.nelement() / 1024 / 1024
                total_size += size_mb
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                      f"size={size_mb:.2f} MB")
        print(f"  [TOTAL] 单个样本总大小: {total_size:.2f} MB\n")

        # 计算或加载拉普拉斯矩阵
        laplace_matrix, d_vector = self._get_or_compute_laplace(first_sample)
        lap_size = laplace_matrix.element_size() * laplace_matrix.nelement() / 1024 / 1024
        dvec_size = d_vector.element_size() * d_vector.nelement() / 1024 / 1024
        print(f"[DIAG] laplace_matrix: {laplace_matrix.shape}, {lap_size:.2f} MB")
        print(f"[DIAG] d_vector: {d_vector.shape}, {dvec_size:.2f} MB")
        mem_info("计算拉普拉斯后")
        del first_sample
        gc.collect()

        # ===== 第2步：创建所有样本（不存储 laplace_matrix） =====
        print("-" * 60)
        print("[STEP 2] 创建所有样本（laplace_matrix 单独存储，不放入样本）")
        all_samples = []
        total_samples = 0

        for i in range(self.dataset_start, self.dataset_used):
            t_ds = _time.time()
            g = file_handler[str(i)]
            U = g['U'][:]  # (t, n, 2)
            print(f"\n[DATASET {i}] 加载数据 shape={U.shape}")

            # 无量纲化
            U = U / Uinf

            # 转换为张量
            U_t = torch.tensor(U, dtype=self.dtype)
            del U
            mem_info(f"数据集 {i} 加载后")

            n_windows = 0
            for idx in torch.arange(self.time_start,
                                    self.time_start + self.time_used,
                                    step=self.window_size):
                if idx + self.window_size > self.time_start + self.time_used:
                    break

                # [t, n, c] -> [n, t, c]
                y = U_t[idx:idx + self.window_size].transpose(0, 1)
                if self.training:
                    y = y.clone()
                    y[:, 0, :] = add_noise(y[:, 0, :], percentage=0.03)

                # 读取 inlet 时变速度序列
                inlet_U_profile = U_t[idx:idx + self.window_size, inlet_index, :].transpose(0, 1).clone()

                data = Graph(
                    pos=pos_t,
                    y=y,
                    truth_index=truth_index,
                    Uinf=Uinf_scalar,
                    delta99=delta99_scalar,
                    inlet_value=inlet_U_profile
                )

                # 应用预变换（Delaunay、边界条件等）
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # 设置边界条件值（但不设置 laplace_matrix 和 d_vector）
                data.dirichlet_value = torch.zeros(
                    (data.dirichlet_index.shape[0], data.y.shape[2]))
                if not hasattr(data, 'inlet_value'):
                    data.inlet_value = self.inlet_velocity(data.inlet_index, 1.)
                graph_padding(data, clone=True)

                all_samples.append(data)
                total_samples += 1
                n_windows += 1

            print(f"[DATASET {i}] 完成: {n_windows} 个窗口, "
                  f"耗时 {_time.time() - t_ds:.2f}s")

            # 每个数据集结束后释放 U_t
            del U_t
            gc.collect()
            mem_info(f"数据集 {i} 处理后")

        print(f"\n[SUMMARY] 共创建 {total_samples} 个样本")

        # ===== 第3步：合并所有样本 =====
        print("-" * 60)
        print("[STEP 3] 合并所有样本（无 laplace_matrix，内存占用低）")
        t_merge = _time.time()
        data, slices = self.collate(all_samples)
        all_samples = None
        gc.collect()
        print(f"  [TIME] 合并耗时: {_time.time() - t_merge:.2f}s")
        mem_info("合并所有样本后")

        print("[SAVE] 保存数据")
        t_save = _time.time()
        torch.save((data, slices), self.processed_paths[0])
        print(f"  [TIME] 保存耗时: {_time.time() - t_save:.2f}s")
        mem_info("保存数据后")

        print("=" * 60)
        print(f"[DONE] 总耗时: {_time.time() - t_start:.2f}s")
        print("=" * 60)

    def _create_single_sample(self, file_handler, dataset_idx, time_idx,
                              pos_t, truth_index, Uinf, Uinf_scalar,
                              delta99_scalar, inlet_index):
        """创建单个样本（未变换）"""
        g = file_handler[str(dataset_idx)]
        U = g['U'][time_idx:time_idx + self.window_size]  # (t, n, 2)
        U = U / Uinf

        U_t = torch.tensor(U, dtype=self.dtype)
        y = U_t.transpose(0, 1)  # [n, t, c]

        inlet_U_profile = U_t[:, inlet_index, :].transpose(0, 1).clone()

        return Graph(
            pos=pos_t,
            y=y,
            truth_index=truth_index,
            Uinf=Uinf_scalar,
            delta99=delta99_scalar,
            inlet_value=inlet_U_profile
        )

    def _get_or_compute_laplace(self, sample_data):
        """计算或加载拉普拉斯矩阵"""
        print("计算或加载拉普拉斯矩阵")
        if osp.exists(self.processed_paths[1]):
            print("加载拉普拉斯矩阵")
            laplace_matrix = torch.load(
                self.processed_paths[1], weights_only=False)
            d_vector = torch.load(self.processed_paths[2], weights_only=False)
            # 加载的已经是 unsqueezed 的版本
        else:
            print("计算拉普拉斯矩阵")
            laplace_matrix, d_vector = compute_discrete_laplace(sample_data)
            d_vector = d_vector.unsqueeze(dim=-1)  # 先 unsqueeze
            torch.save(laplace_matrix, self.processed_paths[1])
            torch.save(d_vector, self.processed_paths[2])  # 保存 unsqueezed 版本
        return laplace_matrix, d_vector

    @staticmethod
    def inlet_velocity(inlet_index, u_inf, pos=None, timesteps=1):
        """
        inlet 速度剖面（时不变）

        Args:
            inlet_index: inlet 节点索引
            u_inf: 无量纲来流速度
            pos: 节点位置 (n, 2)，可选
            timesteps: 时间步数

        Returns:
            inlet 速度，形状 (n_inlet, 2) 或 (n_inlet, timesteps, 2)
        """
        if pos is None:
            u = u_inf * torch.ones(inlet_index.shape[0])
            v = torch.zeros_like(u)
        else:
            inlet_pos = pos[inlet_index]
            u = u_inf * torch.ones(inlet_index.shape[0])
            v = torch.zeros_like(u)

        inlet_val = torch.stack((u, v), dim=-1)  # (n_inlet, 2)

        if timesteps > 1:
            inlet_val = inlet_val.unsqueeze(1).repeat(1, timesteps, 1)

        return inlet_val

    @staticmethod
    def dimensional(U_pred, U_gt, pos, Uinf, delta99):
        """
        将无量纲结果转换回有量纲形式
        
        Args:
            U_pred: 预测速度（无量纲）
            U_gt: 真实速度（无量纲）
            pos: 位置（无量纲）
            Uinf: 标量或张量形式的参考速度
            delta99: 标量或张量形式的特征长度
        """
        # 如果是标量，自动广播；如果是张量，按原逻辑处理
        U_pred = U_pred * Uinf
        U_gt = U_gt * Uinf
        pos = pos * delta99
        return U_pred, U_gt, pos
