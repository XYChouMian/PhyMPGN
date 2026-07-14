#!/usr/bin/env python3
"""
波方程数据生成器脚本
生成与2d_cf相同数据结构的波方程数据集
"""

from src.pretty_print import UI
from src.datagenerators import FixedBCGenerator
from src.graph_management import MeshGraph
from src.data_management import GraphDataset
import os
import sys
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def extract_graph_info(graph: MeshGraph):
    """
    从图结构中提取网格信息 (igraph对象)

    Args:
        graph: MeshGraph对象 (继承自igraph.Graph)

    Returns:
        pos: 节点位置 (n_nodes, 2)
        mesh: 三角形网格 (n_faces, 3)
        node_types: 节点类型字典
    """
    # 从igraph图中提取节点位置
    if 'pos' in graph.vs.attribute_names():
        pos = np.array(graph.vs['pos'])  # [n_nodes, 2]
    elif 'x' in graph.vs.attribute_names() and 'y' in graph.vs.attribute_names():
        x = np.array(graph.vs['x'])
        y = np.array(graph.vs['y'])
        pos = np.column_stack([x, y])
    else:
        # 如果没有位置信息，创建默认位置
        n_nodes = graph.vcount()
        pos = np.zeros((n_nodes, 2))
        print("警告: 图中没有位置信息，使用默认零位置")

    # 从igraph图中提取三角形面
    if 'face' in graph.es.attribute_names():
        face_data = graph.es['face']
        mesh = np.array(face_data)
        if mesh.ndim == 1:
            mesh = mesh.reshape(-1, 3)
    else:
        # 如果没有面信息，尝试从图结构推导三角形
        mesh = []
        for edge in graph.es:
            u, v = edge.source, edge.target
            common_neighbors = set(
                graph.neighbors(u)) & set(graph.neighbors(v))
            for w in common_neighbors:
                if w != u and w != v:
                    triangle = sorted([u, v, w])
                    if triangle not in [sorted(list(m)) for m in mesh]:
                        mesh.append(triangle)

        mesh = np.array(mesh) if mesh else np.zeros((0, 3))

    # 提取边界节点信息
    boundary_manager = graph.boundary_manager

    # 直接获取所有边界节点
    all_boundary_nodes = boundary_manager.get_boundary_vertices(None)

    # 构建节点类型：boundary节点和inner节点
    node_types = {
        'boundary': np.array(sorted(list(all_boundary_nodes))),
        'inner': np.array(sorted(set(range(graph.vcount())) - all_boundary_nodes))
    }

    UI.info(f"节点类型: boundary={len(node_types['boundary'])}个,"
            f"inner={len(node_types['inner'])}个")

    return pos, mesh, node_types


def extract_time_series(graph_dataset: GraphDataset):
    """
    从图数据集中提取时间序列数据

    Args:
        graph_dataset: GraphDataset 对象

    Returns:
        u: 波高度场 (timesteps, n_nodes, 1)
    """
    u = graph_dataset.data.cpu().numpy()

    # 确保形状为 (timesteps, n_nodes, 1)
    if u.ndim == 2:
        u = u[:, :, None]  # [timesteps, n_nodes, 1]
    elif u.ndim != 3:
        raise ValueError(f"GraphDataset.data 的维度应该是2或3，但得到了 {u.ndim} 维")
    return u


def _generate_wave_data_common(output_dir, n_timesteps, density, is_test=False, print_structure=False, wave_speeds=None):
    """
    通用波方程数据生成函数（内部函数）

    Args:
        output_dir: 输出目录
        n_timesteps: 每个样本的时间步数
        grid_size: 网格大小
        density: 网格密度
        is_test: 是否为测试数据
        print_structure: 是否打印文件结构
        wave_speeds: 自定义波速列表（如果为None则自动生成）
    """
    dataset_type = "测试" if is_test else "训练"
    prefix = "test" if is_test else "train"

    # 使用提供的波速列表
    wave_speeds = np.array(wave_speeds)
    n_samples = len(wave_speeds)
    UI.info(
        f"使用自定义波速: {n_samples}个，从{wave_speeds[0]:.4f}到{wave_speeds[-1]:.4f}")

    UI.subsubsection(f"开始生成波方程{dataset_type}数据集...")
    UI.info(f"参数: n_samples={n_samples}, n_timesteps={n_timesteps}")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化生成器
    generator = FixedBCGenerator()

    # 生成数据集
    graph: MeshGraph = generator.generate_conditions(
        n=n_samples,
        density=density,
        n_eval=n_timesteps,
        progress_bar=True,
        c_list=wave_speeds,
    )

    # 提取图结构信息（所有样本共享）
    pos, mesh, node_types = extract_graph_info(graph)

    UI.info(f"图结构信息:")
    UI.info(f"  节点数: {len(pos)}")
    UI.info(f"  三角形面数: {len(mesh)}")
    UI.info(f"  节点类型: {list(node_types.keys())}")

    # 创建 HDF5 文件
    output_file = output_dir / \
        f"{prefix}_wave_{n_samples}x{n_timesteps}x{len(pos)}x1.h5"
    UI.info(f"保存到: {output_file}")

    with h5py.File(output_file, 'w') as h5f:
        # 保存图结构（所有样本共享）
        h5f.create_dataset('pos', data=pos)
        h5f.create_dataset('mesh', data=mesh)

        # 保存节点类型
        node_type_group = h5f.create_group('node_type')
        for node_type_name, node_indices in node_types.items():
            node_type_group.create_dataset(node_type_name, data=node_indices)

        # 保存每个样本的时间序列数据和参数
        for i, graph_dataset in enumerate(generator.graphdatasets):
            # 提取时间序列
            u = extract_time_series(graph_dataset)

            # 调整时间步数（如果需要）
            if len(u) != n_timesteps:
                if len(u) > n_timesteps:
                    u = u[:n_timesteps]
                    UI.warning(f"样本{i}时间步数{len(u)}超过要求的{n_timesteps}，已截断")
                else:
                    raise UI.error(
                        f"样本{i+1}时间步数{len(u)}小于要求的{n_timesteps}，无法填充",
                        raise_type=ValueError)

            # 保存时间序列
            sample_group = h5f.create_group(str(i))
            sample_group.create_dataset('U', data=u)

            # 保存该数据集的物理参数（使用指定的波速）
            for param_name, param in zip(graph_dataset.params_name, graph_dataset.params):
                sample_group.attrs[param_name] = param
            if 'dt' not in sample_group.attrs:
                sample_group.attrs['dt'] = graph_dataset.dt

    UI.success(f"波方程{dataset_type}数据集生成完成!")
    UI.info(f"文件: {output_file}")
    UI.info(f"结构: {n_samples}样本 x {n_timesteps}时间步 x {len(pos)}节点 x 1特征")
    UI.info(f"波速范围: [{wave_speeds[0]:.4f}, {wave_speeds[-1]:.4f}]")

    # 打印文件结构（仅训练数据）
    if print_structure:
        with h5py.File(output_file, 'r') as h5f:
            UI.subsection("生成的数据结构:")
            UI.info(f"顶层键: {list(h5f.keys())}")
            UI.info(f"pos: {h5f['pos'].shape} {h5f['pos'].dtype}")
            UI.info(f"mesh: {h5f['mesh'].shape} {h5f['mesh'].dtype}")

            UI.info(f"node_type: {list(h5f['node_type'].keys())}")
            for key in h5f['node_type'].keys():
                UI.info(f"  {key}: {h5f[f'node_type/{key}'].shape}")

            # 打印前两个样本的参数
            for i in range(min(2, n_samples)):
                UI.info(f"{i}/U: {h5f[f'{i}/U'].shape} {h5f[f'{i}/U'].dtype}")
                if 'c' in h5f[f'{i}'].attrs:
                    UI.info(f"  属性 c={h5f[f'{i}'].attrs['c']:.4f}, dt={h5f[f'{i}'].attrs['dt']:.6f}")

            if n_samples > 2:
                UI.info(f"...")
                UI.info(
                    f"{n_samples-1}/U: {h5f[f'{n_samples-1}/U'].shape} {h5f[f'{n_samples-1}/U'].dtype}")
                if 'c' in h5f[f'{n_samples-1}'].attrs:
                    UI.info(f"  属性 c={h5f[f'{n_samples-1}'].attrs['c']                            :.4f}, dt={h5f[f'{n_samples-1}'].attrs['dt']:.6f}")


def generate_train_data(output_dir, n_timesteps=2000, density=1.0):
    """
    生成波方程训练数据集，共10个波速

    Args:
        output_dir: 输出目录
        n_timesteps: 每个样本的时间步数
        density: 网格密度
    """
    UI.section("生成训练数据集")
    # 训练集：10个波速，从0.1到10指数增长
    n_samples = 10
    wave_speeds = np.logspace(np.log10(0.1), np.log10(10), n_samples)
    UI.attention(
        f"自动生成训练集波速: {n_samples}个，从{wave_speeds[0]:.4f}到{wave_speeds[-1]:.4f}")
    _generate_wave_data_common(output_dir, n_timesteps, density,
                               is_test=False, print_structure=True, wave_speeds=wave_speeds)


def generate_test_data(output_dir, n_timesteps=2000, density=1.0):
    """
    生成波方程测试数据集，共30个波速
    Args:
        output_dir: 输出目录
        n_timesteps: 每个样本的时间步数
        density: 网格密度
    """
    UI.section("生成测试数据集")
    # 测试集：20个波速，从0.01到100指数增长
    n_samples = 20
    wave_speeds = np.logspace(np.log10(0.01), np.log10(100), n_samples)
    UI.attention(
        f"自动生成测试集波速: {n_samples}个，从{wave_speeds[0]:.4f}到{wave_speeds[-1]:.4f}")
    _generate_wave_data_common(output_dir, n_timesteps, density,
                               is_test=True, print_structure=False, wave_speeds=wave_speeds)


def main():
    """主函数"""
    UI.header("生成波方程数据集")
    # 设置输出目录
    output_dir = Path("/home/wqx/projects/PhyMPGN/data/wave")

    # 生成训练数据（10个波速，从0.1到10指数增长）
    generate_train_data(
        output_dir=output_dir,
        n_timesteps=2000,
        density=1.5,
    )

    # 生成测试数据（30个波速，从0.01到100指数增长）
    generate_test_data(
        output_dir=output_dir,
        n_timesteps=2000,
        density=1.5,
    )


if __name__ == "__main__":
    main()
