#!/usr/bin/env python3
"""
边界层数据生成器脚本
生成与 2d_cf 相同数据结构的边界层数据集
支持网格截断功能以减少计算量
"""

import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple


def create_mesh_structure(height: int, width: int) -> np.ndarray:
    """
    为规则网格创建网格连接结构

    Args:
        height: 网格高度
        width: 网格宽度

    Returns:
        mesh: 网格边数组，每条边由两个节点索引组成
    """
    mesh = []
    for i in range(height):
        for j in range(width):
            node_idx = i * width + j

            # 水平连接
            if j < width - 1:
                neighbor_idx = i * width + (j + 1)
                mesh.append([node_idx, neighbor_idx])

            # 垂直连接
            if i < height - 1:
                neighbor_idx = (i + 1) * width + j
                mesh.append([node_idx, neighbor_idx])

    return np.array(mesh, dtype=np.int64)


def create_node_types(height: int, width: int) -> dict:
    """
    为边界层流创建节点类型分类

    网格布局（height包含wall行）：
    - 第0行 (y=0): wall节点（无滑移壁面，U=V=0）
    - 左边界 (x=0, 第1~height-1行): inlet节点（来流）
    - 上边界 (y=y_max, 第height-1行): inlet节点（Dirichlet边界）
    - 右边界 (x=x_max, 第1~height-2行): outlet节点（出流）
    - 其余: inner节点

    节点索引规则：node_idx = row * width + col

    Args:
        height: 网格高度（包含wall行）
        width: 网格宽度

    Returns:
        node_type: 包含不同类型节点索引的字典
    """
    n_nodes = height * width

    # wall节点：第0行（y=0），所有列
    wall_indices = list(range(width))  # 0 to width-1

    # inlet节点：左边界（第1行到最后一行的第0列）+ 上边界（最后一行所有列）
    inlet_indices = []
    for i in range(1, height):  # 左边界，排除第0行（wall）
        inlet_indices.append(i * width)
    for j in range(width):  # 上边界（最后一行）
        inlet_indices.append((height - 1) * width + j)

    # outlet节点：右边界（第1行到倒数第2行的最后一列，排除上边界角点）
    outlet_indices = []
    for i in range(1, height - 1):
        outlet_indices.append(i * width + (width - 1))

    # inner节点：所有非边界节点
    boundary_set = set(wall_indices) | set(inlet_indices) | set(outlet_indices)
    inner_indices = [i for i in range(n_nodes) if i not in boundary_set]

    node_type = {
        'inner': np.array(inner_indices, dtype=np.int32),
        'inlet': np.array(inlet_indices, dtype=np.int32),
        'outlet': np.array(outlet_indices, dtype=np.int32),
        'wall': np.array(wall_indices, dtype=np.int32)
    }

    return node_type


def read_and_truncate_group_data(
    h5_source: h5py.File,
    group_idx: int,
    x_truncate: int = None,
    y_truncate: int = None,
    t_truncate: int = None
) -> tuple:
    """
    读取并截断单个组的数据

    Args:
        h5_source: 源HDF5文件对象
        group_idx: 组索引
        x_truncate: x方向截断宽度，None表示不截断
        y_truncate: y方向截断高度，None表示不截断
        t_truncate: 时间步截断数量，None表示不截断

    Returns:
        (U_data, V_data): 该组的速度场数据（可能被截断）
    """
    U_data = h5_source['U'][group_idx]  # shape: (5433, 169, 169)
    V_data = h5_source['V'][group_idx]  # shape: (5433, 169, 169)

    # 在时间维度截断（沿第0个维度）
    if t_truncate is not None:
        U_data = U_data[:t_truncate]  # shape: (t_truncate, 169, 169)
        V_data = V_data[:t_truncate]

    # 在y方向截断（沿第1个维度）
    if y_truncate is not None:
        U_data = U_data[:, :y_truncate]  # shape: (t, y_truncate, 169)
        V_data = V_data[:, :y_truncate]

    # 在x方向截断（沿第2个维度）
    if x_truncate is not None:
        U_data = U_data[:, :, :x_truncate]  # shape: (t, y, x_truncate)
        V_data = V_data[:, :, :x_truncate]

    return U_data, V_data


def truncate_grid_coordinates(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    x_truncate: int = None,
    y_truncate: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    截断网格坐标

    Args:
        grid_x: 原始x坐标网格，shape: (169, 169)
        grid_y: 原始y坐标网格，shape: (169, 169)
        x_truncate: x方向截断宽度，None表示不截断
        y_truncate: y方向截断高度，None表示不截断

    Returns:
        (grid_x_trunc, grid_y_trunc): 截断后的坐标网格
    """
    grid_x_trunc = grid_x
    grid_y_trunc = grid_y

    if y_truncate is not None:
        grid_x_trunc = grid_x_trunc[:y_truncate, :]  # shape: (y_truncate, 169)
        grid_y_trunc = grid_y_trunc[:y_truncate, :]

    if x_truncate is not None:
        grid_x_trunc = grid_x_trunc[:, :x_truncate]  # shape: (y, x_truncate)
        grid_y_trunc = grid_y_trunc[:, :x_truncate]

    return grid_x_trunc, grid_y_trunc


def add_wall_boundary_to_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在网格坐标中添加wall边界行（y=0）

    将wall行插入到第0行位置，x坐标与原始第一行对齐，y坐标为0

    Args:
        grid_x: 截断后的x坐标网格，shape: (height, width)
        grid_y: 截断后的y坐标网格，shape: (height, width)

    Returns:
        (grid_x_new, grid_y_new): 添加wall行后的坐标网格，shape: (height+1, width)
    """
    width = grid_x.shape[1]

    # wall行的x坐标与原始第一行相同
    wall_x = grid_x[0:1, :].copy()  # shape: (1, width)
    # wall行的y坐标全为0
    wall_y = np.zeros((1, width), dtype=grid_y.dtype)  # shape: (1, width)

    # 在第0行位置插入wall行
    # shape: (height+1, width)
    grid_x_new = np.concatenate([wall_x, grid_x], axis=0)
    # shape: (height+1, width)
    grid_y_new = np.concatenate([wall_y, grid_y], axis=0)

    return grid_x_new, grid_y_new


def process_single_group(U_data: np.ndarray, V_data: np.ndarray) -> np.ndarray:
    """
    处理单个组的数据，合并速度分量并展平空间维度
    在第0行添加wall边界行（U=V=0）

    Args:
        U_data: U速度分量，shape: (timesteps, height, width)
        V_data: V速度分量，shape: (timesteps, height, width)

    Returns:
        velocity_data: 处理后的速度数据，shape: (timesteps, (height+1)*width, 2)
    """
    timesteps, height, width = U_data.shape

    # 创建wall行数据（U=0, V=0）
    # (timesteps, 1, width)
    wall_U = np.zeros((timesteps, 1, width), dtype=U_data.dtype)
    wall_V = np.zeros((timesteps, 1, width), dtype=V_data.dtype)

    # 在第0行位置插入wall行
    # (timesteps, height+1, width)
    U_data = np.concatenate([wall_U, U_data], axis=1)
    V_data = np.concatenate([wall_V, V_data], axis=1)

    # 合并速度分量：stack后在最后一个维度
    # (timesteps, height+1, width, 2)
    velocity_data = np.stack([U_data, V_data], axis=-1)

    # 展平空间维度
    new_height = height + 1
    velocity_data = velocity_data.reshape(timesteps, new_height * width, 2)

    return velocity_data


def generate_bl_data_file(
    source_file: Path,
    output_file: Path,
    group_indices: List[int],
    source_h5: h5py.File,
    pos: np.ndarray,
    mesh: np.ndarray,
    node_type: dict,
    source_attrs: dict,
    x_truncate: int = None,
    y_truncate: int = None,
    t_truncate: int = None
) -> None:
    """
    生成边界层数据文件（支持分块处理大数据和网格截断）

    Args:
        source_file: 源数据文件路径
        output_file: 输出文件路径
        group_indices: 要包含的组索引列表
        source_h5: 源HDF5文件对象
        pos: 节点位置数组
        mesh: 网格连接数组
        node_type: 节点类型字典
        source_attrs: 源数据属性
        x_truncate: x方向截断宽度，None表示不截断
        y_truncate: y方向截断高度，None表示不截断
        t_truncate: 时间步截断数量，None表示不截断
    """
    print(f"💾 生成数据文件: {output_file}")
    print(f"   包含组: {group_indices}")
    if x_truncate is not None:
        print(f"   x方向截断宽度: {x_truncate}")
    if y_truncate is not None:
        print(f"   y方向截断高度: {y_truncate}")
    if t_truncate is not None:
        print(f"   时间步截断: {t_truncate}")

    with h5py.File(output_file, 'w') as f:
        # 添加全局属性
        f.attrs['mu'] = source_attrs.get('nu', 7.878999781496532e-07)
        f.attrs['rho'] = 1000.0
        f.attrs['Uinf'] = source_attrs.get('Uinf', 0.35199999809265137)
        f.attrs['delta99'] = source_attrs.get('delta99', 0.04490000009536743)
        f.attrs['dT'] = source_attrs.get('dT', 0.0010000000474974513)

        # 添加截断信息
        if x_truncate is not None:
            f.attrs['x_truncate'] = x_truncate
        if y_truncate is not None:
            f.attrs['y_truncate'] = y_truncate
        if t_truncate is not None:
            f.attrs['t_truncate'] = t_truncate
        f.attrs['original_grid_size'] = '169x169'

        # 添加网格信息
        f.create_dataset('mesh', data=mesh)
        f.create_dataset('pos', data=pos)

        # 添加节点类型
        node_type_group = f.create_group('node_type')
        for name, indices in node_type.items():
            node_type_group.create_dataset(name, data=indices)

        # 分组添加数据
        for i, group_idx in enumerate(group_indices):
            print(f"   处理第 {group_idx} 组...")

            # 读取该组数据（可能截断）
            U_data, V_data = read_and_truncate_group_data(
                source_h5, group_idx, x_truncate, y_truncate, t_truncate
            )

            # 处理数据
            velocity_data = process_single_group(U_data, V_data)

            # 创建组并保存数据
            group = f.create_group(str(i))
            group.create_dataset('U', data=velocity_data)
            group.attrs['dt'] = source_attrs.get('dT', 0.0010000000474974513)
            group.attrs['Re'] = (source_attrs.get('Uinf', 0.35199999809265137) *
                                 source_attrs.get('delta99', 0.04490000009536743) /
                                 source_attrs.get('nu', 7.878999781496532e-07))

            print(f"      完成: shape = {velocity_data.shape}")


def generate_grid_and_topology(source_h5: h5py.File, x_truncate: int, y_truncate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, int, int, int]:
    """
    根据截断参数生成网格坐标、mesh、节点类型等拓扑信息

    Args:
        source_h5: 源HDF5文件对象
        x_truncate: x方向截断宽度
        y_truncate: y方向截断高度

    Returns:
        pos: 节点位置数组 (n_nodes, 2)
        mesh: 网格连接数组 (n_edges, 2)
        node_type: 节点类型字典
        height: 含wall行的总高度
        width: 宽度
        n_nodes: 节点总数
    """
    # 读取网格坐标并截断
    grid_x = source_h5['grid_x'][:]
    grid_y = source_h5['grid_y'][:]
    grid_x, grid_y = truncate_grid_coordinates(
        grid_x, grid_y, x_truncate, y_truncate)

    # 添加wall边界行（y=0, U=V=0）
    grid_x, grid_y = add_wall_boundary_to_grid(grid_x, grid_y)
    height = grid_x.shape[0]  # raw_height + 1（wall行）
    width = grid_x.shape[1]
    n_nodes = height * width

    # 生成节点位置信息
    pos = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)

    # 生成网格结构
    mesh = create_mesh_structure(height, width)

    # 生成节点类型
    node_type = create_node_types(height, width)

    return pos, mesh, node_type, height, width, n_nodes


def generate_bl_data():
    """主函数：生成边界层数据集"""
    # 数据路径配置
    source_file = Path("/home/wqx/projects/CFD-paradigm/data/Data_1mm.h5")
    output_dir = Path("/home/wqx/projects/PhyMPGN/data/2d_bl")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 训练集配置 =====
    TRAIN_GROUPS = [1, 2]              # 训练集组索引
    TRAIN_X_TRUNCATE = 50              # 训练集x方向截断宽度
    TRAIN_Y_TRUNCATE = 160              # 训练集y方向截断高度
    TRAIN_T_TRUNCATE = 2000            # 训练集时间步截断数量

    # ===== 测试集配置 =====
    TEST_GROUPS = [2,]      # 测试集组索引
    TEST_X_TRUNCATE = 50               # 测试集x方向截断宽度
    TEST_Y_TRUNCATE = 160               # 测试集y方向截断高度
    TEST_T_TRUNCATE = 4000             # 测试集时间步截断数量

    print(f"📖 打开源数据文件: {source_file}")
    with h5py.File(source_file, 'r') as source_h5:
        # 读取元数据
        source_attrs = dict(source_h5.attrs)
        U_shape = source_h5['U'].shape

        print(f"   原始U shape: {U_shape}")
        n_groups, n_timesteps_original, original_height, original_width = U_shape
        print(
            f"   原始: {n_groups}组, {n_timesteps_original}时间步, {original_height}x{original_height}网格")

        # ===== 生成训练数据 =====
        print(f"\n{'='*50}")
        print(f"📋 生成训练数据")
        print(f"{'='*50}")

        pos_tr, mesh_tr, node_type_tr, height_tr, width_tr, n_nodes_tr = generate_grid_and_topology(
            source_h5, TRAIN_X_TRUNCATE, TRAIN_Y_TRUNCATE
        )
        n_timesteps_tr = TRAIN_T_TRUNCATE if TRAIN_T_TRUNCATE is not None else n_timesteps_original

        print(f"   组索引: {TRAIN_GROUPS}")
        print(f"   时间步: {n_timesteps_original} -> {n_timesteps_tr}")
        print(f"   网格尺寸: {height_tr} x {width_tr} (含wall行)")
        print(f"   节点总数: {n_nodes_tr}")
        for name, indices in node_type_tr.items():
            print(f"   {name}节点数: {len(indices)}")

        train_file = output_dir / \
            f"train_bl_{len(TRAIN_GROUPS)}x{n_timesteps_tr}x{n_nodes_tr}x2.h5"

        generate_bl_data_file(
            source_file, train_file, TRAIN_GROUPS, source_h5,
            pos_tr, mesh_tr, node_type_tr, source_attrs,
            TRAIN_X_TRUNCATE, TRAIN_Y_TRUNCATE, TRAIN_T_TRUNCATE
        )

        # ===== 生成测试数据 =====
        print(f"\n{'='*50}")
        print(f"📋 生成测试数据")
        print(f"{'='*50}")

        pos_te, mesh_te, node_type_te, height_te, width_te, n_nodes_te = generate_grid_and_topology(
            source_h5, TEST_X_TRUNCATE, TEST_Y_TRUNCATE
        )
        n_timesteps_te = TEST_T_TRUNCATE if TEST_T_TRUNCATE is not None else n_timesteps_original

        print(f"   组索引: {TEST_GROUPS}")
        print(f"   时间步: {n_timesteps_original} -> {n_timesteps_te}")
        print(f"   网格尺寸: {height_te} x {width_te} (含wall行)")
        print(f"   节点总数: {n_nodes_te}")
        for name, indices in node_type_te.items():
            print(f"   {name}节点数: {len(indices)}")

        test_file = output_dir / \
            f"test_bl_{len(TEST_GROUPS)}x{n_timesteps_te}x{n_nodes_te}x2.h5"

        generate_bl_data_file(
            source_file, test_file, TEST_GROUPS, source_h5,
            pos_te, mesh_te, node_type_te, source_attrs,
            TEST_X_TRUNCATE, TEST_Y_TRUNCATE, TEST_T_TRUNCATE
        )

    print(f"\n✅ 数据生成完成!")
    print(f"   训练数据: {train_file}")
    print(f"   测试数据: {test_file}")


if __name__ == "__main__":
    generate_bl_data()
