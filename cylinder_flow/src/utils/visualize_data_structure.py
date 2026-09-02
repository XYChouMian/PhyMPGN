#!/usr/bin/env python3
"""
通用数据可视化脚本
支持可视化H5文件中的节点类型和速度场分布
自适应不同的节点类型，适用于各种图结构数据
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors


def plot_data_structure(
    h5_file_path: str,
    show_velocity_background: bool = True,
    dataset_idx: int = 0
):
    """
    绘制H5文件数据的空间结构示意图

    Args:
        h5_file_path: H5文件路径
        show_velocity_background: 是否显示速度场背景，True=显示，False=隐藏
    """
    with h5py.File(h5_file_path, 'r') as f:
        # 读取图结构信息
        pos = f['pos'][:]
        mesh = f['mesh'][:]

        # 读取节点类型
        node_type_group = f['node_type']
        node_types = {}
        for name in node_type_group.keys():
            node_types[name] = node_type_group[name][:]

        # 读取速度场数据（第一个时刻）
        group = f[str(dataset_idx)]
        velocity_data = group['U'][:]  # shape: (timesteps, n_nodes, 2)
        first_timestep_velocity = velocity_data[0]  # 第一个时刻的速度场

        # 读取网格信息
        n_nodes = pos.shape[0]
        print(f"Total nodes: {n_nodes}")
        for name, indices in node_types.items():
            print(f"{name} nodes: {len(indices)}")

    # 计算速度大小
    try:
        velocity_magnitude = np.sqrt(
            first_timestep_velocity[:, 0]**2 + first_timestep_velocity[:, 1]**2)
    except:
        velocity_magnitude = first_timestep_velocity[:, 0]

    # 设置画布
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制速度场背景（如果启用）
    if show_velocity_background:
        scatter = ax.scatter(pos[:, 0], pos[:, 1],
                             c=velocity_magnitude,
                             cmap='RdBu_r',
                             s=50,
                             alpha=0.3,
                             vmin=-velocity_magnitude.max(),
                             vmax=velocity_magnitude.max())
        plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
        file_name = Path(h5_file_path).name
        ax.set_title(
            f'Data Structure (with Velocity Background) - {file_name}')
    else:
        file_name = Path(h5_file_path).name
        ax.set_title(f'Data Structure (without Background) - {file_name}')

    # 自动定义颜色映射（支持任意数量的节点类型）
    predefined_colors = ['red', 'blue', 'green', 'orange',
                         'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
    color_map = {}
    for i, node_type_name in enumerate(node_types.keys()):
        color_map[node_type_name] = predefined_colors[i %
                                                      len(predefined_colors)]

    # 计算总节点数
    total_nodes = n_nodes

    # 绘制每个节点类型的点（排除超过50%的类型）
    for node_type_name, indices in node_types.items():
        if len(indices) > 0:
            # 如果该类型节点数量超过总节点数的50%，则跳过
            if len(indices) > total_nodes * 0.5:
                print(
                    f"Skipping {node_type_name} (too many nodes: {len(indices)} > {total_nodes * 0.5})")
                continue

            node_positions = pos[indices]
            ax.scatter(node_positions[:, 0], node_positions[:, 1],
                       c=color_map.get(node_type_name, 'gray'),
                       s=30,
                       marker='o',
                       edgecolors='black',
                       linewidths=0.5,
                       label=f'{node_type_name} ({len(indices)})',
                       zorder=5)

    # 设置坐标轴比例为1:1
    ax.set_aspect('equal')

    # 添加图例
    ax.legend(loc='upper right', fontsize=10)

    # 设置坐标轴标签
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # 添加网格
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax


def visualize_directory(data_dir: str, show_velocity_background: bool = True):
    """
    可视化指定目录中的所有H5文件

    Args:
        data_dir: 数据目录路径
        show_velocity_background: 是否显示速度场背景
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory does not exist: {data_dir}")
        return

    # 自动识别所有H5文件
    h5_files = list(data_path.glob("*.h5"))
    if not h5_files:
        print(f"No H5 files found in: {data_dir}")
        return

    print(f"Found {len(h5_files)} H5 files")

    # 创建输出目录
    output_dir = data_path / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # 逐个文件处理
    for h5_file in h5_files:
        with h5py.File(str(h5_file), 'r') as f:
            if 'num_dataset' in f.attrs:
                num_dataset = f.attrs['num_dataset']
            else:
                num_dataset = 1

        for dataset_idx in range(num_dataset):
            # 绘制数据结构
            fig, ax = plot_data_structure(
                str(h5_file),
                show_velocity_background=show_velocity_background,
                dataset_idx=dataset_idx
            )

            # 保存图片
            output_file = output_dir / f"{h5_file.stem}_{dataset_idx}.png"
            fig.savefig(output_file, dpi=150)
            print(f"Saved to: {output_file}")

            plt.close(fig)

    print(f"\nVisualization complete! All images saved to: {output_dir}")


def main():
    """主函数：可视化数据目录中的H5文件"""
    # 数据目录路径（可通过命令行参数或修改此处配置）
    # data_dir = "/home/wqx/projects/PhyMPGN/data/2d_bl_sparse_7070"
    data_dir = "/home/wqx/projects/PhyMPGN/data/2d_bl_sparse_7070_multi"
    # data_dir = "/home/wqx/projects/PhyMPGN/data/2d_bl_7100"
    # data_dir = "/home/wqx/projects/PhyMPGN/data/2d_bl_8050"
    # data_dir = "/home/wqx/projects/PhyMPGN/data/2d_cf"
    # data_dir = "/home/wqx/projects/PhyMPGN/data/wave"

    # 解析命令行参数
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    # 可视化数据
    visualize_directory(data_dir, show_velocity_background=True)


if __name__ == "__main__":
    main()
