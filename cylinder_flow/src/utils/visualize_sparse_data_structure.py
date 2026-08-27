#!/usr/bin/env python3
"""
边界层稀疏数据可视化脚本
验证生成的H5文件中节点类型和速度场分布
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors


def plot_sparse_data_structure(h5_file_path: str, show_velocity_background: bool = True):
    """
    绘制边界层稀疏数据的空间结构示意图

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
        group = f['0']
        velocity_data = group['U'][:]  # shape: (timesteps, n_nodes, 2)
        first_timestep_velocity = velocity_data[0]  # 第一个时刻的速度场

        # 读取网格信息
        n_nodes = pos.shape[0]
        print(f"节点总数: {n_nodes}")
        for name, indices in node_types.items():
            print(f"{name}节点数: {len(indices)}")

    # 计算速度大小
    velocity_magnitude = np.sqrt(first_timestep_velocity[:, 0]**2 + first_timestep_velocity[:, 1]**2)

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
        ax.set_title(f'Boundary Layer Sparse Data Structure (with Velocity Background) - Timestep 1')
    else:
        ax.set_title(f'Boundary Layer Sparse Data Structure (without Background) - Timestep 1')

    # Define node type colors
    color_map = {
        'inner': 'lightgreen',
        'exp': 'red',
        'wall': 'blue'
    }

    # Plot points for each node type
    for node_type, indices in node_types.items():
        if len(indices) > 0:
            node_positions = pos[indices]
            ax.scatter(node_positions[:, 0], node_positions[:, 1],
                      c=color_map.get(node_type, 'gray'),
                      s=30,
                      marker='o',
                      edgecolors='black',
                      linewidths=0.5,
                      label=f'{node_type} ({len(indices)})',
                      zorder=5)  # Ensure above background

    # Set aspect ratio to 1:1
    ax.set_aspect('equal')

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Set axis labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax


def plot_multiple_files(h5_files: list, show_velocity_background: bool = True):
    """
    绘制多个H5文件的空间结构

    Args:
        h5_files: H5文件路径列表
        show_velocity_background: 是否显示速度场背景
    """
    n_files = len(h5_files)
    fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 6))

    if n_files == 1:
        axes = [axes]

    for idx, h5_file in enumerate(h5_files):
        with h5py.File(h5_file, 'r') as f:
            # 读取图结构信息
            pos = f['pos'][:]

            # 读取节点类型
            node_type_group = f['node_type']
            node_types = {}
            for name in node_type_group.keys():
                node_types[name] = node_type_group[name][:]

            # 读取速度场数据（第一个时刻）
            group = f['0']
            velocity_data = group['U'][:]
            first_timestep_velocity = velocity_data[0]

        # 计算速度大小
        velocity_magnitude = np.sqrt(first_timestep_velocity[:, 0]**2 + first_timestep_velocity[:, 1]**2)

        ax = axes[idx]

        # Plot velocity field background
        if show_velocity_background:
            scatter = ax.scatter(pos[:, 0], pos[:, 1],
                               c=velocity_magnitude,
                               cmap='RdBu_r',
                               s=50,
                               alpha=0.3,
                               vmin=-velocity_magnitude.max(),
                               vmax=velocity_magnitude.max())
            plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')

        # Define node type colors
        color_map = {
            'inner': 'lightgreen',
            'exp': 'red',
            'wall': 'blue'
        }

        # Plot points for each node type
        for node_type, indices in node_types.items():
            if len(indices) > 0:
                node_positions = pos[indices]
                ax.scatter(node_positions[:, 0], node_positions[:, 1],
                          c=color_map.get(node_type, 'gray'),
                          s=30,
                          marker='o',
                          edgecolors='black',
                          linewidths=0.5,
                          label=f'{node_type} ({len(indices)})',
                          zorder=5)

        # Set aspect ratio to 1:1
        ax.set_aspect('equal')

        # Add legend
        ax.legend(loc='upper right', fontsize=8)

        # Set title
        file_name = Path(h5_file).name
        ax.set_title(f'{file_name}\n({len(node_types)} Node Types)')

        # Set axis labels
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Add grid
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def main():
    """主函数：可视化边界层稀疏数据"""
    # 数据文件路径
    data_dir = Path("/home/wqx/projects/PhyMPGN/data/2d_bl_sparse_7070")

    train_file = data_dir / "train_bl_sparse_2x2000x7070x2.h5"
    test_file = data_dir / "test_bl_sparse_1x4000x7070x2.h5"

    # 检查文件是否存在
    if not train_file.exists():
        print(f"训练数据文件不存在: {train_file}")
        return

    if not test_file.exists():
        print(f"测试数据文件不存在: {test_file}")
        return

    print("📊 开始可视化边界层稀疏数据...")

    # 创建输出目录
    output_dir = data_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # 绘制训练数据（带速度场背景）
    print("\n绘制训练数据（带速度场背景）...")
    fig_train_with_bg, ax_train_with_bg = plot_sparse_data_structure(
        str(train_file), show_velocity_background=True)
    fig_train_with_bg.savefig(output_dir / "train_structure_with_velocity.png", dpi=150)
    print(f"保存到: {output_dir / 'train_structure_with_velocity.png'}")
    plt.close(fig_train_with_bg)

    # 绘制训练数据（不带速度场背景）
    print("\n绘制训练数据（不带速度场背景）...")
    fig_train_no_bg, ax_train_no_bg = plot_sparse_data_structure(
        str(train_file), show_velocity_background=False)
    fig_train_no_bg.savefig(output_dir / "train_structure_no_background.png", dpi=150)
    print(f"保存到: {output_dir / 'train_structure_no_background.png'}")
    plt.close(fig_train_no_bg)

    # 绘制测试数据（带速度场背景）
    print("\n绘制测试数据（带速度场背景）...")
    fig_test_with_bg, ax_test_with_bg = plot_sparse_data_structure(
        str(test_file), show_velocity_background=True)
    fig_test_with_bg.savefig(output_dir / "test_structure_with_velocity.png", dpi=150)
    print(f"保存到: {output_dir / 'test_structure_with_velocity.png'}")
    plt.close(fig_test_with_bg)

    # 绘制测试数据（不带速度场背景）
    print("\n绘制测试数据（不带速度场背景）...")
    fig_test_no_bg, ax_test_no_bg = plot_sparse_data_structure(
        str(test_file), show_velocity_background=False)
    fig_test_no_bg.savefig(output_dir / "test_structure_no_background.png", dpi=150)
    print(f"保存到: {output_dir / 'test_structure_no_background.png'}")
    plt.close(fig_test_no_bg)

    # 绘制对比图
    print("\n绘制训练和测试数据对比图...")
    fig_comparison, axes_comparison = plot_multiple_files(
        [str(train_file), str(test_file)], show_velocity_background=True)
    fig_comparison.savefig(output_dir / "train_test_comparison.png", dpi=150)
    print(f"保存到: {output_dir / 'train_test_comparison.png'}")
    plt.close(fig_comparison)

    print(f"\n✅ 可视化完成！所有图片保存在: {output_dir}")


if __name__ == "__main__":
    main()