"""
PhyMPGN预测结果云图可视化
生成真实数据vs预测数据的速度场对比云图
"""

import matplotlib
import yaml
from src.models.model import Model
from src.datasets.dataset import PDEGraphDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from scipy.interpolate import CloughTocher2DInterpolator
from matplotlib.tri import Triangulation
import matplotlib.tri as mtri

# 添加src到路径
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei',
                                   'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
# 解决上标字符显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # 使用CM字体渲染数学公式


class VelocityFieldVisualizer:
    def __init__(self, work_path='.'):
        self.work_path = work_path
        # 使用绝对路径
        self.project_root = os.path.abspath(work_path)
        self.ckpt_dir = os.path.join(self.project_root, 'ckpts')
        self.data_dir = os.path.join(self.project_root, 'data/2d_cf')
        self.config_dir = os.path.join(work_path, 'cylinder_flow/configs')
        # 创建专门的速度场可视化子文件夹
        self.visualization_dir = os.path.join(
            self.project_root, 'visualization', 'velocity_comparison')

        # 确保可视化目录存在
        os.makedirs(self.visualization_dir, exist_ok=True)

        print(f"项目根目录: {self.project_root}")
        print(f"可视化结果目录: {self.visualization_dir}")

    def load_model_and_data(self):
        """加载训练好的模型和测试数据"""
        print("正在加载模型和数据...")

        # 读取配置文件
        config_path = os.path.join(self.config_dir, 'train.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 加载测试数据
        te_dataset = PDEGraphDataset(
            root=self.data_dir,
            raw_files=config['te_raw_data'],
            processed_file=config['te_processed_file'],
            dataset_start=config['te_dataset_start'],
            dataset_used=1,  # 只加载1个样本用于可视化
            time_start=config['time_start'],
            time_used=2000,  # 全部时间步
            window_size=2000,  # 完整序列
            dtype=torch.float32,
            training=False
        )

        # 构建模型
        model = Model(
            encoder_config=config['encoder_config'],
            mpnn_block_config=config['mpnn_block_config'],
            decoder_config=config['decoder_config'],
            laplace_block_config=config['laplace_block_config'],
            dtype=torch.float32,
            device='cuda',
            integral=config['integral']
        )

        # 加载模型权重
        full_ckpt_path = os.path.join(self.ckpt_dir, 'ckpt-0-val')
        if os.path.exists(os.path.join(full_ckpt_path, 'model.safetensors')):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(
                full_ckpt_path, 'model.safetensors'))
            model.load_state_dict(state_dict)
            print(f"成功加载模型权重: {full_ckpt_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {full_ckpt_path}")

        # 设置为评估模式并移动到GPU
        model.eval()
        model = model.cuda()

        return model, te_dataset, config

    def predict_velocity_field(self, model, dataset):
        """预测速度场"""
        print("正在进行速度场预测...")
        model.eval()

        with torch.no_grad():
            # 创建DataLoader来处理batch
            from torch_geometric.loader import DataLoader
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

            # 获取第一个batch
            for batch in test_loader:
                # 确保数据在GPU上
                batch = batch.to('cuda')

                # 按照train.py的方式准备数据
                target = batch.y.transpose(0, 1)  # [t, n, 2]
                batch.y = target[0]  # [n, 2] 初始时刻

                steps = target.shape[0] - 1  # 预测步数

                # 进行预测
                pred_sequence = [batch.y.clone()]
                for step in range(steps):
                    if step % 100 == 0:
                        print(f"正在预测时间步 {step}/{steps}...")

                    pred = model(batch, steps=1)  # [t+1, bxn, 2]
                    batch.y = pred[-1]  # 使用最后一个时间步
                    pred_sequence.append(batch.y.clone())

                # 转换为numpy数组（先移回CPU）
                pred_sequence = torch.stack(
                    pred_sequence, dim=0)  # [t+1, n, 2]
                truth_sequence = target  # [t+1, n, 2]

                # 只取第一个batch，然后移到CPU转换为numpy
                # 取前1598个节点（不包括幽灵节点）
                pred_sequence = pred_sequence.cpu().numpy()
                truth_sequence = truth_sequence.cpu().numpy()

                # batch也移到CPU用于后续可视化
                batch = batch.cpu()

                break

        return pred_sequence, truth_sequence, batch

    def compute_velocity_magnitude(self, velocity_field):
        """计算速度大小"""
        u = velocity_field[:, :, 0]
        v = velocity_field[:, :, 1]
        magnitude = np.sqrt(u**2 + v**2)
        return magnitude

    def plot_comprehensive_comparison(self, truth_u, pred_u, truth_v, pred_v, time_idx,
                                      pos, face, save_path=None,
                                      global_ranges=None):
        """
        绘制综合对比图 - 3行3列布局

        布局:
        第1行: u真实, u预测, u误差
        第2行: v真实, v预测, v误差
        第3行: 速度大小真实, 速度大小预测, 总误差

        Args:
            truth_u: 真实速度u分量
            pred_u: 预测速度u分量  
            truth_v: 真实速度v分量
            pred_v: 预测速度v分量
            time_idx: 时间步索引
            pos: 节点位置
            face: 三角形面
            save_path: 保存路径
            global_ranges: 全局colorbar范围，用于所有时间步保持一致
        """
        # 计算速度大小和误差
        truth_mag = np.sqrt(truth_u**2 + truth_v**2)
        pred_mag = np.sqrt(pred_u**2 + pred_v**2)

        error_u = pred_u - truth_u
        error_v = pred_v - truth_v
        error_mag = np.sqrt(error_u**2 + error_v**2)

        # 创建三角网格
        tri = Triangulation(pos[:, 0], pos[:, 1], face)
        # 创建3x3图像网格
        fig, axes = plt.subplots(3, 3, figsize=(20, 10))
        fig.suptitle(f'时间步 t={time_idx} PhyMPGN 速度场预测综合对比',
                     fontsize=18, fontweight='bold')

        # 计算颜色映射范围（使用全局范围或当前图的范围）
        if global_ranges is not None:
            # 使用全局统一范围
            v_min_u, v_max_u = global_ranges['u']
            v_min_v, v_max_v = global_ranges['v']
            v_min_mag, v_max_mag = global_ranges['mag']
            max_error_u, max_error_v = global_ranges['error']
            max_error_mag = global_ranges['error_mag']
        else:
            # 计算当前图的范围（单个图独立范围）
            v_min_u = min(truth_u.min(), pred_u.min())
            v_max_u = max(truth_u.max(), pred_u.max())
            v_min_v = min(truth_v.min(), pred_v.min())
            v_max_v = max(truth_v.max(), pred_v.max())
            v_min_mag = min(truth_mag.min(), pred_mag.min())
            v_max_mag = max(truth_mag.max(), pred_mag.max())
            max_error_u = max(abs(error_u.min()), abs(error_u.max()))
            max_error_v = max(abs(error_v.min()), abs(error_v.max()))

        # ===== 第1行：u分量 =====

        # u真实值
        plot1 = axes[0, 0].tripcolor(tri, truth_u, shading='gouraud', cmap='jet',
                                     vmin=v_min_u, vmax=v_max_u)
        axes[0, 0].set_title('u分量 (真实)', fontsize=12, fontweight='bold')
        axes[0, 0].set_aspect('equal')
        axes[0, 0].axis('off')
        plt.colorbar(plot1, ax=axes[0, 0], shrink=0.6)

        # u预测值
        plot2 = axes[0, 1].tripcolor(tri, pred_u, shading='gouraud', cmap='jet',
                                     vmin=v_min_u, vmax=v_max_u)
        axes[0, 1].set_title('u分量 (预测)', fontsize=12, fontweight='bold')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].axis('off')
        plt.colorbar(plot2, ax=axes[0, 1], shrink=0.6)

        # u误差
        plot3 = axes[0, 2].tripcolor(tri, error_u, shading='gouraud', cmap='seismic',
                                     vmin=-max_error_u, vmax=max_error_u)
        axes[0, 2].set_title('u分量误差 (预测-真实)', fontsize=12, fontweight='bold')
        axes[0, 2].set_aspect('equal')
        axes[0, 2].axis('off')
        plt.colorbar(plot3, ax=axes[0, 2], shrink=0.6)

        # ===== 第2行：v分量 =====

        # v真实值
        plot4 = axes[1, 0].tripcolor(tri, truth_v, shading='gouraud', cmap='jet',
                                     vmin=v_min_v, vmax=v_max_v)
        axes[1, 0].set_title('v分量 (真实)', fontsize=12, fontweight='bold')
        axes[1, 0].set_aspect('equal')
        axes[1, 0].axis('off')
        plt.colorbar(plot4, ax=axes[1, 0], shrink=0.6)

        # v预测值
        plot5 = axes[1, 1].tripcolor(tri, pred_v, shading='gouraud', cmap='jet',
                                     vmin=v_min_v, vmax=v_max_v)
        axes[1, 1].set_title('v分量 (预测)', fontsize=12, fontweight='bold')
        axes[1, 1].set_aspect('equal')
        axes[1, 1].axis('off')
        plt.colorbar(plot5, ax=axes[1, 1], shrink=0.6)

        # v误差
        plot6 = axes[1, 2].tripcolor(tri, error_v, shading='gouraud', cmap='seismic',
                                     vmin=-max_error_v, vmax=max_error_v)
        axes[1, 2].set_title('v分量误差 (预测-真实)', fontsize=12, fontweight='bold')
        axes[1, 2].set_aspect('equal')
        axes[1, 2].axis('off')
        plt.colorbar(plot6, ax=axes[1, 2], shrink=0.6)

        # ===== 第3行：速度大小 =====

        # 速度大小真实值
        plot7 = axes[2, 0].tripcolor(tri, truth_mag, shading='gouraud', cmap='jet',
                                     vmin=v_min_mag, vmax=v_max_mag)
        axes[2, 0].set_title('速度大小 |v| (真实)', fontsize=12, fontweight='bold')
        axes[2, 0].set_aspect('equal')
        axes[2, 0].axis('off')
        cbar7 = plt.colorbar(plot7, ax=axes[2, 0], shrink=0.6)
        cbar7.set_label('速度大小', rotation=270, labelpad=20, fontsize=10)

        # 速度大小预测值
        plot8 = axes[2, 1].tripcolor(tri, pred_mag, shading='gouraud', cmap='jet',
                                     vmin=v_min_mag, vmax=v_max_mag)
        axes[2, 1].set_title('速度大小 |v| (预测)', fontsize=12, fontweight='bold')
        axes[2, 1].set_aspect('equal')
        axes[2, 1].axis('off')
        cbar8 = plt.colorbar(plot8, ax=axes[2, 1], shrink=0.6)
        cbar8.set_label('速度大小', rotation=270, labelpad=20, fontsize=10)

        # 总误差大小
        plot9 = axes[2, 2].tripcolor(
            tri, error_mag, shading='gouraud', cmap='hot', vmin=0, vmax=max_error_mag)
        axes[2, 2].set_title('总误差大小 √(Δu²+Δv²)',
                             fontsize=12, fontweight='bold')
        axes[2, 2].set_aspect('equal')
        axes[2, 2].axis('off')
        cbar9 = plt.colorbar(plot9, ax=axes[2, 2], shrink=0.6)
        cbar9.set_label('误差大小', rotation=270, labelpad=20, fontsize=10)

        # 调整子图间距
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92,
                            bottom=0.08, wspace=0.15, hspace=0.15)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存到: {save_path}")

        plt.close()
        return fig

    def create_time_series_visualization(self, pred_sequence, truth_sequence, batch,
                                         save_dir=None, time_steps=None):
        """
        创建时序可视化

        Args:
            pred_sequence: 预测序列 [t+1, n, 2]
            truth_sequence: 真实序列 [t+1, n, 2]
            batch: 图数据batch对象
            save_dir: 保存目录，如果为None则使用默认绝对路径
            time_steps: 要可视化的时间步列表，如果为None则自动选择
        """
        print("正在生成速度场对比云图...")

        # 使用绝对路径
        if save_dir is None:
            save_dir = self.visualization_dir
        else:
            save_dir = os.path.abspath(save_dir)

        print(f"保存目录: {save_dir}")

        # 获取网格信息（从batch中提取）
        pos = batch.pos.numpy()  # 只取前1598个节点（不包括幽灵节点）

        # 处理三角形面，确保形状正确
        face_tensor = batch.face
        if face_tensor.dim() == 2:
            # [3, num_triangles] -> [num_triangles, 3]
            face = face_tensor.permute(1, 0).numpy()
        else:
            face = face_tensor.numpy()

        # 确定要可视化的时间步
        total_steps = pred_sequence.shape[0]
        if time_steps is None:
            # 自动选择均匀分布的时间步
            num_frames = min(50, total_steps)  # 最多50帧
            time_steps = np.linspace(0, total_steps-1, num_frames, dtype=int)

        print(f"将要可视化 {len(time_steps)} 个时间步: {time_steps}")

        # ===== 计算全局统一的 colorbar 范围 =====
        print("正在计算全局 colorbar 范围...")

        # 计算所有时间步的全局min/max
        all_truth_u = truth_sequence[:, :, 0]  # [t, n, 2]
        all_pred_u = pred_sequence[:, :, 0]
        all_truth_v = truth_sequence[:, :, 1]  # [t, n, 2]
        all_pred_v = pred_sequence[:, :, 1]
        all_truth_mag = np.sqrt(all_truth_u**2 + all_truth_v**2)  # [t, n]
        all_pred_mag = np.sqrt(all_pred_u**2 + all_pred_v**2)
        all_error_u = all_pred_u - all_truth_u  # [t, n]
        all_error_v = all_pred_v - all_truth_v
        all_error_mag = np.sqrt(all_error_u**2 + all_error_v**2)  # [t, n]

        # 计算全局范围
        v_min_u = min(all_truth_u.min(), all_pred_u.min())
        v_max_u = max(all_truth_u.max(), all_pred_u.max())
        v_min_v = min(all_truth_v.min(), all_pred_v.min())
        v_max_v = max(all_truth_v.max(), all_pred_v.max())
        v_min_mag = min(all_truth_mag.min(), all_pred_mag.min())
        v_max_mag = max(all_truth_mag.max(), all_pred_mag.max())
        max_error_u = max(abs(all_error_u.min()), abs(all_error_u.max()))
        max_error_v = max(abs(all_error_v.min()), abs(all_error_v.max()))
        max_error_mag = all_error_mag.max()

        global_ranges = {
            'u': (v_min_u, v_max_u),
            'v': (v_min_v, v_max_v),
            'mag': (0, v_max_mag),
            'error': (max(max_error_u, max_error_v), max(max_error_u, max_error_v)),
            'error_mag': max_error_mag
        }

        print(f"全局colorbar范围:")
        print(f"  u分量: [{v_min_u:.6f}, {v_max_u:.6f}]")
        print(f"  v分量: [{v_min_v:.6f}, {v_max_v:.6f}]")
        print(f"  速度大小: [{v_min_mag:.6f}, {v_max_mag:.6f}]")
        print(f"  误差: [0, {max_error_u:.6f}, 0, {max_error_v:.6f}]")

        # 为每个时间步生成对比图（使用统一的全局范围）
        for t_idx in time_steps:
            print(f"处理时间步 {t_idx}/{total_steps-1}...")

            truth_u = truth_sequence[t_idx, :, 0]
            pred_u = pred_sequence[t_idx, :, 0]
            truth_v = truth_sequence[t_idx, :, 1]
            pred_v = pred_sequence[t_idx, :, 1]

            # 生成综合对比图 (3行3列)，传入全局colorbar范围
            save_path = os.path.join(
                save_dir, f'velocity_comparison_t{t_idx:04d}.png')
            self.plot_comprehensive_comparison(truth_u, pred_u, truth_v, pred_v,
                                               t_idx, pos, face, save_path,
                                               global_ranges=global_ranges)

            print(f"已生成时间步 {t_idx} 的综合对比图")


def main():
    """主函数"""
    print("=" * 50)
    print("PhyMPGN 速度场预测云图可视化")
    print("=" * 50)

    # 创建可视化器
    visualizer = VelocityFieldVisualizer(work_path='.')

    # 加载模型和数据
    model, dataset, config = visualizer.load_model_and_data()

    # 进行预测
    pred_sequence, truth_sequence, batch = visualizer.predict_velocity_field(
        model, dataset)

    # 生成可视化结果（使用绝对路径）
    print(f"\n正在生成可视化结果到项目根目录")

    # 按照等间隔50步输出，生成40个时间步
    total_steps = 2000  # 总时间步数
    interval = 50      # 输出间隔
    # [0, 50, 100, 150, 200, ..., 1950] (40个时间步)
    time_steps = list(range(0, total_steps, interval))

    print(f"将按照{interval}步间隔生成可视化，共{len(time_steps)}个时间步: {time_steps}")

    # 不传递save_dir，使用默认的绝对路径
    visualizer.create_time_series_visualization(
        pred_sequence, truth_sequence, batch,
        save_dir=None,  # 使用默认绝对路径
        time_steps=time_steps
    )

    print("\n" + "=" * 50)
    print("可视化完成！")
    print(f"结果已保存到: {visualizer.visualization_dir}")
    print("=" * 50)


if __name__ == '__main__':
    print(os.getcwd())
    main()
