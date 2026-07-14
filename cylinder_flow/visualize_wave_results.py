"""
PhyMPGN波场预测云图可视化
生成真实数据vs预测数据的波场对比云图
"""

import matplotlib
import yaml
from src.models.model import WaveModel
from src.datasets.dataset import WaveGraphDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.tri import Triangulation

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


class WaveFieldVisualizer:
    def __init__(self, work_path='.'):
        self.work_path = work_path
        # work_path 是项目根目录，即 PhyMPGN/
        self.project_root = os.path.abspath(work_path)  # PhyMPGN的绝对路径
        self.cylinder_flow_dir = os.path.join(self.project_root, 'cylinder_flow')
        self.data_dir = os.path.join(self.project_root, 'data/wave')
        self.config_dir = os.path.join(self.cylinder_flow_dir, 'configs')
        # 检查点目录根据配置文件：/home/wqx/projects/PhyMPGN/ckpts
        self.ckpt_dir = os.path.join(self.project_root, 'ckpts')
        # 创建专门的波场可视化子文件夹
        self.visualization_dir = os.path.join(
            self.project_root, 'visualization', 'wave_results')

        # 确保可视化目录存在
        os.makedirs(self.visualization_dir, exist_ok=True)

        print(f"项目根目录: {self.project_root}")
        print(f"cylinder_flow目录: {self.cylinder_flow_dir}")
        print(f"数据目录: {self.data_dir}")
        print(f"配置目录: {self.config_dir}")
        print(f"检查点目录: {self.ckpt_dir}")
        print(f"可视化结果目录: {self.visualization_dir}")

    def load_model_and_data(self):
        """加载训练好的模型和测试数据"""
        print("正在加载模型和数据...")

        # 读取配置文件
        config_path = os.path.join(self.config_dir, 'train_wave.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 加载测试数据
        te_dataset = WaveGraphDataset(
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
        model = WaveModel(
            encoder_config=config['encoder_config'],
            mpnn_block_config=config['mpnn_block_config'],
            decoder_config=config['decoder_config'],
            laplace_block_config=config['laplace_block_config'],
            dtype=torch.float32,
            device='cuda',
            integral=config['integral']
        )

        # 加载模型权重
        full_ckpt_path = os.path.join(self.ckpt_dir, 'wave_ckpt-wave_expr_0-val')
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

    def plot_wave_comparison(self, truth_wave, pred_wave, error_wave, abs_error,
                            time_idx, pos, face, save_path=None, global_ranges=None):
        """
        绘制波场对比图 - 1行3列布局

        布局:
        第1列: 真实波场
        第2列: 预测波场
        第3列: 绝对误差

        Args:
            truth_wave: 真实波场高度 [n,]
            pred_wave: 预测波场高度 [n,]
            error_wave: 误差波场 [n,]
            abs_error: 绝对误差 [n,]
            time_idx: 时间步索引
            pos: 节点位置 [n, 2]
            face: 三角形面 [m, 3]
            save_path: 保存路径
            global_ranges: 全局colorbar范围
        """
        # 创建三角网格
        tri = Triangulation(pos[:, 0], pos[:, 1], face)

        # 创建1行3列图像网格
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'波方程预测对比 - 时间步 t={time_idx}',
                     fontsize=16, fontweight='bold')

        # 计算颜色映射范围
        if global_ranges is not None:
            v_min_wave, v_max_wave = global_ranges['wave_field']
            max_error = global_ranges['error'][1]
        else:
            v_min_wave = min(truth_wave.min(), pred_wave.min())
            v_max_wave = max(truth_wave.max(), pred_wave.max())
            max_error = abs_error.max()

        # ===== 第1列：真实波场 =====
        plot1 = axes[0].tripcolor(tri, truth_wave, shading='gouraud', cmap='jet',
                                   vmin=v_min_wave, vmax=v_max_wave)
        axes[0].set_title('真实波场', fontsize=12, fontweight='bold')
        axes[0].set_aspect('equal')
        axes[0].axis('off')
        cbar1 = plt.colorbar(plot1, ax=axes[0], shrink=0.7)
        cbar1.set_label('波场高度', rotation=270, labelpad=15, fontsize=10)

        # ===== 第2列：预测波场 =====
        plot2 = axes[1].tripcolor(tri, pred_wave, shading='gouraud', cmap='jet',
                                   vmin=v_min_wave, vmax=v_max_wave)
        axes[1].set_title('预测波场', fontsize=12, fontweight='bold')
        axes[1].set_aspect('equal')
        axes[1].axis('off')
        cbar2 = plt.colorbar(plot2, ax=axes[1], shrink=0.7)
        cbar2.set_label('波场高度', rotation=270, labelpad=15, fontsize=10)

        # ===== 第3列：绝对误差 =====
        plot3 = axes[2].tripcolor(tri, abs_error, shading='gouraud', cmap='hot',
                                   vmin=0, vmax=max_error)
        axes[2].set_title('绝对误差 |预测-真实|', fontsize=12, fontweight='bold')
        axes[2].set_aspect('equal')
        axes[2].axis('off')
        cbar3 = plt.colorbar(plot3, ax=axes[2], shrink=0.7)
        cbar3.set_label('绝对误差', rotation=270, labelpad=15, fontsize=10)

        # 调整子图间距
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90,
                            bottom=0.12, wspace=0.15)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存到: {save_path}")

        plt.close()
        return fig

    def create_time_series_visualization(self, pred_sequence, truth_sequence, batch,
                                         save_dir=None, time_steps=None):
        """
        创建时序可视化（波方程版本）

        Args:
            pred_sequence: 预测序列 [t+1, n, 1]
            truth_sequence: 真实序列 [t+1, n, 1]
            batch: 图数据batch对象
            save_dir: 保存目录，如果为None则使用默认绝对路径
            time_steps: 要可视化的时间步列表，如果为None则自动选择
        """
        print("正在生成波场对比云图...")

        # 使用绝对路径
        if save_dir is None:
            save_dir = self.visualization_dir
        else:
            save_dir = os.path.abspath(save_dir)

        print(f"保存目录: {save_dir}")

        # 获取网格信息（从batch中提取）
        pos = batch.pos.numpy()  # 节点位置 [n, 2]

        # 处理三角形面
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

        # 计算所有时间步的全局min/max（波方程只有1维波场高度）
        all_truth_u = truth_sequence[:, :, 0]  # [t, n]
        all_pred_u = pred_sequence[:, :, 0]    # [t, n]
        all_error = all_pred_u - all_truth_u   # [t, n]
        all_error_abs = np.abs(all_error)      # [t, n]

        # 计算全局范围
        v_min = min(all_truth_u.min(), all_pred_u.min())
        v_max = max(all_truth_u.max(), all_pred_u.max())
        max_error = all_error_abs.max()

        global_ranges = {
            'wave_field': (v_min, v_max),
            'error': (0, max_error)
        }

        print(f"全局colorbar范围:")
        print(f"  波场高度: [{v_min:.6f}, {v_max:.6f}]")
        print(f"  绝对误差: [0, {max_error:.6f}]")

        # 为每个时间步生成对比图（使用统一的全局范围）
        for t_idx in time_steps:
            print(f"处理时间步 {t_idx}/{total_steps-1}...")

            truth_wave = truth_sequence[t_idx, :, 0]  # [n,]
            pred_wave = pred_sequence[t_idx, :, 0]    # [n,]
            error_wave = pred_wave - truth_wave      # [n,]
            abs_error = np.abs(error_wave)           # [n,]

            # 生成综合对比图，传入全局colorbar范围
            save_path = os.path.join(
                save_dir, f'wave_comparison_t{t_idx:04d}.png')
            self.plot_wave_comparison(truth_wave, pred_wave, error_wave, abs_error,
                                      t_idx, pos, face, save_path,
                                      global_ranges=global_ranges)

            print(f"已生成时间步 {t_idx} 的综合对比图")


def main():
    """主函数"""
    print("=" * 50)
    print("PhyMPGN 波场预测云图可视化")
    print("=" * 50)

    # 创建可视化器 - work_path 指向项目根目录
    # 文件位于 cylinder_flow/，需要向上两级到达项目根目录
    file_path = os.path.abspath(__file__)
    cylinder_flow_dir = os.path.dirname(file_path)
    project_root = os.path.dirname(cylinder_flow_dir)
    visualizer = WaveFieldVisualizer(work_path=project_root)

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
