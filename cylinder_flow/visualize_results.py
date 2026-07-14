"""
PhyMPGN训练结果可视化程序
用于分析训练过程和模型性能
"""

import h5py
from matplotlib.tri import Triangulation
from scipy.interpolate import CloughTocher2DInterpolator
from src.utils.utils import correlation
from src.utils.draw_utils import plot_meshcolor_evolution, plot_meshcolor_single
from src.models.model import Model
from src.utils.utils import NodeType
from src.datasets.dataset import PDEGraphDataset
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

# 添加src到路径
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII字符代替Unicode负号
plt.rcParams['font.family'] = 'sans-serif'


class TrainingVisualizer:
    def __init__(self, work_path='./'):
        self.work_path = work_path
        # 使用绝对路径
        self.project_root = os.path.abspath(work_path)
        self.log_dir = os.path.join(self.project_root, 'runs')
        self.ckpt_dir = os.path.join(self.project_root, 'ckpts')
        self.data_dir = os.path.join(self.project_root, 'data/2d_cf')
        # 创建专门的训练结果可视化子文件夹
        self.visualization_dir = os.path.join(
            self.project_root, 'visualization', 'training_analysis')

        # 确保可视化目录存在
        os.makedirs(self.visualization_dir, exist_ok=True)

        print(f"项目根目录: {self.project_root}")
        print(f"可视化结果目录: {self.visualization_dir}")

    def read_tensorboard_logs(self, log_dir):
        """
        读取TensorBoard日志文件

        Args:
            log_dir: 日志目录路径

        Returns:
            dict: 包含训练和验证损失的数据
        """
        # 查找所有事件文件
        event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
        if not event_files:
            print(f"在 {log_dir} 中没有找到TensorBoard日志文件")
            return None

        # 读取最新的事件文件
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={
                'scalars': 1000000000,  # 足够大的缓冲区
            }
        )
        ea.Reload()

        # 获取所有标量标签
        scalar_tags = ea.Tags()['scalars']
        print(f"找到的标量标签: {scalar_tags}")

        # 提取数据
        data = {}
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = {'steps': np.array(steps), 'values': np.array(values)}

        return data

    def plot_training_curves(self, save_dir=None):
        """
        绘制训练曲线（训练损失和验证损失）

        Args:
            save_dir: 保存目录，如果为None则不保存
        """
        # 读取训练日志
        train_data = self.read_tensorboard_logs(
            os.path.join(self.log_dir, 'expr_0_train'))
        val_data = self.read_tensorboard_logs(
            os.path.join(self.log_dir, 'expr_0_val'))

        if not train_data or not val_data:
            print("无法读取训练日志数据")
            return

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制训练损失
        for tag, data in train_data.items():
            ax[0].plot(data['steps'], data['values'], label=tag)
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')
        ax[0].legend()
        ax[0].set_yscale('log')
        ax[0].grid(True, alpha=0.3)

        # 绘制验证损失
        for tag, data in val_data.items():
            ax[1].plot(data['steps'], data['values'],
                       label=tag, marker='o', markersize=3)
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Validation Loss')
        ax[1].legend()
        ax[1].set_yscale('log')
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'training_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")

        plt.show()

        return train_data, val_data

    def plot_learning_rate(self, save_dir=None):
        """
        绘制学习率曲线
        """
        train_data = self.read_tensorboard_logs(
            os.path.join(self.log_dir, 'expr_0_train'))

        if not train_data:
            return

        # 查找学习率相关的标量
        lr_tags = [tag for tag in train_data.keys() if 'lr' in tag.lower()
                   or 'learning_rate' in tag.lower()]

        if not lr_tags:
            print("没有找到学习率数据")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for tag in lr_tags:
            data = train_data[tag]
            ax.plot(data['steps'], data['values'], label=tag)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'learning_rate.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"学习率曲线已保存到: {save_path}")

        plt.show()

    def load_model_and_data(self, ckpt_path='ckpt-0-val'):
        """
        加载训练好的模型和测试数据

        Args:
            ckpt_path: 模型检查点路径

        Returns:
            model: 加载的模型
            test_loader: 测试数据加载器
            config: 配置信息
        """
        # 加载测试数据
        te_dataset = PDEGraphDataset(
            root=self.data_dir,
            raw_files='test_cf_9x2000x1598x2.h5',
            processed_file='te_data.pt',
            dataset_start=0,
            dataset_used=9,
            time_start=0,
            time_used=2000,
            window_size=2000,
            dtype=torch.float32
        )

        test_loader = DataLoader(te_dataset, batch_size=1, shuffle=False)

        # 读取配置文件
        import yaml
        config_path = os.path.join(
            self.work_path, 'cylinder_flow/configs/train.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 构建模型
        model = Model(
            encoder_config=config['encoder_config'],
            mpnn_block_config=config['mpnn_block_config'],
            decoder_config=config['decoder_config'],
            laplace_block_config=config['laplace_block_config'],
            integral=config['integral']
        )

        # 加载模型权重
        full_ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
        if os.path.exists(os.path.join(full_ckpt_path, 'model.safetensors')):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(
                full_ckpt_path, 'model.safetensors'))
            model.load_state_dict(state_dict)
            print(f"成功加载模型权重: {full_ckpt_path}")
        else:
            print(f"警告: 模型文件不存在: {full_ckpt_path}")

        model.eval()

        return model, test_loader, config

    def visualize_velocity_field(self, model, test_loader, num_samples=3, save_dir=None):
        """
        可视化预测速度场与真实速度场的对比

        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            num_samples: 可视化的样本数量
            save_dir: 保存目录
        """
        model.eval()
        predictions = []
        truths = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_samples:
                    break

                # 准备输入数据
                graph = batch
                graph.y = batch.y[:, 0, :]  # 初始时刻
                steps = batch.y.shape[1] - 1  # 预测步数

                # 进行预测
                pred_sequence = [graph.y.clone()]
                for _ in range(steps):
                    pred = model(graph, steps=1)
                    graph.y = pred[:, 0, :]
                    pred_sequence.append(graph.y.clone())

                pred_sequence = torch.stack(
                    pred_sequence, dim=1)  # [1, t+1, n, 2]
                predictions.append(pred_sequence[0].numpy())
                truths.append(batch.y[0].numpy())

        # 获取网格信息用于可视化
        batch = next(iter(test_loader))
        coarse_pos = batch.pos[:1598].numpy()  # 前1598个节点（不包括幽灵节点）
        face = batch.face.permute(1, 0).numpy()  # 三角网格

        # 创建精细网格用于平滑可视化
        from scipy.spatial import Delaunay
        fine_tri = Delaunay(coarse_pos)
        fine_pos = fine_tri.points

        # 边界信息（简化处理）
        bdry_pos = coarse_pos[:10]  # 取前10个节点作为边界示例
        bdry_elem = [[i, (i+1) % 10] for i in range(10)]  # 简化边界元素

        # 可视化每个样本
        for i in range(min(num_samples, len(predictions))):
            pred_u = predictions[i]
            truth_u = truths[i]

            # 选择几个时间步进行可视化
            time_steps = [0, pred_u.shape[0]//4,
                          pred_u.shape[0]//2, pred_u.shape[0]-1]
            time_labels = ['初始时刻', '1/4时刻', '1/2时刻', '最终时刻']

            print(f"正在可视化样本 {i+1}，时间步: {time_steps}")

            if save_dir:
                sample_dir = os.path.join(save_dir, f'sample_{i+1}')
                os.makedirs(sample_dir, exist_ok=True)

            for t_idx, t in enumerate(time_steps):
                pred_step = pred_u[t]
                truth_step = truth_u[t]

                # 调用现有的可视化函数
                if save_dir:
                    save_path = os.path.join(sample_dir, f'pred_t{t}_{{}}.png')

                    try:
                        plot_meshcolor_single(
                            coarse_pos=coarse_pos,
                            coarse_U_pred=pred_step,
                            coarse_U_gt=truth_step,
                            fine_pos=fine_pos,
                            fine_tri=fine_tri.simplices,
                            bdry_pos=bdry_pos,
                            bdry_elem=bdry_elem,
                            save_path=save_path
                        )
                        print(
                            f"  生成图片: pred_t{t}_{{}}.png (时间步 {t}, {time_labels[t_idx]})")
                    except Exception as e:
                        print(f"可视化样本{i+1}时间步{t}时出错: {e}")
                        continue

    def analyze_prediction_errors(self, model, test_loader, save_dir=None):
        """
        分析预测误差

        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            save_dir: 保存目录
        """
        model.eval()
        all_errors = []
        all_correlations = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # 准备输入数据
                graph = batch
                graph.y = batch.y[:, 0, :]
                steps = batch.y.shape[1] - 1

                # 进行预测
                pred_sequence = [graph.y.clone()]
                for _ in range(steps):
                    pred = model(graph, steps=1)
                    graph.y = pred[:, 0, :]
                    pred_sequence.append(graph.y.clone())

                pred_sequence = torch.stack(pred_sequence, dim=1)

                # 计算误差
                truth = batch.y
                error = torch.mean((pred_sequence - truth) **
                                   2, dim=[2, 3])  # [batch, time]
                all_errors.append(error[0].numpy())

                # 计算相关性
                pred_flat = pred_sequence[0].reshape(-1)
                truth_flat = truth[0].reshape(-1)
                corr = correlation(pred_flat, truth_flat)
                all_correlations.append(corr)

                if i >= 4:  # 只分析前5个样本
                    break

        # 绘制误差随时间变化
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # 误差随时间变化
        time_steps = range(len(all_errors[0]))
        for i, errors in enumerate(all_errors):
            ax[0, 0].plot(time_steps, errors, label=f'Sample {i+1}')
        ax[0, 0].set_xlabel('Time Step')
        ax[0, 0].set_ylabel('MSE Error')
        ax[0, 0].set_title('Prediction Error Over Time')
        ax[0, 0].legend()
        ax[0, 0].set_yscale('log')
        ax[0, 0].grid(True, alpha=0.3)

        # 平均误差
        mean_error = np.mean(all_errors, axis=0)
        ax[0, 1].plot(time_steps, mean_error, 'r-',
                      linewidth=2, label='Mean Error')
        ax[0, 1].set_xlabel('Time Step')
        ax[0, 1].set_ylabel('Mean MSE Error')
        ax[0, 1].set_title('Average Prediction Error')
        ax[0, 1].set_yscale('log')
        ax[0, 1].grid(True, alpha=0.3)

        # 相关性
        ax[1, 0].bar(range(len(all_correlations)), all_correlations)
        ax[1, 0].set_xlabel('Sample')
        ax[1, 0].set_ylabel('Correlation')
        ax[1, 0].set_title('Prediction Correlation')
        ax[1, 0].set_ylim([0, 1])
        ax[1, 0].grid(True, alpha=0.3)

        # 误差分布
        all_errors_flat = np.array(all_errors).flatten()
        ax[1, 1].hist(all_errors_flat, bins=50, alpha=0.7)
        ax[1, 1].set_xlabel('MSE Error')
        ax[1, 1].set_ylabel('Frequency')
        ax[1, 1].set_title('Error Distribution')
        ax[1, 1].set_xscale('log')
        ax[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'error_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"误差分析已保存到: {save_path}")

        plt.show()

        # 打印统计信息
        print(f"\n预测误差统计:")
        print(f"平均误差: {np.mean(all_errors):.6e}")
        print(f"最小误差: {np.min(all_errors):.6e}")
        print(f"最大误差: {np.max(all_errors):.6e}")
        print(f"平均相关性: {np.mean(all_correlations):.4f}")

    def create_comprehensive_report(self, save_dir=None):
        """
        创建综合训练报告

        Args:
            save_dir: 保存目录，如果为None则使用默认绝对路径
        """
        # 使用绝对路径
        if save_dir is None:
            save_dir = self.visualization_dir
        else:
            save_dir = os.path.abspath(save_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        print("=" * 50)
        print("PhyMPGN训练结果综合分析报告")
        print("=" * 50)

        # 1. 绘制训练曲线
        print("\n1. 绘制训练曲线...")
        self.plot_training_curves(save_dir)

        # 2. 绘制学习率曲线
        print("\n2. 绘制学习率曲线...")
        self.plot_learning_rate(save_dir)

        # 3. 加载模型和数据
        print("\n3. 加载模型和测试数据...")
        try:
            model, test_loader, config = self.load_model_and_data()

            # 4. 速度场可视化
            print("\n4. 可视化预测速度场...")
            print(f"将生成 {num_samples} 个样本的速度场可视化")
            self.visualize_velocity_field(
                model, test_loader, num_samples=2, save_dir=save_dir)

            # 5. 误差分析
            print("\n5. 进行预测误差分析...")
            self.analyze_prediction_errors(model, test_loader, save_dir)

        except Exception as e:
            print(f"模型分析部分出错: {e}")
            print("跳过模型可视化部分")

        print("\n" + "=" * 50)
        print(f"分析完成！结果已保存到: {save_dir}")
        print("=" * 50)

        # 确保所有文件已保存完成
        import time
        time.sleep(1)  # 等待1秒确保文件系统写入完成

    def get_image_description(self, filename):
        """从文件名生成友好的描述"""
        try:
            # 解析文件名，例如 pred_t0_u_gt.png
            parts = filename.replace('.png', '').split('_')

            time_step = "未知时刻"
            component = "速度"
            data_type = "数据"

            for part in parts:
                if part.startswith('t') and part[1:].isdigit():
                    time_step = f"t={part[1:]}"
                elif part == 'u':
                    component = "u分量"
                elif part == 'v':
                    component = "v分量"
                elif part == 'gt':
                    data_type = "真实"
                elif part == 'prd':
                    data_type = "预测"

            if component == "速度":
                return f"{data_type}速度 ({time_step})"
            else:
                return f"{data_type}{component} ({time_step})"
        except:
            return filename


def main():
    # 创建可视化器（从cylinder_flow目录运行）
    visualizer = TrainingVisualizer(work_path='.')

    # 创建综合报告（使用绝对路径）
    visualizer.create_comprehensive_report()  # 使用默认绝对路径

    print(f"\n所有可视化结果已保存到: {visualizer.visualization_dir}")


if __name__ == '__main__':
    print(os.getcwd())
    main()
