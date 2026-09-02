"""
边界层流 PhyMPGN 预测结果云图可视化
生成真实数据 vs 预测数据的速度场对比云图

用法:
    cd /home/wqx/projects/PhyMPGN/cylinder_flow
    python visualize_BL_velocity_fields.py --file configs/train_BL.yaml
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

from matplotlib.tri import Triangulation

from src.models.model import BLModel
from src.datasets.dataset import BLGraphDataset


# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei',
                                   'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'


class BLVelocityFieldVisualizer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()

        self.project_root = self.config.get('work_path', '.')
        self.ckpt_dir = os.path.join(self.project_root, 'ckpts')
        self.data_dir = self.config['data_root_dir']
        self.visualization_dir = os.path.join(
            self.project_root, 'visualization', 'bl_results')
        os.makedirs(self.visualization_dir, exist_ok=True)

        print(f"配置文件: {self.config_path}")
        print(f"项目根目录: {self.project_root}")
        print(f"可视化结果目录: {self.visualization_dir}")

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model_and_data(self):
        """加载训练好的 BLModel 和测试数据"""
        print("\n正在加载模型和数据...")

        # 测试数据集（与 train_BL.py 中的配置一致）
        te_dataset = BLGraphDataset(
            root=self.config['data_root_dir'],
            raw_files=self.config['te_raw_data'],
            processed_file=self.config['te_processed_file'],
            dataset_start=self.config['te_dataset_start'],
            dataset_used=self.config['te_dataset_used'],
            time_start=self.config['time_start'],
            time_used=self.config['te_window_size'],
            window_size=self.config['te_window_size'],
            dtype=torch.float32
        )

        # 构建 BLModel
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = BLModel(
            encoder_config=self.config['encoder_config'],
            mpnn_block_config=self.config['mpnn_block_config'],
            decoder_config=self.config['decoder_config'],
            laplace_block_config=self.config['laplace_block_config'],
            dtype=torch.float32,
            device=device,
            integral=self.config['integral']
        )

        # 加载验证集最优权重
        experiment_name = self.config.get('experiment_name', 'bl_expr_0')
        ckpt_path = self.config['ckpt_path_val'].format(experiment_name)
        if os.path.exists(os.path.join(ckpt_path, 'model.safetensors')):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(ckpt_path, 'model.safetensors'))
            model.load_state_dict(state_dict)
            print(f"成功加载模型权重: {ckpt_path}")
        elif os.path.exists(os.path.join(ckpt_path, 'pytorch_model.bin')):
            state_dict = torch.load(
                os.path.join(ckpt_path, 'pytorch_model.bin'),
                map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"成功加载模型权重: {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"模型文件不存在: {ckpt_path}\n"
                f"请先运行训练: python train_BL.py --file {self.config_path}")

        model.eval()
        # 将模型移动到指定设备
        model = model.to(device)

        return model, te_dataset, device

    def predict_velocity_field(self, model: BLModel, dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        预测速度场（逐步 unroll，与 train_BL.py 一致）

        Returns:
            pred_sequence: [t, n', 2] 预测序列（有量纲）
            truth_sequence: [t, n', 2] 真实序列（有量纲）
            pos: [n', 2] 节点位置（有量纲）
        """
        print("\n正在进行速度场预测...")
        model.eval()

        from torch_geometric.loader import DataLoader
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='预测'):
                batch = batch.to(device)

                # 与 train_BL.py 一致的数据准备
                target = batch.y.transpose(0, 1)  # [t, n, 2]
                batch.y = target[0]  # [n, 2] 初始时刻
                graph = batch
                steps = target.shape[0] - 1
                t_total = target.shape[0]

                print(f"  总时间步: {t_total}, 预测步数: {steps}")
                print(f"  节点数: {graph.y.shape[0]}")

                # 一次性 unroll 全部步数
                U_pred = model.forward(graph, steps=steps, progress_bar=True)  # [t, n', 2]

                # 提取真实值对应的节点
                target_selected = torch.index_select(
                    target, 1, batch.truth_index)  # [t, n', 2]

                # 有量纲化
                pos = torch.index_select(
                    graph.pos, 0, batch.truth_index)  # [n', 2]
                Uinf = graph.Uinf
                delta99 = graph.delta99
                dt = graph.dt  # 提取时间步长

                # 处理标量/向量形式的 Uinf, delta99, dt
                if Uinf.dim() == 0:
                    Uinf_val = Uinf.item()
                else:
                    Uinf_val = Uinf[0].item()

                if delta99.dim() == 0:
                    delta99_val = delta99.item()
                else:
                    delta99_val = delta99[0].item()

                if dt.dim() == 0:
                    dt_val = dt.item()
                else:
                    dt_val = dt[0].item()

                U_pred_dim = U_pred * Uinf_val
                target_dim = target_selected * Uinf_val
                pos_dim = pos * delta99_val

                pred_sequence = U_pred_dim.to(device='cpu').numpy()
                truth_sequence = target_dim.to(device='cpu').numpy()
                pos_np = pos_dim.to(device='cpu').numpy()

                # 获取三角网格
                face = batch.face.to(device='cpu').numpy()  # [3, num_faces]
                if face.ndim == 2:
                    face = face.T  # [num_faces, 3]

                print(f"  预测完成: shape={pred_sequence.shape}")
                print(f"  物理时间步长 dt={dt_val:.6f}s")
                break

        return pred_sequence, truth_sequence, pos_np, face, dt_val

    def plot_comprehensive_comparison(self, truth_u, pred_u, truth_v, pred_v,
                                      time_idx, pos, face, dt_physical,
                                      save_path=None, global_ranges=None):
        """
        绘制综合对比图 - 3行3列布局

        第1行: u真实, u预测, u误差
        第2行: v真实, v预测, v误差
        第3行: 速度大小真实, 速度大小预测, 总误差
        """
        # 计算速度大小和误差
        truth_mag = np.sqrt(truth_u**2 + truth_v**2)
        pred_mag = np.sqrt(pred_u**2 + pred_v**2)
        error_u = pred_u - truth_u
        error_v = pred_v - truth_v
        error_mag = np.sqrt(error_u**2 + error_v**2)

        # 创建三角网格
        tri = Triangulation(pos[:, 0], pos[:, 1], face)

        # 根据数据比例计算figsize
        x_range = pos[:, 0].max() - pos[:, 0].min()
        y_range = pos[:, 1].max() - pos[:, 1].min()
        aspect_ratio = x_range / y_range
        
        # 基础大小，根据比例调整
        base_size = 12
        if aspect_ratio > 1:
            figsize = (base_size * aspect_ratio, base_size)
        else:
            figsize = (base_size, base_size / aspect_ratio)
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)

        # 物理时间
        phys_time = time_idx * dt_physical
        fig.suptitle(
            f'边界层流 PhyMPGN 速度场预测综合对比\n'
            f'时间步 t={time_idx} (T={phys_time:.4f}s)',
            fontsize=16, fontweight='bold')

        # 颜色映射范围
        if global_ranges is not None:
            v_min_u, v_max_u = global_ranges['u']
            v_min_v, v_max_v = global_ranges['v']
            v_min_mag, v_max_mag = global_ranges['mag']
            max_error_u = global_ranges['error_u']
            max_error_v = global_ranges['error_v']
            max_error_mag = global_ranges['error_mag']
        else:
            v_min_u = min(truth_u.min(), pred_u.min())
            v_max_u = max(truth_u.max(), pred_u.max())
            v_min_v = min(truth_v.min(), pred_v.min())
            v_max_v = max(truth_v.max(), pred_v.max())
            v_min_mag = min(truth_mag.min(), pred_mag.min())
            v_max_mag = max(truth_mag.max(), pred_mag.max())
            max_error_u = max(abs(error_u.min()), abs(error_u.max()))
            max_error_v = max(abs(error_v.min()), abs(error_v.max()))
            max_error_mag = error_mag.max()

        # ===== 第1行: u 分量 =====
        p = axes[0, 0].tripcolor(tri, truth_u, shading='gouraud', cmap='jet',
                                 vmin=v_min_u, vmax=v_max_u)
        axes[0, 0].set_title('u分量 (真实)', fontsize=12, fontweight='bold')
        axes[0, 0].set_aspect('equal')
        axes[0, 0].axis('off')
        plt.colorbar(p, ax=axes[0, 0], shrink=0.6, label='u (m/s)')

        p = axes[0, 1].tripcolor(tri, pred_u, shading='gouraud', cmap='jet',
                                 vmin=v_min_u, vmax=v_max_u)
        axes[0, 1].set_title('u分量 (预测)', fontsize=12, fontweight='bold')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].axis('off')
        plt.colorbar(p, ax=axes[0, 1], shrink=0.6, label='u (m/s)')

        p = axes[0, 2].tripcolor(tri, error_u, shading='gouraud', cmap='seismic',
                                 vmin=-max_error_u, vmax=max_error_u)
        axes[0, 2].set_title('u分量误差 (预测-真实)', fontsize=12, fontweight='bold')
        axes[0, 2].set_aspect('equal')
        axes[0, 2].axis('off')
        plt.colorbar(p, ax=axes[0, 2], shrink=0.6, label='Δu (m/s)')

        # ===== 第2行: v 分量（有正有负，用发散型 colormap）=====
        # v 分量对称范围，使 0 对应白色
        v_abs_max = max(abs(v_min_v), abs(v_max_v))
        p = axes[1, 0].tripcolor(tri, truth_v, shading='gouraud', cmap='RdBu_r',
                                 vmin=-v_abs_max, vmax=v_abs_max)
        axes[1, 0].set_title('v分量 (真实)', fontsize=12, fontweight='bold')
        axes[1, 0].set_aspect('equal')
        axes[1, 0].axis('off')
        plt.colorbar(p, ax=axes[1, 0], shrink=0.6, label='v (m/s)')

        p = axes[1, 1].tripcolor(tri, pred_v, shading='gouraud', cmap='RdBu_r',
                                 vmin=-v_abs_max, vmax=v_abs_max)
        axes[1, 1].set_title('v分量 (预测)', fontsize=12, fontweight='bold')
        axes[1, 1].set_aspect('equal')
        axes[1, 1].axis('off')
        plt.colorbar(p, ax=axes[1, 1], shrink=0.6, label='v (m/s)')

        p = axes[1, 2].tripcolor(tri, error_v, shading='gouraud', cmap='seismic',
                                 vmin=-max_error_v, vmax=max_error_v)
        axes[1, 2].set_title('v分量误差 (预测-真实)', fontsize=12, fontweight='bold')
        axes[1, 2].set_aspect('equal')
        axes[1, 2].axis('off')
        plt.colorbar(p, ax=axes[1, 2], shrink=0.6, label='Δv (m/s)')

        # ===== 第3行: 速度大小 =====
        p = axes[2, 0].tripcolor(tri, truth_mag, shading='gouraud', cmap='jet',
                                 vmin=v_min_mag, vmax=v_max_mag)
        axes[2, 0].set_title('速度大小 |U| (真实)', fontsize=12, fontweight='bold')
        axes[2, 0].set_aspect('equal')
        axes[2, 0].axis('off')
        plt.colorbar(p, ax=axes[2, 0], shrink=0.6, label='|U| (m/s)')

        p = axes[2, 1].tripcolor(tri, pred_mag, shading='gouraud', cmap='jet',
                                 vmin=v_min_mag, vmax=v_max_mag)
        axes[2, 1].set_title('速度大小 |U| (预测)', fontsize=12, fontweight='bold')
        axes[2, 1].set_aspect('equal')
        axes[2, 1].axis('off')
        plt.colorbar(p, ax=axes[2, 1], shrink=0.6, label='|U| (m/s)')

        p = axes[2, 2].tripcolor(tri, error_mag, shading='gouraud', cmap='hot',
                                 vmin=0, vmax=max_error_mag)
        axes[2, 2].set_title('总误差大小 √(Δu²+Δv²)', fontsize=12, fontweight='bold')
        axes[2, 2].set_aspect('equal')
        axes[2, 2].axis('off')
        plt.colorbar(p, ax=axes[2, 2], shrink=0.6, label='误差 (m/s)')

        # 轴标签
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlabel('x (m)', fontsize=9)
                ax.set_ylabel('y (m)', fontsize=9)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.90,
                            bottom=0.05, wspace=0.15, hspace=0.15)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            # print(f"  已保存: {save_path}")

        plt.close()

    def compute_global_ranges(self, pred_sequence, truth_sequence):
        """计算全局 colorbar 范围"""
        all_truth_u = truth_sequence[:, :, 0]
        all_pred_u = pred_sequence[:, :, 0]
        all_truth_v = truth_sequence[:, :, 1]
        all_pred_v = pred_sequence[:, :, 1]
        all_error_u = all_pred_u - all_truth_u
        all_error_v = all_pred_v - all_truth_v
        all_error_mag = np.sqrt(all_error_u**2 + all_error_v**2)
        all_truth_mag = np.sqrt(all_truth_u**2 + all_truth_v**2)
        all_pred_mag = np.sqrt(all_pred_u**2 + all_pred_v**2)

        return {
            'u': (min(all_truth_u.min(), all_pred_u.min()),
                  max(all_truth_u.max(), all_pred_u.max())),
            'v': (min(all_truth_v.min(), all_pred_v.min()),
                  max(all_truth_v.max(), all_pred_v.max())),
            'mag': (0, max(all_truth_mag.max(), all_pred_mag.max())),
            'error_u': max(abs(all_error_u.min()), abs(all_error_u.max())),
            'error_v': max(abs(all_error_v.min()), abs(all_error_v.max())),
            'error_mag': all_error_mag.max(),
        }

    def create_time_series_visualization(
        self, pred_sequence, truth_sequence,
        pos, face,dt_physical,
        time_steps=None, save_dir=None,):
        """
        创建时序可视化

        Args:
            pred_sequence: [t, n', 2] 有量纲预测序列
            truth_sequence: [t, n', 2] 有量纲真实序列
            pos: [n', 2] 有量纲节点位置
            face: [num_faces, 3] 三角网格
            Uinf: 来流速度 (m/s)
            delta99: 边界层厚度 (m)
            time_steps: 要可视化的时间步列表
            save_dir: 保存目录
            dt_physical: 物理时间步长 (s)
        """
        print("\n正在生成速度场对比云图...")

        if save_dir is None:
            save_dir = self.visualization_dir
        else:
            save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"保存目录: {save_dir}")

        total_steps = pred_sequence.shape[0]
        if time_steps is None:
            num_frames = min(201, total_steps)
            time_steps = [int(t) for t in np.linspace(0, total_steps - 1,
                                          num_frames, dtype=int)]
        print(f"将可视化 {len(time_steps)} 个时间步")

        # 计算全局 colorbar 范围
        print("正在计算全局 colorbar 范围...")
        global_ranges = self.compute_global_ranges(
            pred_sequence, truth_sequence)
        print(f"  u: [{global_ranges['u'][0]:.4f}, {global_ranges['u'][1]:.4f}] m/s")
        print(f"  v: [{global_ranges['v'][0]:.4f}, {global_ranges['v'][1]:.4f}] m/s")
        print(f"  |U|: [{global_ranges['mag'][0]:.4f}, {global_ranges['mag'][1]:.4f}] m/s")

        # 导出时间步列表到JSON文件
        import json
        time_steps_json_path = os.path.join(save_dir, 'time_steps.json')
        completed_time_steps = []

        # 逐时间步生成对比图
        for t_idx in tqdm(time_steps, desc='生成对比图'):
            truth_u = truth_sequence[t_idx, :, 0]
            pred_u = pred_sequence[t_idx, :, 0]
            truth_v = truth_sequence[t_idx, :, 1]
            pred_v = pred_sequence[t_idx, :, 1]

            save_path = os.path.join(
                save_dir, f'bl_velocity_t{t_idx:04d}.png')
            self.plot_comprehensive_comparison(
                truth_u, pred_u, truth_v, pred_v,
                t_idx, pos, face, dt_physical,
                save_path=save_path, global_ranges=global_ranges)

            # 每生成一张图片就更新JSON文件
            completed_time_steps.append(int(t_idx))
            with open(time_steps_json_path, 'w') as f:
                json.dump(completed_time_steps, f, indent=2)

        print(f"\n完成！共生成 {len(time_steps)} 张对比图")
        print(f"已导出时间步列表到: {time_steps_json_path}")

    def create_error_evolution_plot(self, pred_sequence, truth_sequence,
                                    dt_physical=None, save_dir=None):
        """
        绘制误差随时间演化曲线

        Args:
            pred_sequence: [t, n', 2]
            truth_sequence: [t, n', 2]
            dt_physical: 物理时间步长 (s)
        """
        print("\n正在生成误差演化曲线...")

        if save_dir is None:
            save_dir = self.visualization_dir
        os.makedirs(save_dir, exist_ok=True)

        total_steps = pred_sequence.shape[0]
        # 逐时间步计算 MSE
        mse_per_step = np.mean(
            (pred_sequence - truth_sequence) ** 2,
            axis=(1, 2))  # [t,]
        rmse_per_step = np.sqrt(mse_per_step)

        # 相对误差
        truth_mag = np.sqrt(
            truth_sequence[:, :, 0] ** 2 + truth_sequence[:, :, 1] ** 2)
        mean_truth_mag = np.mean(truth_mag, axis=1)  # [t,]
        relative_error = rmse_per_step / (mean_truth_mag + 1e-8)

        if dt_physical is not None:
            time_axis = np.arange(total_steps) * dt_physical
            xlabel = '物理时间 (s)'
        else:
            time_axis = np.arange(total_steps)
            xlabel = '时间步'

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # RMSE
        axes[0].plot(time_axis, rmse_per_step, 'b-', linewidth=1.5)
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel('RMSE (m/s)')
        axes[0].set_title('速度场 RMSE 随时间演化')
        axes[0].grid(True, alpha=0.3)

        # 相对误差
        axes[1].plot(time_axis, relative_error * 100, 'r-', linewidth=1.5)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel('相对误差 (%)')
        axes[1].set_title('速度场相对误差随时间演化')
        axes[1].grid(True, alpha=0.3)

        # u/v 分量分别 RMSE
        rmse_u = np.sqrt(np.mean(
            (pred_sequence[:, :, 0] - truth_sequence[:, :, 0]) ** 2, axis=1))
        rmse_v = np.sqrt(np.mean(
            (pred_sequence[:, :, 1] - truth_sequence[:, :, 1]) ** 2, axis=1))
        axes[2].plot(time_axis, rmse_u, 'b-', linewidth=1.5, label='u RMSE')
        axes[2].plot(time_axis, rmse_v, 'r-', linewidth=1.5, label='v RMSE')
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel('RMSE (m/s)')
        axes[2].set_title('u/v 分量 RMSE 随时间演化')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'bl_error_evolution.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")


def main():
    print("=" * 60)
    print("边界层流 PhyMPGN 速度场预测云图可视化")
    print("=" * 60)

    # 创建可视化器
    visualizer = BLVelocityFieldVisualizer(config_path='configs/train_BL_sparse_7070.yaml')

    # 加载模型和数据
    model, dataset, device = visualizer.load_model_and_data()

    # 预测
    pred_sequence, truth_sequence, pos, face, dt_val = \
        visualizer.predict_velocity_field(model, dataset, device)

    # 确定可视化时间步（均匀分布 200 帧）
    total_steps = pred_sequence.shape[0]
    num_frames = min(201, total_steps)
    time_steps = list(np.linspace(0, total_steps - 1, num_frames, dtype=int))
    print(f"\n将可视化 {len(time_steps)} 个时间步")

    # 生成速度场对比云图
    visualizer.create_time_series_visualization(
        pred_sequence=pred_sequence,
        truth_sequence=truth_sequence,
        pos=pos,
        face=face,
        dt_physical=dt_val,
        time_steps=time_steps,
        save_dir=None,
    )

    # 生成误差演化曲线
    visualizer.create_error_evolution_plot(
        pred_sequence=pred_sequence,
        truth_sequence=truth_sequence,
        dt_physical=dt_val,
        save_dir=None
    )

    print("\n" + "=" * 60)
    print("可视化完成！")
    print(f"结果保存到: {visualizer.visualization_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
