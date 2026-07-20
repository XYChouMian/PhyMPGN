"""
边界层流训练脚本
使用 BLGraphDataset 和 BLModel
"""

import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from accelerate import Accelerator
from src.utils.kogger import Logger
import pprint

from src.common.config import Config
from src.utils.utils import AverageMeter, NodeType
from src.datasets.dataset import BLGraphDataset
from src.models.model import BLModel, MyLoss
from test import test


def train(model: BLModel, accelerator: Accelerator, tr_writer: SummaryWriter,
          val_writer: SummaryWriter, tr_loader, val_loader, epochs, loss_func,
          optimizer, scheduler, ckpt_path_val, ckpt_path_tr, logger, start_epoch):
    """训练循环"""
    min_val_loss = 1e6
    batch_time = AverageMeter()
    data_time = AverageMeter()

    for epoch in range(start_epoch, epochs + start_epoch):
        total_loss = 0
        end = time.time()
        for batch_idx, batch in enumerate(tr_loader):
            optimizer.zero_grad()

            # measure data loading time
            data_time.update(time.time() - end)
            model.train()

            # BatchData(y=[n, t, 2], pos=[n, 2], edge_index=[2, e], batch=[n])
            target = batch.y.transpose(0, 1)  # [t, n, 2]
            batch.y = target[0]  # [n, 2] 初始状态
            graph = batch
            t = target.shape[0]
            U_pred = model(graph, steps=t-1)  # [t, n', 2]
            target = torch.index_select(target, 1, batch.truth_index)  # [t, n', 2]

            # 损失掩码：只计算内部节点和出口节点（排除 wall 和 inlet）
            mask = torch.logical_or(graph.node_type == NodeType.NORMAL,
                                    graph.node_type == NodeType.OUTLET)
            tr_batch_loss = loss_func(U_pred, target, mask)

            # 检查 NaN
            assert not torch.any(torch.isnan(tr_batch_loss)), \
                f'[BL Train] epoch: {epoch}, batch_idx: {batch_idx}, tr_loss is nan'

            # Backward and optimize
            accelerator.backward(tr_batch_loss)
            # 梯度裁剪
            if accelerator.sync_gradients:
                max_norm = 0.15
                accelerator.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            with torch.no_grad():
                total_loss = total_loss + tr_batch_loss.item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        scheduler.step()

        mean_loss = total_loss / len(tr_loader)
        if accelerator.is_main_process:
            tr_writer.add_scalar('Loss', mean_loss, epoch)

        # 每 10 个 epoch 评估一次
        if epoch == start_epoch or epoch % 10 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # save state
                accelerator.save_state(output_dir=ckpt_path_tr)
            # evaluate
            val_mean_loss, min_val_loss, is_min = val_evaluate(
                accelerator=accelerator,
                model=model,
                val_loader=val_loader,
                loss_func=loss_func,
                min_val_loss=min_val_loss,
                ckpt_path_val=ckpt_path_val
            )
            if accelerator.is_main_process:
                val_writer.add_scalar('Loss', val_mean_loss, epoch)

            time_str = '[Epoch {:>4}/{}] Batch Time: {:.3f} ({:.3f}) \tData Time: {:.3f} ({:.3f})'\
                .format(epoch, start_epoch+epochs-1, batch_time.val, batch_time.avg, data_time.val,
                        data_time.avg)
            info_str = '[Epoch {:>4}/{}] tr_loss: {:.2e} \t\tval_loss: {:.2e} {}'\
                .format(epoch, start_epoch+epochs-1, mean_loss, val_mean_loss, '{}')
            if is_min:
                info_str = info_str.format('[MIN]')
            else:
                info_str = info_str.format('     ')

            if accelerator.is_main_process:
                logger.info(time_str)
                logger.info(info_str)
            accelerator.wait_for_everyone()


def val_evaluate(accelerator: Accelerator, model: BLModel, val_loader,
                 loss_func, min_val_loss, ckpt_path_val):
    """验证评估"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # batch: DataBatch(y=[n, t, 2], pos=[n, 2], edge_index=[2, e], batch=[n])
            target = batch.y.transpose(0, 1)  # [t, n, 2]
            batch.y = target[0]  # [n, 2]
            graph = batch
            t = target.shape[0]
            U_pred = model(graph, steps=t - 1)  # [t, n', 2]
            target = torch.index_select(target, 1, batch.truth_index)  # [t, n', 2]
            mask = torch.logical_or(graph.node_type == NodeType.NORMAL,
                                    graph.node_type == NodeType.OUTLET)
            cur_val_loss = loss_func(U_pred, target, mask)
            # gather loss from all devices
            gathered_cur_val_loss = accelerator.gather_for_metrics(cur_val_loss)
            val_loss += torch.mean(gathered_cur_val_loss).item()
        val_mean_loss = val_loss / len(val_loader)

    is_min = False
    if val_mean_loss < min_val_loss:
        min_val_loss = val_mean_loss
        is_min = True
        if accelerator.is_main_process:
            accelerator.save_state(output_dir=ckpt_path_val)

    return val_mean_loss, min_val_loss, is_min


def main():
    """主函数"""
    # load and set config
    args = Config.get_parser().parse_args()
    config = Config(yaml_filename=args.filename)

    accelerator = Accelerator()
    tr_writer = SummaryWriter(log_dir=config['log_dir'] + '_train')
    val_writer = SummaryWriter(log_dir=config['log_dir'] + '_val')

    logger = Logger('PID %d' % accelerator.process_index, file=config['log_file'])
    if accelerator.is_main_process:
        logger.info('Load config successfully!')
        logger.info(pprint.pformat(config.data))

    # 训练数据集
    tr_dataset = BLGraphDataset(
        root=config['data_root_dir'],
        raw_files=config['tr_raw_data'],
        processed_file=config['tr_processed_file'],
        dataset_start=config['dataset_start'],
        dataset_used=config['dataset_used'],
        time_start=config['time_start'],
        time_used=config['time_used'],
        window_size=config['window_size'],
        dtype=config['dtype'],
        training=True
    )

    tr_loader = DataLoader(
        dataset=tr_dataset,
        shuffle=config['window_shuffle'],
        batch_size=config['batch_size'],
        pin_memory=True
    )

    # 验证数据集
    val_dataset = BLGraphDataset(
        root=config['data_root_dir'],
        raw_files=config['val_raw_data'],
        processed_file=config['val_processed_file'],
        dataset_start=config['dataset_start'],
        dataset_used=config['dataset_used'],
        time_start=config['time_start'],
        time_used=config['time_used'],
        window_size=config['window_size'],
        dtype=config['dtype']
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=config['window_shuffle'],
        batch_size=config['batch_size'],
        # num_workers=config['num_workers'],
        pin_memory=True
    )

    if accelerator.is_main_process:
        logger.info('Num of datasets: {}'.format(tr_dataset.dataset_used))

    # 创建边界层模型
    model = BLModel(
        encoder_config=config['encoder_config'],
        mpnn_block_config=config['mpnn_block_config'],
        decoder_config=config['decoder_config'],
        laplace_block_config=config['laplace_block_config'],
        dtype=config['dtype'],
        device=accelerator.device,
        integral=config['integral']
    )

    if accelerator.is_main_process:
        total, mpnn, laplace = model.count_parameters()
        logger.info('Parameters: {} ({} + {})'.format(total, mpnn, laplace))

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    loss_func = MyLoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config['steplr_size'], gamma=config['steplr_gamma'])

    model, optimizer, tr_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, tr_loader, val_loader, scheduler
    )

    if config['continuous_train']:
        accelerator.load_state(input_dir=config['ckpt_path_tr'])

    if accelerator.is_main_process:
        logger.info('Train BL Model...')

    train(
        model=model,
        accelerator=accelerator,
        tr_writer=tr_writer,
        val_writer=val_writer,
        tr_loader=tr_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        ckpt_path_val=config['ckpt_path_val'],
        ckpt_path_tr=config['ckpt_path_tr'],
        logger=logger,
        start_epoch=config['start_epoch']
    )

    if accelerator.is_main_process:
        # 测试数据集
        te_dataset = BLGraphDataset(
            root=config['data_root_dir'],
            raw_files=config['te_raw_data'],
            processed_file=config['te_processed_file'],
            dataset_start=config['te_dataset_start'],
            dataset_used=config['te_dataset_used'],
            time_start=config['time_start'],
            time_used=config['time_used'],
            window_size=config['te_window_size'],
            dtype=config['dtype']
        )
        te_loader = DataLoader(
            dataset=te_dataset
        )

        # 在单 GPU 上测试
        accelerator.load_state(input_dir=config['ckpt_path_val'])
        te_writer = SummaryWriter(log_dir=config['log_dir'] + '_test')
        logger.info('Test BL Model...')

        test(
            model=model,
            te_writer=te_writer,
            te_loader=te_loader,
            device=accelerator.device,
            logger=logger
        )

        tr_writer.close()
        val_writer.close()
        te_writer.close()

        logger.info('BL Training Done!')


if __name__ == '__main__':
    main()