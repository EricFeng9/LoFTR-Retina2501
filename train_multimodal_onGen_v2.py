import sys
import os
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，防止多线程 Tkinter 报错
import matplotlib.pyplot as plt
import math
import argparse
import pprint
from pathlib import Path
from loguru import logger
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
import logging

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.lightning.lightning_loftr import PL_LoFTR
from data.FIVES_extract.FIVES_extract import MultiModalDataset
from src.utils.plotting import make_matching_figures

# 数据集根目录硬编码
DATA_ROOT = "/data/student/Fengjunming/LoFTR/data/FIVES_extract_v2"

# 配置日志格式
loguru_logger = get_rank_zero_only_logger(logger)
loguru_logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
loguru_logger.add(sys.stderr, format=log_format, level="INFO")

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("fsspec").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="""
    LoFTR 多模态眼底图像配准训练脚本 (V2 - 生成数据集 + 端到端学习策略)
    
    依据 plan.md 实现：
    - 完整图像输入（含背景），不做掩码过滤
    - 血管掩码仅用于损失权重和注意力偏置（不拼接到 Backbone）
    - 支持大角度旋转（±90°）和翻转（10%概率）
    - 端到端学习：模型自动学习血管区域的重要性
    """)
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'], help='配准模式')
    parser.add_argument('--name', '-n', type=str, default='loftr_multimodal_fives', help='本次训练的名称')
    parser.add_argument('--batch_size', type=int, default=4, help='每个 GPU 的 Batch Size')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=512, help='图像输入尺寸')
    parser.add_argument('--vessel_sigma', type=float, default=6.0, help='血管高斯软掩码的 σ（像素单位），用于损失加权')
    parser.add_argument('--main_cfg_path', type=str, default=None, help='主配置文件路径')
    
    # 自动添加 Lightning Trainer 参数 (如 --gpus, --max_epochs, --accelerator 等)
    parser = pl.Trainer.add_argparse_args(parser)
    
    # 设置 Lightning 参数的默认值
    parser.set_defaults(max_epochs=500, gpus='1')
    
    return parser.parse_args()

# 屏蔽不重要的 Tkinter/Matplotlib 异常日志
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*main thread is not in main loop.*")

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # 训练集：每次形变随机
            self.train_dataset = MultiModalDataset(
                DATA_ROOT, mode=self.args.mode, split='train', img_size=self.args.img_size, vessel_sigma=self.args.vessel_sigma)
            # 验证集：形变固定（Dataset内部通过idx固定种子实现）
            self.val_dataset = MultiModalDataset(
                DATA_ROOT, mode=self.args.mode, split='val', img_size=self.args.img_size, vessel_sigma=self.args.vessel_sigma)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, **self.loader_params)

def filter_valid_area(img1, img2):
    """筛选有效区域：只保留两张图片都不为纯黑像素的部分，并裁剪使有效区域填满画布"""
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
    
    if len(img1.shape) == 3:
        mask1 = np.any(img1 > 10, axis=2)
    else:
        mask1 = img1 > 0
    
    if len(img2.shape) == 3:
        mask2 = np.any(img2 > 10, axis=2)
    else:
        mask2 = img2 > 0
    
    valid_mask = mask1 & mask2
    
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return img1, img2
    
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    filtered_img1 = img1[row_min:row_max+1, col_min:col_max+1].copy()
    filtered_img2 = img2[row_min:row_max+1, col_min:col_max+1].copy()
    
    valid_mask_cropped = valid_mask[row_min:row_max+1, col_min:col_max+1]
    
    if len(filtered_img1.shape) == 3:
        filtered_img1[~valid_mask_cropped] = 0
    else:
        filtered_img1[~valid_mask_cropped] = 0
    
    if len(filtered_img2.shape) == 3:
        filtered_img2[~valid_mask_cropped] = 0
    else:
        filtered_img2[~valid_mask_cropped] = 0
    
    return filtered_img1, filtered_img2

def create_chessboard(img1, img2, grid_size=4):
    """
    创建棋盘图，将两张图交替拼接成4x4的棋盘
    Args:
        img1: numpy array [H, W]，第一张图
        img2: numpy array [H, W]，第二张图
        grid_size: 棋盘格子数量，默认4x4
    Returns:
        chessboard: numpy array [H, W]，棋盘图
    """
    H, W = img1.shape
    assert img2.shape == (H, W), "Two images must have the same size"
    
    # 每个格子的大小
    cell_h = H // grid_size
    cell_w = W // grid_size
    
    # 创建棋盘图
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算当前格子的位置
            y_start = i * cell_h
            y_end = (i + 1) * cell_h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w
            
            # 交替选择图像：如果 (i+j) 是偶数，用img1，否则用img2
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    
    return chessboard

class MultimodalValidationCallback(Callback):
    """自定义验证回调，负责保存图像、计算MSE Loss、手动管理最优/最新模型"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val_mse = float('inf')
        self.result_dir = Path(f"results/{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.sanity_checking:
            sub_dir = "step0"
        else:
            sub_dir = f"epoch{trainer.current_epoch + 1}"
            
        epoch_dir = self.result_dir / sub_dir
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # 批量处理并保存当前 batch 的所有样本
        # 直接使用 validation_step 计算好的结果，避免重复推理
        batch_mses = self._save_batch_results(trainer, pl_module, batch, outputs, epoch_dir)
        self.epoch_mses.extend(batch_mses)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        
        # 简化训练指标显示
        display_metrics = {}
        for k in ['loss', 'train/loss_c', 'train/loss_f']:
            # 优先检查 epoch 平均值
            epoch_key = f"{k}_epoch"
            if epoch_key in metrics:
                name = k.replace('train/', '')
                display_metrics[name] = metrics[epoch_key].item()
            elif k in metrics:
                name = k.replace('train/', '')
                display_metrics[name] = metrics[k].item()
        
        if display_metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
            loguru_logger.info(f"Epoch {epoch} 训练总结 >> {metric_str}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.epoch_mses:
            return
            
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        
        # 计算整个验证集的平均 MSE
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        
        # 简化验证指标显示
        display_metrics = {'mse_viz': avg_mse}
        
        # 提取 AUC 指标
        for k in ['auc@5', 'auc@10', 'auc@20']:
            if k in metrics:
                display_metrics[k] = metrics[k].item()
        
        # 提取验证 Loss
        for k in ['val/avg_loss_c', 'val/avg_loss_f', 'val/avg_loss']:
            if k in metrics:
                name = k.replace('val/avg_', '')
                display_metrics[name] = metrics[k].item()
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        loguru_logger.info(f"Epoch {epoch} 验证总结 >> {metric_str}")
        
        # 记录验证 MSE 供 Trainer 和 EarlyStopping 监控
        pl_module.log("val_mse", avg_mse, on_epoch=True, prog_bar=False, logger=True)
        
        # 更新最新权重
        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")
        with open(latest_path / "log.txt", "w") as f:
            f.write(f"Epoch: {epoch}\nLatest MSE: {avg_mse:.6f}")
            
        # 更新最优权重
        if avg_mse < self.best_val_mse:
            self.best_val_mse = avg_mse
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\nBest MSE: {avg_mse:.6f}")
            loguru_logger.info(f"发现新的最优模型! Epoch {epoch}, MSE: {avg_mse:.6f}")

    def _save_batch_results(self, trainer, pl_module, batch, outputs, epoch_dir):
        # 此时 batch 已经在 validation_step 中被处理过，包含了 H_est 和匹配点信息
        batch_size = batch['image0'].shape[0]
        mses = []
        
        # 获取估计的单应矩阵 H_est (由 metrics.py 中的 compute_homography_errors 提供)
        # 注意：H_est 是在 validation_step 的 _compute_metrics 中生成的，存储在 outputs 中
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        
        # 【调试】打印 H_est 信息，检查是否正确获取
        if 'H_est' not in outputs:
            loguru_logger.warning(f"⚠️ H_est 未在 outputs 中找到，使用单位矩阵（这会导致 moving_result 与 moving 完全一致）")
        else:
            num_identity = sum([np.allclose(H, np.eye(3)) for H in H_ests])
            if num_identity > 0:
                loguru_logger.warning(f"⚠️ Batch 中有 {num_identity}/{batch_size} 个样本的 H_est 是单位矩阵")
        
        pair_names0 = batch['pair_names'][0]
        pair_names1 = batch['pair_names'][1]

        for i in range(batch_size):
            # 为每个样本创建独立文件夹
            sample_name = f"{Path(pair_names0[i]).stem}_vs_{Path(pair_names1[i]).stem}"
            save_path = epoch_dir / sample_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1_origin = (batch['image1_origin'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            # 保存血管掩码（如果存在）
            if 'mask0' in batch:
                m0 = batch['mask0'][i].cpu().numpy()
                m1 = batch['mask1'][i].cpu().numpy()
                # 兼容 squeeze 后的形状 (H, W) 或 (1, H, W)
                if m0.ndim == 3: m0 = m0[0]
                if m1.ndim == 3: m1 = m1[0]
                cv2.imwrite(str(save_path / "mask0_vessel.png"), (m0 * 255).astype(np.uint8))
                cv2.imwrite(str(save_path / "mask1_vessel_warped.png"), (m1 * 255).astype(np.uint8))
            
            H_est = H_ests[i]
            
            # 【调试】检查当前样本的匹配点数量和 H_est 状态
            is_identity = np.allclose(H_est, np.eye(3))
            
            if 'm_bids' in batch:
                mask_i = batch['m_bids'] == i
                num_matches = mask_i.sum().item()
                if is_identity:
                    loguru_logger.warning(f"样本 {sample_name}: H_est 是单位矩阵 (匹配点数: {num_matches})")
            elif is_identity:
                loguru_logger.warning(f"样本 {sample_name}: H_est 是单位矩阵")
            
            # 计算 moving_result: 将 img1 使用 H_est 的逆变换回 img0 空间
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except Exception as e:
                loguru_logger.error(f"样本 {sample_name}: H_est 求逆失败: {e}")
                img1_result = np.zeros_like(img0)
                
            # 计算有效区域 MSE
            try:
                res_f, orig_f = filter_valid_area(img1_result, img1_origin)
                mask = (res_f > 0)
                if np.any(mask):
                    mse = np.mean(((res_f[mask] / 255.0) - (orig_f[mask] / 255.0)) ** 2)
                else:
                    mse = np.mean(((img1_result / 255.0) - (img1_origin / 255.0)) ** 2)
            except:
                mse = np.mean(((img1_result / 255.0) - (img1_origin / 255.0)) ** 2)
            
            mses.append(mse)
            
            # 保存各模态图像
            cv2.imwrite(str(save_path / "fix.png"), img0)
            cv2.imwrite(str(save_path / "moving.png"), img1)
            cv2.imwrite(str(save_path / "moving_origin.png"), img1_origin)
            cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
            
            # 保存棋盘图
            try:
                chessboard = create_chessboard(img1_result, img1_origin)
                cv2.imwrite(str(save_path / "chessboard.png"), chessboard)
            except:
                pass
            
            # 复用 Matches 可视化图 (针对单个样本生成)
            try:
                # 优先从 outputs 中获取已经生成的图
                mode = pl_module.config.TRAINER.PLOT_MODE
                if 'figures' in outputs and mode in outputs['figures'] and len(outputs['figures'][mode]) > i:
                    fig = outputs['figures'][mode][i]
                    fig.savefig(str(save_path / "matches.png"), bbox_inches='tight')
                    plt.close(fig) # 显式关闭，释放内存
                elif 'm_bids' in batch and 'mkpts0_f' in batch and 'mkpts1_f' in batch:
                    # 如果当前 batch 没有生成图，但我们确实需要保存，则在此生成（这部分保持兼容性）
                    mini_batch = {
                        'image0': batch['image0'][i:i+1],
                        'image1': batch['image1'][i:i+1],
                        'mkpts0_f': batch['mkpts0_f'][batch['m_bids'] == i],
                        'mkpts1_f': batch['mkpts1_f'][batch['m_bids'] == i],
                        'm_bids': torch.zeros_like(batch['m_bids'][batch['m_bids'] == i]),
                        'dataset_name': batch['dataset_name'],
                        'epi_errs': batch['epi_errs'][batch['m_bids'] == i]
                    }
                    if 'scale0' in batch: mini_batch['scale0'] = batch['scale0'][i:i+1]
                    if 'scale1' in batch: mini_batch['scale1'] = batch['scale1'][i:i+1]
                    
                    figures = make_matching_figures(mini_batch, pl_module.config, mode='evaluation')
                    if 'evaluation' in figures:
                        fig = figures['evaluation'][0]
                        fig.savefig(str(save_path / "matches.png"), bbox_inches='tight')
                        plt.close(fig) # 显式关闭
                    plt.close('all') # 双重保险
            except Exception as e:
                pass
                
        return mses

class DelayedEarlyStopping(EarlyStopping):
    """自定义早停回调，在指定 epoch 之后才开始计数"""
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        
    def on_validation_end(self, trainer, pl_module):
        # 只有在达到 start_epoch 后才开始早停检查
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

def main():
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # 1. 初始化配置与日志路径
    config = get_cfg_defaults()
    if args.main_cfg_path:
        config.merge_from_file(args.main_cfg_path)
    
    # 设置日志文件
    result_dir = Path(f"results/{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "log.txt"
    loguru_logger.add(log_file, enqueue=True, mode="a") # 追加模式
    loguru_logger.info(f"日志将同时保存到: {log_file}")

    config.DATASET.MGDPT_IMG_RESIZE = args.img_size
    config.LOFTR.RESOLUTION = (8, 2)
    config.TRAINER.SEED = 66
    pl.seed_everything(config.TRAINER.SEED)

    # 2. GPU 环境设置
    # setup_gpus 会返回 GPU 的数量 (int)
    _n_gpus = setup_gpus(args.gpus)
    # 确保 WORLD_SIZE 至少为 1，用于学习率缩放
    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1) * getattr(args, 'num_nodes', 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    
    # 3. 初始化模型与数据
    model = PL_LoFTR(config)
    data_module = MultimodalDataModule(args, config)

    # 4. 初始化训练器
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.name)
    
    # 启用自定义验证逻辑回调
    val_callback = MultimodalValidationCallback(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # 启用早停机制：epoch 50 后，如果连续 5 次验证 (即 25 epoch) loss 不下降则停止
    # 注意：check_val_every_n_epoch=5，所以 patience=5 对应 5 次验证检查
    # 使用自定义 EarlyStopping，在 epoch 50 之前不计数
    early_stop_callback = DelayedEarlyStopping(
        start_epoch=50,
        monitor='val_mse', 
        patience=5,
        verbose=True,
        mode='min',
        min_delta=0.0001
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        min_epochs=20, # 确保至少训练 20 epoch
        check_val_every_n_epoch=5, # 每 5 个 epoch 验证一次
        num_sanity_val_steps=3, # 训练前跑 3 个 batch (12 样本)
        callbacks=[val_callback, lr_monitor, early_stop_callback],
        logger=tb_logger,
        plugins=DDPPlugin(find_unused_parameters=False) if _n_gpus > 1 else None,
        replace_sampler_ddp=True,
        weights_summary=None,
        progress_bar_refresh_rate=1
    )

    loguru_logger.info(f"开始训练: {args.name} (模式: {args.mode})")
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
