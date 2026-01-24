import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# 屏蔽不重要的 Tkinter/Matplotlib 异常日志
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*main thread is not in main loop.*")

def parse_args():
    parser = argparse.ArgumentParser(description="""
    Plan v3: LoFTR Light-weight Fine-tuning (Freeze Backbone + Vessel Guidance)
    
    核心改动：
    - 冻结 Backbone，仅训练 Transformer
    - Strict Vessel Loss (背景权重 0.001)
    - 其他功能与 v2 完全对齐
    """)
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'], help='配准模式')
    parser.add_argument('--name', '-n', type=str, default='loftr_v3_freeze_backbone', help='本次训练的名称')
    parser.add_argument('--batch_size', type=int, default=4, help='每个 GPU 的 Batch Size')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=512, help='图像输入尺寸')
    parser.add_argument('--pretrained_ckpt', type=str, 
                        default='/data/student/Fengjunming/LoFTR/third_party/MINIMA/weights/minima_loftr.ckpt', 
                        help='MINIMA 预训练权重路径')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率 (仅针对 Transformer)')
    parser.add_argument('--main_cfg_path', type=str, default=None, help='主配置文件路径')
    
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=500, gpus='1')
    return parser.parse_args()

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
            # 训练集：Vessel Sigma 设为 0 (二值掩码)，因为我们使用 Strict Loss
            self.train_dataset = MultiModalDataset(
                DATA_ROOT, mode=self.args.mode, split='train', img_size=self.args.img_size, vessel_sigma=0.0)
            # 验证集：形变固定（Dataset内部通过idx固定种子实现）
            self.val_dataset = MultiModalDataset(
                DATA_ROOT, mode=self.args.mode, split='val', img_size=self.args.img_size, vessel_sigma=0.0)

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
                    plt.close(fig)
                elif 'm_bids' in batch and 'mkpts0_f' in batch and 'mkpts1_f' in batch:
                    # 如果当前 batch 没有生成图，但我们确实需要保存，则在此生成
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
                        plt.close(fig)
                    plt.close('all')
            except Exception as e:
                pass
                
        return mses

class PL_LoFTR_V3(PL_LoFTR):
    def __init__(self, config, pretrained_ckpt=None, *args, **kwargs):
        super().__init__(config, pretrained_ckpt, *args, **kwargs)
        
        # 【核心操作】冻结 Backbone 及相关层 (由于 loftr.py 已回归 Shared Backbone，直接冻结 backbone 即可)
        self.freeze_backbone()
        
        # 强制设置 Loss 参数
        self.background_weight_strict = 0.001 # 几乎不看背景
        # 修改 Coarse matching 的温度系数，使其更尖锐
        if hasattr(self.matcher.coarse_matching, 'temperature'):
             self.matcher.coarse_matching.temperature = 0.1

    def freeze_backbone(self):
        """冻结不需要更新的层"""
        logger.info("Plan v3: 正在冻结特征提取网络...")
        
        frozen_modules = [
            self.matcher.backbone, # 标准 Shared Backbone
            self.matcher.pos_encoding,
            self.matcher.fine_preprocess # 精细特征预处理
        ]

        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in self.matcher.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.matcher.parameters())
        logger.info(f"冻结完成。可训练参数: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M ({(trainable_params/total_params)*100:.1f}%)")

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    # 重写 compute_c_weight 实现 Strict Loss
    def _compute_c_weight_strict(self, data):
        """
        计算 Strict Loss 权重：只对血管区域给予高权重，背景区域给予极低权重
        注意：此时 data 中的 vessel_mask0/mask1 已经被 LoFTR.forward() 下采样到 coarse 分辨率
        """
        mask0 = data.get('vessel_mask0', data.get('mask0'))
        mask1 = data.get('vessel_mask1', data.get('mask1'))
        
        if mask0 is None or mask1 is None:
            return None

        # 兼容不同输入形状
        if mask0.dim() == 4: 
            mask0 = mask0.squeeze(1)
        if mask1.dim() == 4: 
            mask1 = mask1.squeeze(1)

        # 此时 mask0 和 mask1 已经是 coarse 分辨率 [N, H_c, W_c]，无需再次下采样
        # 直接展平
        m0_flat = mask0.flatten(1) # [N, L0] 其中 L0 = H_c * W_c
        m1_flat = mask1.flatten(1) # [N, L1]
        
        # 计算权重矩阵 [N, L0, L1]
        # w[i, j] = 1 if (i in vessel and j in vessel) else background_weight_strict
        c_weight = (m0_flat.unsqueeze(-1) * m1_flat.unsqueeze(1)).float()
        
        # 将 0 的区域设为 background_weight_strict
        c_weight = torch.where(c_weight > 0.5, 
                               torch.tensor(1.0, device=c_weight.device), 
                               torch.tensor(self.background_weight_strict, device=c_weight.device))
        
        return c_weight

    # 劫持 Loss 计算逻辑
    def _trainval_inference(self, batch):
        # 动态替换方法 (Monkey Patching instance method)
        # 为当前 batch 的 loss 实例注入 strict weight 逻辑
        if not hasattr(self.loss, '_is_patched_strict'):
            # 保存原始方法
            original_compute_c_weight = self.loss.compute_c_weight
            
            # 定义补丁方法，闭包绑定 self (PL_LoFTR_V3 instance)
            def patched_compute(data):
                # 优先使用 Strict Logic
                try:
                    w = self._compute_c_weight_strict(data)
                    if w is not None: 
                        return w
                except Exception as e:
                    logger.error(f"Strict weight computation failed: {e}, falling back to original method")
                    import traceback
                    traceback.print_exc()
                return original_compute_c_weight(data)
                
            # 替换
            self.loss.compute_c_weight = patched_compute
            self.loss._is_patched_strict = True
            
        super()._trainval_inference(batch)

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
    loguru_logger.add(log_file, enqueue=True, mode="a")  # 追加模式
    loguru_logger.info(f"日志将同时保存到: {log_file}")

    config.DATASET.MGDPT_IMG_RESIZE = args.img_size
    config.LOFTR.RESOLUTION = (8, 2)
    config.TRAINER.SEED = 66
    config.TRAINER.CANONICAL_LR = args.lr  # 使用用户指定的 LR
    config.TRAINER.WARMUP_STEP = 200  # 减少 Warmup，因为是微调
    pl.seed_everything(config.TRAINER.SEED)

    # 2. GPU 环境设置
    _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1) * getattr(args, 'num_nodes', 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    
    # 3. 初始化模型与数据
    loguru_logger.info(f"正在加载 MINIMA 预训练权重: {args.pretrained_ckpt}")
    model = PL_LoFTR_V3(config, pretrained_ckpt=args.pretrained_ckpt)
    loguru_logger.info("MINIMA 预训练权重加载完成，开始在此基础上训练 (Plan v3: Freeze Backbone)")
    data_module = MultimodalDataModule(args, config)

    # 4. 初始化训练器
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs_v3', name=args.name)
    
    # 启用自定义验证逻辑回调
    val_callback = MultimodalValidationCallback(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # 启用早停机制：如果连续 5 次验证 (即 25 epoch) loss 不下降则停止
    early_stop_callback = EarlyStopping(
        monitor='val_mse', 
        patience=5,
        verbose=True,
        mode='min',
        min_delta=0.0001
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        min_epochs=20,  # 确保至少训练 20 epoch
        check_val_every_n_epoch=5,  # 每 5 个 epoch 验证一次
        num_sanity_val_steps=3,  # 训练前跑 3 个 batch
        callbacks=[val_callback, lr_monitor, early_stop_callback],
        logger=tb_logger,
        plugins=DDPPlugin(find_unused_parameters=False) if _n_gpus > 1 else None,
        replace_sampler_ddp=True,
        weights_summary=None,
        progress_bar_refresh_rate=1
    )

    loguru_logger.info(f"开始训练: {args.name} (模式: {args.mode}, Plan v3)")
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
