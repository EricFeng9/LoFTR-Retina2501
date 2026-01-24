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

# 导入真实数据集（用于验证）
from data.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# 数据集包装器：将真实数据集的元组格式转换为 LoFTR 需要的字典格式
class RealDatasetWrapper(torch.utils.data.Dataset):
    """包装真实数据集，使其输出格式与 MultiModalDataset 一致"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 真实数据集返回: (fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1)
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        
        # 注意：真实数据集的 moving_original_tensor 和 moving_gt_tensor 在 [-1, 1] 范围
        # 需要转换回 [0, 1] 范围以匹配生成数据集的格式
        moving_original_tensor = (moving_original_tensor + 1) / 2  # [-1, 1] -> [0, 1]
        moving_gt_tensor = (moving_gt_tensor + 1) / 2  # [-1, 1] -> [0, 1]
        # fix_tensor 已经在 [0, 1] 范围，不需要转换
        
        # 转换为 LoFTR 需要的字典格式
        # fix_tensor 和 moving_gt_tensor 是 [C, H, W]，需要转换为 [1, H, W] (灰度)
        if fix_tensor.shape[0] == 3:
            fix_gray = 0.299 * fix_tensor[0] + 0.587 * fix_tensor[1] + 0.114 * fix_tensor[2]
            fix_gray = fix_gray.unsqueeze(0)
        else:
            fix_gray = fix_tensor
            
        if moving_gt_tensor.shape[0] == 3:
            moving_gray = 0.299 * moving_gt_tensor[0] + 0.587 * moving_gt_tensor[1] + 0.114 * moving_gt_tensor[2]
            moving_gray = moving_gray.unsqueeze(0)
        else:
            moving_gray = moving_gt_tensor
            
        if moving_original_tensor.shape[0] == 3:
            moving_orig_gray = 0.299 * moving_original_tensor[0] + 0.587 * moving_original_tensor[1] + 0.114 * moving_original_tensor[2]
            moving_orig_gray = moving_orig_gray.unsqueeze(0)
        else:
            moving_orig_gray = moving_original_tensor
        
        # 从完整路径中提取文件名
        import os
        fix_name = os.path.basename(fix_path)
        moving_name = os.path.basename(moving_path)
        
        # 注意：真实数据集返回的 T_0to1 是从 moving 到 fix 的变换 (fix = H * moving)
        # 而 LoFTR 评估逻辑期望 T_0to1 是从 image0 (fix) 到 image1 (moving) 的变换
        # 因此我们需要取逆矩阵
        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except:
            T_fix_to_moving = T_0to1 # 退化情况
            
        return {
            'image0': fix_gray,  # [1, H, W], [0, 1] 范围 - 固定图 (Target)
            'image1': moving_orig_gray,  # [1, H, W], [0, 1] 范围 - 原始待配准图 (Input)
            'image1_gt': moving_gray,  # [1, H, W], [0, 1] 范围 - GT配准后的待配准图 (Reference for MSE)
            'T_0to1': T_fix_to_moving,  # [3, 3] 从 image0 (fix) 到 image1 (moving) 的 GT 变换
            'pair_names': (fix_name, moving_name),  # 元组 (str, str)
            'dataset_name': 'MultiModal'  # 统一数据集名称
        }

# 导入域随机化增强模块
from gen_data_enhance import apply_domain_randomization, save_batch_visualization

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
    LoFTR 多模态眼底图像配准训练脚本 (V2.1 - 生成数据集 + 完全自主学习)
    
    依据 plan_v2_1.md 实现：
    - 完整图像输入（含背景），不做掩码过滤
    - **完全弃用血管掩码**：vessel_soft_lambda = 0, loss_weight = 1.0
    - 模型必须完全自主学习跨模态特征
    - 支持加载 MINIMA 预训练权重作为初始化
    - 支持大角度旋转（±90°）和翻转（10%概率）
    """)
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'], help='配准模式')
    parser.add_argument('--name', '-n', type=str, default='loftr_multimodal_fives', help='本次训练的名称')
    parser.add_argument('--batch_size', type=int, default=4, help='每个 GPU 的 Batch Size')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=512, help='图像输入尺寸')
    parser.add_argument('--vessel_sigma', type=float, default=6.0, help='血管高斯软掩码的 σ（像素单位），用于损失加权')
    parser.add_argument('--pretrained_ckpt', type=str, default='weights/outdoor_ds.ckpt', help='预训练权重路径（默认使用 LoFTR outdoor_ds 权重）')
    parser.add_argument('--use_domain_randomization', action='store_true', default=True, help='是否启用域随机化增强')
    parser.add_argument('--val_on_real', action='store_true', default=True, help='是否在真实数据集上验证')
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
            # 训练集：每次形变随机（生成数据）
            self.train_dataset = MultiModalDataset(
                DATA_ROOT, mode=self.args.mode, split='train', img_size=self.args.img_size, vessel_sigma=self.args.vessel_sigma)
            
            # 验证集：根据参数选择
            if self.args.val_on_real:
                # 使用真实数据集验证（与 train_multimodal_v3_3.py 一致）
                if self.args.mode == 'cfocta':
                    base_dataset = CFOCTADataset(root_dir='data/CF_OCTA_v2_repaired', split='val', mode='cf2octa')
                elif self.args.mode == 'cffa':
                    base_dataset = CFFADataset(root_dir='data/operation_pre_filtered_cffa', split='val', mode='fa2cf')
                elif self.args.mode == 'cfoct':
                    base_dataset = CFOCTDataset(root_dir='data/operation_pre_filtered_cfoct', split='val', mode='cf2oct')
                elif self.args.mode == 'octfa':
                    base_dataset = OCTFADataset(root_dir='data/operation_pre_filtered_octfa', split='val', mode='fa2oct')
                
                # 包装真实数据集，使其输出格式与 MultiModalDataset 一致
                self.val_dataset_real = RealDatasetWrapper(base_dataset)
                
                # 同时保留生成数据验证集用于可视化对比
                self.val_dataset_gen = MultiModalDataset(
                    DATA_ROOT, mode=self.args.mode, split='val', img_size=self.args.img_size, vessel_sigma=self.args.vessel_sigma)
            else:
                # 仅使用生成数据验证
                self.val_dataset = MultiModalDataset(
                    DATA_ROOT, mode=self.args.mode, split='val', img_size=self.args.img_size, vessel_sigma=self.args.vessel_sigma)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        if self.args.val_on_real:
            # 返回双验证数据集：[生成数据, 真实数据]
            # 注意：生成数据验证仅用于可视化，真实数据用于计算指标
            return [
                torch.utils.data.DataLoader(self.val_dataset_gen, shuffle=False, **self.loader_params),
                torch.utils.data.DataLoader(self.val_dataset_real, shuffle=False, **self.loader_params)
            ]
        else:
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

class PL_LoFTR_WithDomainRand(PL_LoFTR):
    """PL_LoFTR 的包装类，在训练时应用域随机化增强"""
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, use_domain_rand=True, result_dir=None):
        super().__init__(config, pretrained_ckpt, profiler, dump_dir)
        self.use_domain_rand = use_domain_rand
        self.saved_batches = 0  # 用于跟踪已保存的batch数量
        self.result_dir = result_dir  # 保存结果目录
        
    def training_step(self, batch, batch_idx):
        """重写 training_step，在模型输入前应用域随机化（仅训练时）"""
        if self.use_domain_rand and self.training:  # 添加 self.training 检查
            # 保存原始图像（用于前2个batch的可视化，包括sanity check）
            # 只在前2个batch保存可视化
            if self.saved_batches < 2:
                img0_orig = batch['image0'].clone()
                img1_orig = batch['image1'].clone()
            
            # 对 image0 和 image1 分别应用独立的域随机化增强
            # 关键：两边使用完全独立的随机参数，打破它们之间的纹理相关性
            batch['image0'] = apply_domain_randomization(batch['image0'])
            batch['image1'] = apply_domain_randomization(batch['image1'])
            
            # 保存前2个batch的可视化（检查域随机化效果）
            if self.saved_batches < 2:
                if self.result_dir is None:
                    result_dir = Path(f"results/{self.config.DATASET.TRAINVAL_DATA_SOURCE}/{self.config.TRAINER.EXP_NAME}")
                else:
                    result_dir = Path(self.result_dir)
                result_dir.mkdir(parents=True, exist_ok=True)
                
                vessel_mask = batch.get('mask0', None)
                
                # 根据当前状态决定epoch标记
                if self.trainer.sanity_checking:
                    epoch_label = 0  # sanity check阶段标记为epoch 0
                else:
                    epoch_label = self.current_epoch + 1
                
                save_batch_visualization(
                    img0_orig, img1_orig, 
                    batch['image0'], batch['image1'],
                    str(result_dir), 
                    epoch=epoch_label, 
                    step=self.saved_batches + 1, 
                    batch_size=batch['image0'].shape[0],
                    vessel_mask=vessel_mask
                )
                self.saved_batches += 1
                loguru_logger.info(f"已保存第 {self.saved_batches}/2 个batch的域随机化可视化（训练模式，Epoch {epoch_label}, Batch {batch_idx}）")
        
        # 强制移除所有血管掩码相关键，确保模型不接收任何掩码信息 (V2.1 No Mask Strategy)
        # 使用 pop 而不是设为 None，以避免 LoFTR.py 中 "if 'mask0' in data" 的逻辑报错
        keys_to_remove = ['mask0', 'mask1', 'vessel_mask0', 'vessel_mask1', 'vessel_weight0', 'vessel_weight1']
        for k in keys_to_remove:
            if k in batch:
                batch.pop(k)
        
        # 调用原始的 training_step
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """重写 validation_step 以支持多数据集验证（接受 dataloader_idx 参数）
        注意：验证时不应用域随机化，直接使用原始图像"""
        # 验证时同样移除掩码，确保计算的 Loss 是无加权的
        keys_to_remove = ['mask0', 'mask1', 'vessel_mask0', 'vessel_mask1', 'vessel_weight0', 'vessel_weight1']
        for k in keys_to_remove:
            if k in batch:
                batch.pop(k)

        # 验证时不应用域随机化，直接调用基类方法
        return super().validation_step(batch, batch_idx)

class MultimodalValidationCallback(Callback):
    """
    自定义验证回调，负责保存图像、计算MSE Loss、手动管理最优/最新模型
    支持双验证数据集：
    - 验证集0（生成数据）：仅可视化，不计算指标
    - 验证集1（真实数据）：可视化 + 计算指标 + 更新模型
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val_mse = float('inf')
        self.result_dir = Path(f"results/{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """处理验证批次结束事件，支持多数据集验证"""
        if trainer.sanity_checking:
            sub_dir = "step0"
        else:
            sub_dir = f"epoch{trainer.current_epoch + 1}"
        
        # 根据不同的验证数据集选择不同的保存目录
        if self.args.val_on_real:
            if dataloader_idx == 0:
                # 生成数据验证集：仅可视化
                epoch_dir = self.result_dir / f"{sub_dir}_generated"
            else:
                # 真实数据验证集：可视化 + 计算指标
                epoch_dir = self.result_dir / f"{sub_dir}_real"
        else:
            epoch_dir = self.result_dir / sub_dir
            
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # 批量处理并保存当前 batch 的所有样本
        # 直接使用 validation_step 计算好的结果，避免重复推理
        batch_mses = self._save_batch_results(trainer, pl_module, batch, outputs, epoch_dir)
        
        # 仅在真实数据集上收集 MSE 用于模型选择
        if self.args.val_on_real:
            if dataloader_idx == 1:  # 真实数据集
                self.epoch_mses.extend(batch_mses)
        else:
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
        
        # 在 sanity check 阶段不进行模型保存和最优更新
        if trainer.sanity_checking:
            loguru_logger.info(f"Sanity check 完成，跳过模型保存和最优更新")
            return
            
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        
        # 计算整个验证集的平均 MSE（仅来自真实数据集）
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        
        # 简化验证指标显示
        display_metrics = {'mse_real': avg_mse}  # 重命名为 mse_real 以区分
        
        # 提取 AUC 指标（来自真实数据集）
        for k in ['auc@5', 'auc@10', 'auc@20']:
            if k in metrics:
                display_metrics[k] = metrics[k].item()
        
        # 提取验证 Loss（来自真实数据集）
        for k in ['val/avg_loss_c', 'val/avg_loss_f', 'val/avg_loss']:
            if k in metrics:
                name = k.replace('val/avg_', '')
                display_metrics[name] = metrics[k].item()
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        loguru_logger.info(f"Epoch {epoch} 验证总结（真实数据） >> {metric_str}")
        
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
            
            # 兼容性处理：真实数据用 image1_gt，生成数据用 image1_origin
            ref_key = 'image1_gt' if 'image1_gt' in batch else 'image1_origin'
            img1_gt = (batch[ref_key][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
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
                loguru_logger.error(f"样本 {sample_name}: H_est 求逆失败（匹配点可能共线或不足）: {e}")
                # 失败时默认输出原始未配准图，保证 MSE 计算的是初始误差而非全黑图误差
                img1_result = img1.copy()
                
            # 计算有效区域 MSE
            try:
                res_f, orig_f = filter_valid_area(img1_result, img1_gt)
                mask = (res_f > 0)
                if np.any(mask):
                    mse = np.mean(((res_f[mask] / 255.0) - (orig_f[mask] / 255.0)) ** 2)
                else:
                    mse = np.mean(((img1_result / 255.0) - (img1_gt / 255.0)) ** 2)
            except:
                mse = np.mean(((img1_result / 255.0) - (img1_gt / 255.0)) ** 2)
            
            mses.append(mse)
            
            # 保存各模态图像
            cv2.imwrite(str(save_path / "fix.png"), img0)
            cv2.imwrite(str(save_path / "moving.png"), img1)
            cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)
            cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
            
            # 保存棋盘图：必须对比【模型配准后的FA (img1_result)】和【原始固定图CF (img0)】
            # 这样才能直观看到跨模态配准效果
            try:
                chessboard = create_chessboard(img1_result, img0)
                cv2.imwrite(str(save_path / "chessboard.png"), chessboard)
            except Exception as e:
                loguru_logger.error(f"样本 {sample_name}: 生成棋盘图失败: {e}")
            
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
    if args.pretrained_ckpt:
        loguru_logger.info(f"正在加载预训练权重: {args.pretrained_ckpt}")
    else:
        loguru_logger.info("未提供预训练权重，模型将从随机初始化开始训练")
        
    model = PL_LoFTR_WithDomainRand(
        config, 
        pretrained_ckpt=args.pretrained_ckpt, 
        use_domain_rand=args.use_domain_randomization,
        result_dir=str(result_dir)  # 传递结果目录
    )
    
    if args.use_domain_randomization:
        loguru_logger.info("域随机化增强已启用，将在训练时应用（包括sanity check阶段）")
        loguru_logger.info("将在训练开始时保存前2个batch的域随机化可视化")
    
    data_module = MultimodalDataModule(args, config)

    # 4. 初始化训练器
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.name)
    
    # 启用自定义验证逻辑回调
    val_callback = MultimodalValidationCallback(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # 启用早停机制：epoch 50 后，如果连续 5 次验证 (即 25 epoch) loss 不下降则停止
    # 注意：check_val_every_n_epoch=5，所以 patience=5 对应 5 次验证检查
    # 使用自定义 EarlyStopping，在 epoch 50 之前不计数
    # 既然是预训练权重微调，直接启用早停 (patience=5 对应 25个epoch)
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
