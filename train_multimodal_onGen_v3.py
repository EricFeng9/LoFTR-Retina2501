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

# 配置日志
loguru_logger = get_rank_zero_only_logger(logger)
loguru_logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
loguru_logger.add(sys.stderr, format=log_format, level="INFO")

# 屏蔽警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*main thread is not in main loop.*")

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        loguru_logger.opt(exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Plan v3: LoFTR Light-weight Fine-tuning (Freeze Backbone + Vessel Guidance)")
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'], help='配准模式')
    parser.add_argument('--name', '-n', type=str, default='loftr_v3_freeze_backbone', help='本次训练的名称')
    parser.add_argument('--batch_size', type=int, default=8, help='每个 GPU 的 Batch Size')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=512, help='图像输入尺寸')
    parser.add_argument('--pretrained_ckpt', type=str, 
                        default='/data/student/Fengjunming/LoFTR/third_party/MINIMA/weights/minima_loftr.ckpt', 
                        help='MINIMA 预训练权重路径')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率 (仅针对 Transformer)')
    
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=100, gpus='1')
    return parser.parse_args()

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loader_params = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}

    def setup(self, stage=None):
        # 训练集：Vessel Sigma 设为 0 (二值掩码)，因为我们使用 Strict Loss
        self.train_dataset = MultiModalDataset(
            DATA_ROOT, mode=self.args.mode, split='train', img_size=self.args.img_size, vessel_sigma=0.0)
        self.val_dataset = MultiModalDataset(
            DATA_ROOT, mode=self.args.mode, split='val', img_size=self.args.img_size, vessel_sigma=0.0)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, **self.loader_params)

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
        mask0 = data.get('vessel_mask0', data.get('mask0'))
        mask1 = data.get('vessel_mask1', data.get('mask1')) # [N, H, W]
        
        if mask0 is None or mask1 is None:
            return None

        # 下采样到 Coarse 分辨率
        scale = self.config['LOFTR']['RESOLUTION'][0]
        
        # 兼容不同输入形状
        if mask0.dim() == 4: mask0 = mask0.squeeze(1)
        if mask1.dim() == 4: mask1 = mask1.squeeze(1)

        # 使用 Max Pooling 确保血管不断裂 (比 Nearest 更保守)
        m0_c = torch.nn.functional.max_pool2d(mask0.unsqueeze(1).float(), kernel_size=scale, stride=scale).squeeze(1)
        m1_c = torch.nn.functional.max_pool2d(mask1.unsqueeze(1).float(), kernel_size=scale, stride=scale).squeeze(1)
        
        # 展平
        m0_flat = m0_c.flatten(-2) # [N, L0]
        m1_flat = m1_c.flatten(-2) # [N, L1]
        
        # 计算权重矩阵 [N, L0, L1]
        # w[i, j] = 1 if (i in vessel and j in vessel) else 0.001
        c_weight = (m0_flat.unsqueeze(-1) * m1_flat.unsqueeze(1))
        
        # 将 0 的区域设为 background_weight_strict
        c_weight = torch.where(c_weight > 0.5, torch.tensor(1.0, device=c_weight.device), torch.tensor(self.background_weight_strict, device=c_weight.device))
        
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
                    if w is not None: return w
                except Exception as e:
                    pass # Fallback
                return original_compute_c_weight(data)
                
            # 替换
            self.loss.compute_c_weight = patched_compute
            self.loss._is_patched_strict = True
            
        super()._trainval_inference(batch)

def main():
    args = parse_args()
    config = get_cfg_defaults()
    
    # 强制设置 config
    config.DATASET.MGDPT_IMG_RESIZE = args.img_size
    config.TRAINER.CANONICAL_LR = args.lr # 使用用户指定的 LR
    config.TRAINER.WARMUP_STEP = 200 # 减少 Warmup，因为是微调
    
    # 路径设置
    result_dir = Path(f"results/{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化
    pl.seed_everything(66)
    
    # 加载模型
    logger.info(f"Loading MINIMA weights from {args.pretrained_ckpt}")
    model = PL_LoFTR_V3(config, pretrained_ckpt=args.pretrained_ckpt)
    
    data_module = MultimodalDataModule(args)
    
    logger_tb = TensorBoardLogger(save_dir='logs/tb_logs_v3', name=args.name)
    
    # Checkpoint 回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mse',
        dirpath=result_dir / 'checkpoints',
        filename='loftr-v3-{epoch:02d}-{val_mse:.4f}',
        save_top_k=3,
        mode='min'
    )
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        logger=logger_tb,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
    )
    
    logger_tb.log_hyperparams(args)
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
