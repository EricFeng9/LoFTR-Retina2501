import sys
import os
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
from data.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from data.operation_pre_filtered_cffa.operation_pre_filtered_octfa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset
from src.utils.plotting import make_matching_figures

# 数据集根目录
DATA_ROOT_CFFA = "/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cffa"
DATA_ROOT_CFOCT = "/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cfoct"
DATA_ROOT_OCTFA = "/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_octfa"
DATA_ROOT_CFOCTA = "/data/student/Fengjunming/LoFTR/data/CF_OCTA_v2_repaired"

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
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(description="LoFTR 多模态眼底图像配准训练脚本 (V2)")
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'], help='配准模式')
    parser.add_argument('--name', '-n', type=str, default='loftr_multimodal', help='本次训练的名称')
    parser.add_argument('--batch_size', type=int, default=4, help='每个 GPU 的 Batch Size')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=512, help='图像输入尺寸')
    parser.add_argument('--main_cfg_path', type=str, default=None, help='主配置文件路径')
    
    # 自动添加 Lightning Trainer 参数 (如 --gpus, --max_epochs, --accelerator 等)
    parser = pl.Trainer.add_argparse_args(parser)
    
    # 设置 Lightning 参数的默认值
    parser.set_defaults(max_epochs=100, gpus='1')
    
    return parser.parse_args()

class LoFTRDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, split='train', img_size=512):
        self.dataset = dataset
        self.split = split
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def _get_random_affine(self, rng=None):
        if rng is None:
            rng = np.random
        angle = rng.uniform(15, 45)
        if rng.rand() > 0.5:
            angle = -angle
        scale = rng.uniform(0.8, 1.2)
        tx = rng.uniform(-0.1, 0.1) * self.img_size
        ty = rng.uniform(-0.1, 0.1) * self.img_size
        center = (self.img_size // 2, self.img_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        return M.astype(np.float32)

    def __getitem__(self, idx):
        # real dataset 返回: cond_tensor [0,1], tgt_tensor [-1,1], cp, tp
        cond_tensor, tgt_tensor, cp, tp = self.dataset[idx]
        
        # LoFTR 需要 [0, 1] 灰度图
        # 如果 cond_tensor 是 3 通道的，转为单通道灰度
        if cond_tensor.shape[0] == 3:
            img0 = (0.299 * cond_tensor[0] + 0.587 * cond_tensor[1] + 0.114 * cond_tensor[2]).numpy()
        else:
            img0 = cond_tensor[0].numpy()
            
        # tgt_tensor 从 [-1, 1] 转为 [0, 1]
        tgt_tensor_01 = (tgt_tensor + 1.0) / 2.0
        if tgt_tensor_01.shape[0] == 3:
            img1 = (0.299 * tgt_tensor_01[0] + 0.587 * tgt_tensor_01[1] + 0.114 * tgt_tensor_01[2]).numpy()
        else:
            img1 = tgt_tensor_01[0].numpy()
        
        img1_origin = img1.copy()
        
        if self.split in ['train', 'val']:
            if self.split == 'val':
                rng = np.random.RandomState(idx)
                T = self._get_random_affine(rng)
            else:
                T = self._get_random_affine()
                
            img1_warped = cv2.warpAffine(img1, T, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR)
            H = np.eye(3, dtype=np.float32)
            H[:2, :] = T
            
            data = {
                'image0': torch.from_numpy(img0).float()[None],
                'image1': torch.from_numpy(img1_warped).float()[None],
                'image1_origin': torch.from_numpy(img1_origin).float()[None],
                'T_0to1': torch.from_numpy(H),
            }
        else:
            data = {
                'image0': torch.from_numpy(img0).float()[None],
                'image1': torch.from_numpy(img1).float()[None],
                'image1_origin': torch.from_numpy(img1_origin).float()[None],
            }
            
        data.update({
            'dataset_name': 'MultiModal',
            'pair_id': idx,
            'pair_names': (os.path.basename(cp), os.path.basename(tp))
        })
        return data

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
        mode = self.args.mode
        if stage == 'fit' or stage is None:
            if mode == 'cffa':
                train_base = CFFADataset(root_dir=DATA_ROOT_CFFA, split='train', mode='cf2fa')
                val_base = CFFADataset(root_dir=DATA_ROOT_CFFA, split='val', mode='cf2fa')
            elif mode == 'cfoct':
                train_base = CFOCTDataset(root_dir=DATA_ROOT_CFOCT, split='train', mode='cf2oct')
                val_base = CFOCTDataset(root_dir=DATA_ROOT_CFOCT, split='val', mode='cf2oct')
            elif mode == 'octfa':
                train_base = OCTFADataset(root_dir=DATA_ROOT_OCTFA, split='train', mode='oct2fa')
                val_base = OCTFADataset(root_dir=DATA_ROOT_OCTFA, split='val', mode='oct2fa')
            elif mode == 'cfocta':
                train_base = CFOCTADataset(root_dir=DATA_ROOT_CFOCTA, split='train', mode='cf2octa')
                val_base = CFOCTADataset(root_dir=DATA_ROOT_CFOCTA, split='val', mode='cf2octa')
            else:
                raise ValueError(f"Unknown mode: {mode}")

            self.train_dataset = LoFTRDatasetWrapper(
                train_base, split='train', img_size=self.args.img_size)
            self.val_dataset = LoFTRDatasetWrapper(
                val_base, split='val', img_size=self.args.img_size)

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
        batch_mses = self._save_batch_results(trainer, pl_module, batch, epoch_dir)
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

    def _save_batch_results(self, trainer, pl_module, batch, epoch_dir):
        # 确保 batch 中的 Tensor 在正确的设备上
        batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 确保 batch 中包含推理结果和指标
        pl_module.eval()
        with torch.no_grad():
            # 1. 运行推理和损失计算
            pl_module._trainval_inference(batch)
            # 2. 显式调用指标计算以获取 H_est 和匹配点
            from src.utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, pl_module.config)
        pl_module.train()

        batch_size = batch['image0'].shape[0]
        mses = []
        
        # 获取真值 H_gt
        H_gts = batch['T_0to1'].cpu().numpy()
        # 获取估计的单应矩阵 H_est (由 metrics.py 中的 compute_homography_errors 提供)
        H_ests = batch['H_est']
        
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
            
            H_est = H_ests[i]
            
            # 计算 moving_result: 将 img1 使用 H_est 的逆变换回 img0 空间
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except Exception as e:
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
            
            # 生成并保存 Matches 可视化图 (针对单个样本生成)
            try:
                # 构造单样本数据字典以调用 make_matching_figures
                # 注意：make_matching_figures 期望 batch 格式，所以这里我们需要构造一个迷你的 batch
                mini_batch = {
                    'image0': batch['image0'][i:i+1],
                    'image1': batch['image1'][i:i+1],
                    'mkpts0_f': batch['mkpts0_f'][batch['m_bids'] == i],
                    'mkpts1_f': batch['mkpts1_f'][batch['m_bids'] == i],
                    'm_bids': torch.zeros_like(batch['m_bids'][batch['m_bids'] == i]),
                    'dataset_name': batch['dataset_name'],
                    'epi_errs': batch['epi_errs'][batch['m_bids'] == i]
                }
                # 如果有 scale，也需要切片
                if 'scale0' in batch: mini_batch['scale0'] = batch['scale0'][i:i+1]
                if 'scale1' in batch: mini_batch['scale1'] = batch['scale1'][i:i+1]
                if 'conf_matrix_gt' in batch: mini_batch['conf_matrix_gt'] = batch['conf_matrix_gt'][i:i+1]

                import matplotlib.pyplot as plt
                figures = make_matching_figures(mini_batch, pl_module.config, mode='evaluation')
                if 'evaluation' in figures:
                    fig = figures['evaluation'][0]
                    fig.savefig(str(save_path / "matches.png"), bbox_inches='tight')
                    plt.close(fig)
                plt.close('all')
            except Exception as e:
                pass
                
        return mses

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
    
    # 启用早停机制：20 epoch 后，如果连续 8 次验证 (即 40 epoch) loss 不下降则停止
    # 注意：check_val_every_n_epoch=5，所以 patience=8 对应 8 次验证检查
    early_stop_callback = EarlyStopping(
        monitor='val_mse', 
        patience=8,
        verbose=True,
        mode='min',
        min_delta=0.0001
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        min_epochs=20, # 确保至少训练 20 epoch
        check_val_every_n_epoch=5, # 每 5 个 epoch 验证一次
        num_sanity_val_steps=-1, # 训练前跑完整验证
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
