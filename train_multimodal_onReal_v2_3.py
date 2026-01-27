import sys
import os
import shutil
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
from src.utils.plotting import make_matching_figures
from src.utils.metrics import error_auc

# 导入真实数据集
from data.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# ==========================================
# 辅助类: RealDatasetWrapper (格式转换)
# ==========================================
class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        # 数据集返回的已是归一化到 [0, 1] 的 fix，和 [-1, 1] 的 moving
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2
        
        # 转换为灰度图 [1, H, W]
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
        
        fix_name = os.path.basename(fix_path)
        moving_name = os.path.basename(moving_path)
        
        # 数据集内部计算的 T_0to1 是从 Moving 到 Fix 的变换 (warpPerspective 常用)
        # 但 LoFTR 默认输出是从 Image0(Fix) -> Image1(Moving) 的变换
        # 所以这里取逆，保持与 train_multimodal_onGen_v2_3 一致
        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except:
            T_fix_to_moving = T_0to1
            
        return {
            'image0': fix_gray,
            'image1': moving_orig_gray,
            'image1_gt': moving_gray,
            'T_0to1': T_fix_to_moving,
            'pair_names': (fix_name, moving_name),
            'dataset_name': 'MultiModal'
        }

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
            # 训练集和验证集都使用真实数据
            if self.args.mode == 'cfocta':
                train_base = CFOCTADataset(root_dir='data/CF_OCTA_v2_repaired', split='train', mode='cf2octa')
                val_base = CFOCTADataset(root_dir='data/CF_OCTA_v2_repaired', split='val', mode='cf2octa')
            elif self.args.mode == 'cffa':
                train_base = CFFADataset(root_dir='data/operation_pre_filtered_cffa', split='train', mode='fa2cf')
                val_base = CFFADataset(root_dir='data/operation_pre_filtered_cffa', split='val', mode='fa2cf')
            elif self.args.mode == 'cfoct':
                train_base = CFOCTDataset(root_dir='data/operation_pre_filtered_cfoct', split='train', mode='cf2oct')
                val_base = CFOCTDataset(root_dir='data/operation_pre_filtered_cfoct', split='val', mode='cf2oct')
            elif self.args.mode == 'octfa':
                train_base = OCTFADataset(root_dir='data/operation_pre_filtered_octfa', split='train', mode='fa2oct')
                val_base = OCTFADataset(root_dir='data/operation_pre_filtered_octfa', split='val', mode='fa2oct')
            
            self.train_dataset = RealDatasetWrapper(train_base)
            self.val_dataset = RealDatasetWrapper(val_base)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, **self.loader_params)

# ==========================================
# 辅助工具
# ==========================================
def filter_valid_area(img1, img2):
    """筛选有效区域：只保留两张图片都不为纯黑像素的部分"""
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
    filtered_img1[~valid_mask_cropped] = 0
    filtered_img2[~valid_mask_cropped] = 0
    return filtered_img1, filtered_img2

def compute_corner_error(H_est, H_gt, height, width):
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    corners_est_homo = (H_est @ corners_homo.T).T
    corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
    try:
        errors = np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1))
        mace = np.mean(errors)
    except:
        mace = float('inf')
    return mace

class CLAHE_Preprocess:
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, img_tensor):
        device = img_tensor.device
        B = img_tensor.shape[0]
        res = []
        for b in range(B):
            img_np = (img_tensor[b, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            img_clahe = self.clahe.apply(img_np)
            res.append(torch.from_numpy(img_clahe).float() / 255.0)
        return torch.stack(res).unsqueeze(1).to(device)

def create_chessboard(img1, img2, grid_size=4):
    H, W = img1.shape
    cell_h = H // grid_size
    cell_w = W // grid_size
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    return chessboard

# 日志配置
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

# ==========================================
# 核心模型: PL_LoFTR_V3 (与 Gen 版本一致，但关闭 Rand)
# ==========================================
class PL_LoFTR_V3(PL_LoFTR):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, result_dir=None):
        super().__init__(config, pretrained_ckpt, profiler, dump_dir)
        self.result_dir = result_dir
        # 在真实图像上训练，默认不使用额外的血管损失权重（因为数据集没有 mask）
        self.vessel_loss_weight_scaler = 1.0 
        # self.clahe = CLAHE_Preprocess(clip_limit=3.0, tile_grid_size=(8, 8))
        
    def _trainval_inference(self, batch):
        # 统一预处理 (移除 CLAHE)
        for img_key in ['image0', 'image1']:
            if img_key in batch:
                if batch[img_key].min() < 0:
                    batch[img_key] = (batch[img_key] + 1) / 2
                # batch[img_key] = self.clahe(batch[img_key])
        
        if 'image1_gt' in batch:
            if batch['image1_gt'].min() < 0:
                batch['image1_gt'] = (batch['image1_gt'] + 1) / 2
            # batch['image1_gt'] = self.clahe(batch['image1_gt'])

        if hasattr(self, 'vessel_loss_weight_scaler') and hasattr(self.loss, 'vessel_loss_weight_scaler'):
            self.loss.vessel_loss_weight_scaler = self.vessel_loss_weight_scaler
            
        super()._trainval_inference(batch)

    def training_step(self, batch, batch_idx):
        # 在真实图像训练中，去除了 Domain Randomization (直接调用基类)
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._trainval_inference(batch)
        ret_dict, _ = self._compute_metrics(batch)
        
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if getattr(self, 'force_viz', False):
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }

# ==========================================
# 回调逻辑: MultimodalValidationCallback
# ==========================================
class MultimodalValidationCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val = -1.0
        self.result_dir = Path(f"results/{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []
        self.epoch_maces = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_maces = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        batch_mses, batch_maces = self._process_batch(trainer, pl_module, batch, outputs, None, save_images=False)
        self.epoch_mses.extend(batch_mses)
        self.epoch_maces.extend(batch_maces)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        display_metrics = {}
        for k in ['loss', 'train/loss_c', 'train/loss_f']:
            epoch_key = f"{k}_epoch"
            if epoch_key in metrics:
                display_metrics[k.replace('train/', '')] = metrics[epoch_key].item()
        if display_metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
            loguru_logger.info(f"Epoch {epoch} 训练总结 >> {metric_str}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.epoch_mses: return
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        avg_mace = sum(self.epoch_maces) / len(self.epoch_maces) if self.epoch_maces else float('inf')
        
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        
        display_metrics = {'mse': avg_mse, 'mace': avg_mace}
        for k in ['auc@5', 'auc@10', 'auc@20']:
            if k in metrics: display_metrics[k] = metrics[k].item()
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        loguru_logger.info(f"Epoch {epoch} 验证总结 >> {metric_str}")
        
        pl_module.log("val_mse", avg_mse, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("val_mace", avg_mace, on_epoch=True, prog_bar=True, logger=True)
        
        # 保存最新模型
        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")
            
        # 评价最优模型
        auc5 = display_metrics.get('auc@5', 0.0)
        auc10 = display_metrics.get('auc@10', 0.0)
        auc20 = display_metrics.get('auc@20', 0.0)
        combined_auc = (auc5 + auc10 + auc20) / 3.0
        
        if combined_auc > self.best_val:
            self.best_val = combined_auc
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\nBest Combined AUC: {combined_auc:.4f}\nAUC@10: {auc10:.4f}\nMSE: {avg_mse:.6f}\nMACE: {avg_mace:.4f}")
            loguru_logger.info(f"发现新的最优模型! Epoch {epoch}, 综合 AUC: {combined_auc:.4f}")

        if combined_auc > 0 or epoch % 10 == 0:
            self._trigger_visualization(trainer, pl_module, combined_auc > self.best_val, epoch)

    def _trigger_visualization(self, trainer, pl_module, is_best, epoch):
        pl_module.force_viz = True
        target_dir = self.result_dir / (f"epoch{epoch}_best" if is_best else f"epoch{epoch}")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        val_dataloader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx > 5: break # 只可视化前几个 batch
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = pl_module.validation_step(batch, batch_idx)
                self._process_batch(trainer, pl_module, batch, outputs, target_dir, save_images=True)
        pl_module.force_viz = False

    def _process_batch(self, trainer, pl_module, batch, outputs, epoch_dir, save_images=False):
        batch_size = batch['image0'].shape[0]
        mses, maces = [], []
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        Ts_gt = batch['T_0to1'].cpu().numpy()
        
        for i in range(batch_size):
            H_est = H_ests[i]
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1_gt = (batch['image1_gt'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()
            
            try:
                res_f, orig_f = filter_valid_area(img1_result, img1_gt)
                mask = (res_f > 0)
                mse = np.mean(((res_f[mask]/255.)-(orig_f[mask]/255.))**2) if np.any(mask) else 0.0
            except: mse = 0.0
            mses.append(mse)
            maces.append(compute_corner_error(H_est, Ts_gt[i], h, w))
            
            if save_images:
                sample_name = f"{Path(batch['pair_names'][0][i]).stem}_vs_{Path(batch['pair_names'][1][i]).stem}"
                save_path = epoch_dir / sample_name
                save_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path / "fix.png"), img0)
                cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
                try:
                    cb = create_chessboard(img1_result, img0)
                    cv2.imwrite(str(save_path / "chessboard.png"), cb)
                except: pass
                if 'figures' in outputs and len(outputs['figures'][pl_module.config.TRAINER.PLOT_MODE]) > i:
                    fig = outputs['figures'][pl_module.config.TRAINER.PLOT_MODE][i]
                    fig.savefig(str(save_path / "matches.png"), bbox_inches='tight')
        return mses, maces

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

def parse_args():
    parser = argparse.ArgumentParser(description="LoFTR V2.3 Real-Data Baseline")
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'])
    parser.add_argument('--name', '-n', type=str, default='loftr_onReal_v2_3_baseline', help='训练名称')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--pretrained_ckpt', type=str, default='weights/outdoor_ds.ckpt')
    parser.add_argument('--start_point', type=str, default=None)
    parser.add_argument('--main_cfg_path', type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=200, gpus='1')
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_cfg_defaults()
    if args.main_cfg_path: config.merge_from_file(args.main_cfg_path)
    
    result_dir = Path(f"results/onReal_{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    loguru_logger.add(result_dir / "log.txt", enqueue=True, mode="a")
    
    config.DATASET.MGDPT_IMG_RESIZE = args.img_size
    config.LOFTR.RESOLUTION = (8, 2)
    config.TRAINER.SEED = 66
    pl.seed_everything(config.TRAINER.SEED)
    
    _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    
    model = PL_LoFTR_V3(
        config, 
        pretrained_ckpt=args.pretrained_ckpt, 
        result_dir=str(result_dir)
    )
    
    # 强制全权重加载
    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}
        model.matcher.load_state_dict(state_dict, strict=False)
        loguru_logger.info(f"已加载全量预训练权重")
    
    data_module = MultimodalDataModule(args, config)
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=f"onReal_{args.name}")
    
    early_stop_callback = DelayedEarlyStopping(
        start_epoch=100,
        monitor='auc@10', 
        mode='max', 
        patience=15, 
        min_delta=0.0001
    )
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        num_sanity_val_steps=-1,  # 启动前跑完整验证
        check_val_every_n_epoch=1, # 每一轮都验证
        callbacks=[MultimodalValidationCallback(args), LearningRateMonitor(logging_interval='step'), early_stop_callback],
        logger=tb_logger,
        plugins=DDPPlugin(find_unused_parameters=False) if _n_gpus > 1 else None,
        replace_sampler_ddp=True,
        resume_from_checkpoint=args.start_point
    )
    
    loguru_logger.info(f"开始真实数据基准训练: {args.name}")
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
