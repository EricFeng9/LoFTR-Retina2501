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
from data.FIVES_extract_v2.FIVES_extract_v2 import MultiModalDataset
from src.utils.plotting import make_matching_figures
from gen_data_enhance_v2 import save_batch_visualization
from src.utils.metrics import error_auc

# 导入真实数据集
from data.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# 复用 v2.1 中的 RealDatasetWrapper 等辅助类
# 为了完整性，这里重新定义一遍，避免 import 依赖问题
class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 1. 获取基础数据集输出
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        
        # 2. 转换范围 [0, 1]
        # fix_tensor 已经是 [0, 1] (transforms.ToTensor), moving 是 [-1, 1]
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2
        
        # 3. 灰度转换
        def to_gray(tensor):
            if tensor.shape[0] == 3:
                gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
                return gray.unsqueeze(0)
            return tensor
            
        fix_gray = to_gray(fix_tensor)
        moving_orig_gray = to_gray(moving_original_tensor)
        moving_gray = to_gray(moving_gt_tensor)
        
        # 保存原始未处理灰度图用于参考 (这里 raw 和 processed 已经一样了)
        moving_orig_gray_raw = moving_orig_gray.clone()
        
        fix_name = os.path.basename(fix_path)
        moving_name = os.path.basename(moving_path)
        
        # 5. T 矩阵转换 (H_0to1 是从 Fix 到 Moving)
        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except:
            T_fix_to_moving = T_0to1
            
        return {
            'image0': fix_gray,
            'image1': moving_orig_gray,
            'image1_gt': moving_gray,
            'image1_origin': moving_orig_gray_raw, # 原始未 CLAHE 的灰度图，用于 MSE 计算参考
            'image0_orig': fix_gray.clone(), # 真实数据不应用 DR，所以 orig 和 input 一致
            'image1_orig': moving_orig_gray.clone(),
            'T_0to1': T_fix_to_moving,
            'pair_names': (fix_name, moving_name),
            'dataset_name': 'MultiModal',
            # 补全生成数据集特有的键（占位符），确保 Batch Collation 成功
            'mask0': torch.ones_like(fix_gray),        # 真实数据假设全图有效
            'mask1': torch.ones_like(fix_gray),
            'vessel_mask0': torch.zeros_like(fix_gray), # 真实数据没有血管掩码
            'vessel_mask1': torch.zeros_like(fix_gray),
            'vessel_weight0': torch.zeros_like(fix_gray),
            'vessel_weight1': torch.zeros_like(fix_gray),
            'flip_h': False,
            'flip_v': False,
            'mask_quality': 1.0,
            'pair_id': idx
        }

# 数据集根目录硬编码
DATA_ROOT = "data/FIVES_extract_v2"

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
            # 1. 训练集 (生成数据)
            self.train_dataset_gen = MultiModalDataset(
                DATA_ROOT, mode=self.args.mode, split='train', img_size=self.args.img_size, vessel_sigma=self.args.vessel_sigma)
            
            # 2. 训练集 (真实数据)
            if self.args.mode == 'cfocta':
                train_base = CFOCTADataset(root_dir='data/CF_OCTA_v2_repaired', split='train', mode='cf2octa')
            elif self.args.mode == 'cffa':
                train_base = CFFADataset(root_dir='data/operation_pre_filtered_cffa', split='train', mode='fa2cf')
            elif self.args.mode == 'cfoct':
                train_base = CFOCTDataset(root_dir='data/operation_pre_filtered_cfoct', split='train', mode='cf2oct')
            elif self.args.mode == 'octfa':
                train_base = OCTFADataset(root_dir='data/operation_pre_filtered_octfa', split='train', mode='fa2oct')
            
            self.train_dataset_real = RealDatasetWrapper(train_base)
            
            # 3. 混合训练集
            from torch.utils.data import ConcatDataset
            self.train_dataset = ConcatDataset([self.train_dataset_gen, self.train_dataset_real])
            loguru_logger.info(f"训练集混合完成: 生成数据({len(self.train_dataset_gen)}) + 真实数据({len(self.train_dataset_real)}) = 总计 {len(self.train_dataset)}")
            
            # 验证集：根据参数选择
            if self.args.val_on_real:
                # 使用真实数据集验证
                if self.args.mode == 'cfocta':
                    val_base = CFOCTADataset(root_dir='data/CF_OCTA_v2_repaired', split='val', mode='cf2octa')
                elif self.args.mode == 'cffa':
                    val_base = CFFADataset(root_dir='data/operation_pre_filtered_cffa', split='val', mode='fa2cf')
                elif self.args.mode == 'cfoct':
                    val_base = CFOCTDataset(root_dir='data/operation_pre_filtered_cfoct', split='val', mode='cf2oct')
                elif self.args.mode == 'octfa':
                    val_base = OCTFADataset(root_dir='data/operation_pre_filtered_octfa', split='val', mode='fa2oct')
                
                # 包装真实数据集
                self.val_dataset_real = RealDatasetWrapper(val_base)
                
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


def compute_corner_error(H_est, H_gt, height, width):
    """计算角点误差 (Corner Error)"""
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

def create_chessboard(img1, img2, grid_size=4):
    """创建棋盘图"""
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

# 日志配置 (同 v2.1)
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
# 核心改动 1: CurriculumScheduler
# ==========================================


# ==========================================
# 核心改动 2: PL_LoFTR_V2 (恢复 Mask 输入)
# ==========================================
class PL_LoFTR_V3(PL_LoFTR):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, use_domain_rand=True, result_dir=None):
        super().__init__(config, pretrained_ckpt, profiler, dump_dir)
        self.use_domain_rand = use_domain_rand
        self.saved_batches = 0
        self.result_dir = result_dir
        
        self.vessel_loss_weight_scaler = 1.0
        
        # self.clahe = CLAHE_Preprocess(clip_limit=3.0, tile_grid_size=(8, 8)) # Moved to DataLoader
        
    def _trainval_inference(self, batch):
        # [Optimized] CLAHE and Normalization are now done in DataLoader
        # Just ensure data range safety
        for img_key in ['image0', 'image1', 'image1_gt']:
             if img_key in batch and batch[img_key].min() < 0:
                 batch[img_key] = (batch[img_key] + 1) / 2
                 
        if hasattr(self, 'vessel_loss_weight_scaler') and hasattr(self.loss, 'vessel_loss_weight_scaler'):
            self.loss.vessel_loss_weight_scaler = self.vessel_loss_weight_scaler
            
        super()._trainval_inference(batch)

    def training_step(self, batch, batch_idx):
        # 1. 记录可视化 (Domain Randomization 已经在 loader 中应用)
        # batch['image0/1'] 是已经 Augment 过的
        # batch['image0/1_orig'] 是仅 CLAHE 的 (如果有)
        
        if self.use_domain_rand:
            # [Requirement] 保存前两个 batch 的可视化信息
            is_viz_batch = (self.current_epoch == 0 and batch_idx < 2)
            
            if is_viz_batch:
                # 优先使用 Dataset 返回的 _orig 版本作为对比基准
                img0_orig = batch.get('image0_orig', batch['image0']).clone()
                img1_orig = batch.get('image1_orig', batch['image1']).clone()
                img0_aug = batch['image0']
                img1_aug = batch['image1']
            
                result_dir = Path(self.result_dir) if self.result_dir else Path(f"results/{self.config.DATASET.TRAINVAL_DATA_SOURCE}/{self.config.TRAINER.EXP_NAME}")
                result_dir.mkdir(parents=True, exist_ok=True)
                vessel_mask = batch.get('mask0', None)
                
                # A. 保存增强对比图 (包括要求的 comparison_*.png)
                save_batch_visualization(
                    img0_orig, img1_orig, img0_aug, img1_aug,
                    str(result_dir), epoch=self.current_epoch + 1, step=batch_idx + 1, 
                    batch_size=batch['image0'].shape[0], vessel_mask=vessel_mask
                )
                
                # B. 保存模型当前的配准效果可视化
                with torch.no_grad():
                    # 临时运行一次 validation_step 流程获取必需的绘图字典
                    old_training_mode = self.training
                    self.eval()
                    outputs = self.validation_step(batch, batch_idx)
                    self.train(old_training_mode)
                    
                    # 确定保存路径: epoch1_visualization/stepX_sampleY
                    vis_dir = result_dir / f"epoch1_visualization" / f"step{batch_idx+1}_registration"
                    self._save_training_registration_viz(batch, outputs, vis_dir)

        # 2. 调用基类的推理和 loss 计算 
        # (此时 batch['image0'] 已经是增强过的)
        return super().training_step(batch, batch_idx)


    def _save_training_registration_viz(self, batch, outputs, save_dir):
        """在训练期间保存详细的注册效果图"""
        save_dir.mkdir(parents=True, exist_ok=True)
        batch_size = batch['image0'].shape[0]
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        
        for i in range(min(batch_size, 2)): # 每个 batch 只保存前 2 个样本防止磁盘爆满
            sample_dir = save_dir / f"sample{i}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            H_est = H_ests[i]
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()
            
            # 保存基础图和棋盘图
            cv2.imwrite(str(sample_dir / "fix_aug.png"), img0)
            cv2.imwrite(str(sample_dir / "moving_aug.png"), img1)
            cv2.imwrite(str(sample_dir / "moving_result.png"), img1_result)
            
            try:
                from train_multimodal_onGen_v2_1 import create_chessboard
                cb = create_chessboard(img1_result, img0)
                cv2.imwrite(str(sample_dir / "chessboard.png"), cb)
            except: pass
            
            # 保存 matches (如果 figures 已生成)
            try:
                mode = self.config.TRAINER.PLOT_MODE
                if 'figures' in outputs and mode in outputs['figures'] and len(outputs['figures'][mode]) > i:
                    fig = outputs['figures'][mode][i]
                    fig.savefig(str(sample_dir / "matches.png"), bbox_inches='tight')
                    # plt.close(fig) # 注意：如果是训练中途，plt 关闭需谨慎
            except: pass
            
            # 检测点图
            img0_kpts = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
            if 'm_bids' in batch:
                mask_i = (batch['m_bids'] == i)
                kpts0_m = batch['mkpts0_f'][mask_i].cpu().numpy()
                for pt in kpts0_m:
                    cv2.circle(img0_kpts, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            cv2.imwrite(str(sample_dir / "fix_with_matches.png"), img0_kpts)

    def validation_epoch_end(self, outputs):
        """
        核心修正：确保评估指标只来自真实数据
        outputs: 如果启用 val_on_real，则为 [outputs_gen, outputs_real]
        """
        if isinstance(outputs, list) and len(outputs) > 1:
            # 只要有多验证集，主指标 (auc@5/10/20) 强制只取第二个验证集（真实数据）
            # 这样 PL 自动记录的 'auc@10' 就是真实的 auc，
            # 从而 EarlyStopping 和 Callback 拿到的全是真实数据指标
            real_outputs = outputs[1]
            return super().validation_epoch_end(real_outputs)
        else:
            return super().validation_epoch_end(outputs)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._trainval_inference(batch)
        ret_dict, _ = self._compute_metrics(batch)
        
        # 处理多验证集时的绘图间隔
        num_val_batches = self.trainer.num_val_batches
        if isinstance(num_val_batches, list):
            num_val_batches = num_val_batches[dataloader_idx]
        val_plot_interval = max(num_val_batches // self.n_vals_plot, 1)
        
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if getattr(self, 'force_viz', False):
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        # --- 提取所有候选匹配点 (用于判断是取点有问题还是匹配有问题) ---
        all_candidates = []
        if 'conf_matrix' in batch:
            conf_matrix = batch['conf_matrix'] # [N, L, S]
            thr = self.matcher.coarse_matching.thr
            hw0c = batch['hw0_c']
            hw1c = batch['hw1_c']
            # 注意：hw0_i 可能在 batch 里是 [H, W]
            scale0 = batch['image0'].shape[2] / hw0c[1]
            scale1 = batch['image1'].shape[2] / hw1c[1]
            
            for b in range(conf_matrix.shape[0]):
                # image0 中 max_conf > thr 的点
                mask0 = conf_matrix[b].max(dim=1)[0] > thr
                indices0 = torch.where(mask0)[0]
                kpts0 = torch.stack([indices0 % hw0c[1], indices0 // hw0c[1]], dim=1).float() * scale0
                
                # image1 中 max_conf > thr 的点
                mask1 = conf_matrix[b].max(dim=0)[0] > thr
                indices1 = torch.where(mask1)[0]
                kpts1 = torch.stack([indices1 % hw1c[1], indices1 // hw1c[1]], dim=1).float() * scale1
                
                all_candidates.append({
                    'kpts0': kpts0.detach().cpu().numpy(),
                    'kpts1': kpts1.detach().cpu().numpy()
                })

        return {
            **ret_dict,
            'kpts_candidates': all_candidates,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }

# 复用 MultimodalValidationCallback (完全一致)
class MultimodalValidationCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val = -1.0 # 既然修复了 GT 矩阵尺度问题，改回使用 AUC，越高越好
        self.result_dir = Path(f"results/{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []
        self.epoch_maces = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_maces = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 每个 epoch 都会运行，但不再保存结果到磁盘，仅计算指标
        batch_mses, batch_maces = self._process_batch(trainer, pl_module, batch, outputs, None, save_images=False)
        
        if self.args.val_on_real:
            if dataloader_idx == 1:
                self.epoch_mses.extend(batch_mses)
                self.epoch_maces.extend(batch_maces)
        else:
            self.epoch_mses.extend(batch_mses)
            self.epoch_maces.extend(batch_maces)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        display_metrics = {}
        for k in ['loss', 'train/loss_c', 'train/loss_f']:
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
        if not self.epoch_mses: return
        
        # [Requirement] 允许在 Sanity Check 期间输出结果，方便用户确认初始状态
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        avg_mace = sum(self.epoch_maces) / len(self.epoch_maces) if self.epoch_maces else float('inf')
        
        if trainer.sanity_checking:
            loguru_logger.info(f"--- [Sanity Check] 初始状态验证总结 ---")
            loguru_logger.info(f"MSE: {avg_mse:.6f} | MACE: {avg_mace:.4f}")
            # 为 Sanity Check 触发一次特殊的可视化
            self._trigger_visualization(trainer, pl_module, is_best=False, epoch=0)
            return

        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        
        display_metrics = {'mse_real': avg_mse, 'mace_real': avg_mace}
        for k in ['auc@5', 'auc@10', 'auc@20']:
            if k in metrics: display_metrics[k] = metrics[k].item()
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        loguru_logger.info(f"Epoch {epoch} 验证总结（真实数据） >> {metric_str}")
        
        pl_module.log("val_mse", avg_mse, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("val_mace", avg_mace, on_epoch=True, prog_bar=True, logger=True)
        
        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")
        with open(latest_path / "log.txt", "w") as f:
            f.write(f"Epoch: {epoch}\nLatest MSE: {avg_mse:.6f}\nLatest MACE: {avg_mace:.4f}")
            
        # [Update] 既然修复了尺度 Bug，改回使用 AUC 作为 Best 评价指标
        # 优化：使用 AUC@5, 10, 20 的平均值作为综合指标，比单看 AUC@10 更稳健 (防止运气导致的突跳)
        is_best = False
        auc5 = display_metrics.get('auc@5', 0.0)
        auc10 = display_metrics.get('auc@10', 0.0)
        auc20 = display_metrics.get('auc@20', 0.0)
        combined_auc = (auc5 + auc10 + auc20) / 3.0
        
        if combined_auc > self.best_val:
            self.best_val = combined_auc
            is_best = True
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\nBest Combined AUC: {combined_auc:.4f}\nAUC@10: {auc10:.4f}\nMSE: {avg_mse:.6f}\nMACE: {avg_mace:.4f}")
            loguru_logger.info(f"发现新的最优模型! Epoch {epoch}, 综合 AUC: {combined_auc:.4f} (auc@10: {auc10:.4f})")

        # 管理可视化文件夹的保存
        if is_best or (epoch % 5 == 0):
            self._trigger_visualization(trainer, pl_module, is_best, epoch)

    def _trigger_visualization(self, trainer, pl_module, is_best, epoch):
        loguru_logger.info(f"正在为 Epoch {epoch} 生成可视化结果...")
        pl_module.force_viz = True  # 强制模型在 validation_step 中生成 matching figures
        
        target_dirs = []
        if is_best:
            target_dirs.append(self.result_dir / f"epoch{epoch}_best")
        elif epoch % 5 == 0:
            target_dirs.append(self.result_dir / f"epoch{epoch}")
        
        for d in target_dirs: d.mkdir(parents=True, exist_ok=True)
        
        # 获取验证数据加载器
        val_dataloaders = trainer.val_dataloaders
        if val_dataloaders is None: return
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]
            
        pl_module.eval()
        with torch.no_grad():
            for dl_idx, dl in enumerate(val_dataloaders):
                dl_name = "generated" if dl_idx == 0 else "real"
                # 对该数据加载器下的所有 batch 进行可视化
                for batch_idx, batch in enumerate(dl):
                    # 将 batch 移动到模型设备
                    batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    # 运行推理 (转发到模型的 validation_step)
                    outputs = pl_module.validation_step(batch, batch_idx, dl_idx)
                    
                    # 针对每个目标目录保存可视化
                    for target_dir in target_dirs:
                        sub_dir = target_dir
                        if self.args.val_on_real:
                            sub_dir = target_dir / ("generated" if dl_idx == 0 else "real")
                        sub_dir.mkdir(parents=True, exist_ok=True)
                        self._process_batch(trainer, pl_module, batch, outputs, sub_dir, save_images=True)
                        
        pl_module.force_viz = False # 恢复

    def _process_batch(self, trainer, pl_module, batch, outputs, epoch_dir, save_images=False):
        # 简化版实现，同 v2.1
        batch_size = batch['image0'].shape[0]
        mses = []
        maces = []
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        Ts_gt = batch['T_0to1'].cpu().numpy()
        pair_names0 = batch['pair_names'][0]
        pair_names1 = batch['pair_names'][1]
        
        # 直接使用本地定义的辅助函数

        for i in range(batch_size):
            H_est = H_ests[i]
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            ref_key = 'image1_gt' if 'image1_gt' in batch else 'image1_origin'
            img1_gt = (batch[ref_key][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            # 使用图像实际尺寸
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
            except:
                mse = 0.0
            mses.append(mse)
            
            mace = compute_corner_error(H_est, Ts_gt[i], h, w)
            maces.append(mace)
            
            if not save_images:
                continue
                
            sample_name = f"{Path(pair_names0[i]).stem}_vs_{Path(pair_names1[i]).stem}"
            save_path = epoch_dir / sample_name
            save_path.mkdir(parents=True, exist_ok=True)

            # 保存各个结果图像
            cv2.imwrite(str(save_path / "fix.png"), img0)
            cv2.imwrite(str(save_path / "moving.png"), img1)
            cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)
            cv2.imwrite(str(save_path / "moving_result.png"), img1_result)

            # --- 新增：在 fix 和 moving 上画出所有候选点 (白色) 和 最终匹配点 (红色) ---
            img0_kpts = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
            img1_kpts = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            
            # 1. 先画出所有候选点 (白色小点)
            if 'kpts_candidates' in outputs and len(outputs['kpts_candidates']) > i:
                cands = outputs['kpts_candidates'][i]
                for pt in cands['kpts0']:
                    cv2.circle(img0_kpts, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
                for pt in cands['kpts1']:
                    cv2.circle(img1_kpts, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)

            # 2. 再画出匹配成功的点 (红色，略大一点)
            if 'm_bids' in batch:
                mask_i = (batch['m_bids'] == i)
                kpts0_m = batch['mkpts0_f'][mask_i].cpu().numpy()
                kpts1_m = batch['mkpts1_f'][mask_i].cpu().numpy()
                
                for pt in kpts0_m:
                    cv2.circle(img0_kpts, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                for pt in kpts1_m:
                    cv2.circle(img1_kpts, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            
            cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_kpts)
            cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_kpts)

            # --- 新增：保存血管掩码（如果存在） ---
            if 'mask0' in batch:
                m0 = batch['mask0'][i].cpu().numpy()
                m1 = batch['mask1'][i].cpu().numpy()
                if m0.ndim == 3: m0 = m0[0]
                if m1.ndim == 3: m1 = m1[0]
                cv2.imwrite(str(save_path / "mask0_vessel.png"), (m0 * 255).astype(np.uint8))
                cv2.imwrite(str(save_path / "mask1_vessel_warped.png"), (m1 * 255).astype(np.uint8))
                
            try:
                cb = create_chessboard(img1_result, img0)
                cv2.imwrite(str(save_path / "chessboard.png"), cb)
            except: pass
            
            # 保存 matches
            try:
                mode = pl_module.config.TRAINER.PLOT_MODE
                if 'figures' in outputs and mode in outputs['figures'] and len(outputs['figures'][mode]) > i:
                    fig = outputs['figures'][mode][i]
                    fig.savefig(str(save_path / "matches.png"), bbox_inches='tight')
                    plt.close(fig)
            except: pass
            
        return mses, maces

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

def parse_args():
    parser = argparse.ArgumentParser(description="LoFTR V2.3 Minimalist Vessel-Aware")
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'])
    # 默认名改为 v2.3
    parser.add_argument('--name', '-n', type=str, default='loftr_v2_3_1mix_aligned', help='训练名称') 
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--vessel_sigma', type=float, default=6.0)
    parser.add_argument('--pretrained_ckpt', type=str, default='weights/outdoor_ds.ckpt')
    parser.add_argument('--start_point', type=str, default=None, help='训练断点路径 (.ckpt)')
    parser.add_argument('--use_domain_randomization', action='store_true', default=True)
    parser.add_argument('--val_on_real', action='store_true', default=True)
    parser.add_argument('--main_cfg_path', type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=500, gpus='1')
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_cfg_defaults()
    if args.main_cfg_path: config.merge_from_file(args.main_cfg_path)
    
    result_dir = Path(f"results/{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    loguru_logger.add(result_dir / "log.txt", enqueue=True, mode="a")
    
    config.DATASET.MGDPT_IMG_RESIZE = args.img_size
    config.LOFTR.RESOLUTION = (8, 2)
    config.TRAINER.SEED = 66
    pl.seed_everything(config.TRAINER.SEED)
    
    _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1) * getattr(args, 'num_nodes', 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    
    model = PL_LoFTR_V3(
        config, 
        pretrained_ckpt=args.pretrained_ckpt, 
        use_domain_rand=args.use_domain_randomization,
        result_dir=str(result_dir)
    )
    
    # 强制全权重加载 (覆盖 V2.3 的 Random Init 策略)
    # 如果用户提供了 pretrained_ckpt，我们希望利用这里面所有的知识
    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 移除 'matcher.' 前缀 (如果存在)
        state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}
        
        # 加载所有通过 key 匹配的权重 (Backbone + Transformer)
        # strict=False 允许部分不匹配 (如检测头参数可能不同)
        keys = model.matcher.load_state_dict(state_dict, strict=False)
        loguru_logger.info(f"已强制加载全量预训练权重 (Backbone + Transformer)")
        loguru_logger.info(f"Missing keys: {keys.missing_keys}")
        
    
    data_module = MultimodalDataModule(args, config)
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.name)
    val_callback = MultimodalValidationCallback(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # [Monitor Change] MSE -> AUC@10
    # 既然尺度问题已解决，AUC@10 是最能反应配准成功率的指标
    early_stop_callback = DelayedEarlyStopping(
        start_epoch=0, 
        monitor='auc@10', 
        mode='max', 
        patience=10, 
        min_delta=0.0001
    )
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        min_epochs=0,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        callbacks=[val_callback, lr_monitor, early_stop_callback],
        logger=tb_logger,
        plugins=DDPPlugin(find_unused_parameters=False) if _n_gpus > 1 else None,
        replace_sampler_ddp=True,
        resume_from_checkpoint=args.start_point
    )
    
    loguru_logger.info(f"开始混合训练 V2.3 (Gen + Real): {args.name}")
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
