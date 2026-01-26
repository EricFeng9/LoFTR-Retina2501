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
from data.FIVES_extract_v2.FIVES_extract_v2 import MultiModalDataset
from src.utils.plotting import make_matching_figures
from gen_data_enhance_v2 import apply_domain_randomization, save_batch_visualization

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
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2
        
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

# 数据集根目录硬编码
DATA_ROOT = "/data/student/Fengjunming/LoFTR/data/FIVES_extract_v2"

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
class CurriculumScheduler(pl.Callback):
    """
    Plan v2.2: 温和 Mask 监督机制
    根据 Epoch 动态调整 vessel_soft_lambda 和 loss_weight
    """
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        # 1. 计算当前参数 (v2.2 计划)
        if epoch < 25: # [Stage 1: Teaching] 强监督
            sched_lambda = 2.0
            sched_loss_w = 5.0
            phase = "Teaching"
        elif epoch < 50: # [Stage 2: Weaning] 线性衰减
            progress = (epoch - 25) / (50 - 25)
            sched_lambda = 2.0 * (1.0 - progress)
            sched_loss_w = 5.0 - (4.0 * progress) # 5.0 -> 1.0
            phase = "Weaning"
        else: # [Stage 3: Independence] 自主学习
            sched_lambda = 0.0
            sched_loss_w = 1.0
            phase = "Independence"
            
        # 2. 注入模型
        # 修改 Attention Bias (coarse_matching)
        if hasattr(pl_module.matcher, 'coarse_matching'):
            pl_module.matcher.coarse_matching.vessel_soft_lambda = sched_lambda
            
        # 修改 Loss Weight
        # 直接修改 LoFTRLoss 实例的 c_pos_w (positive sample weight)
        if hasattr(pl_module, 'loss') and hasattr(pl_module.loss, 'c_pos_w'):
            # 默认配置中 pos_weight通常为1.0, 我们将其乘以 sched_loss_w 或直接设为 sched_loss_w
            # 根据 Plan v2.2, loss_weight=5.0 是强惩罚, 且 Independence Phase 回归 1.0.
            # 这意味着 sched_loss_w 就是目标权重值.
            pl_module.loss.c_pos_w = sched_loss_w
            
        # 兼容性: 同时也保留到 module 属性中, 以防其他地方引用
        pl_module.curriculum_loss_weight = sched_loss_w
        
        # 3. 记录日志
        if trainer.global_rank == 0:
            loguru_logger.info(f"Curriculum Scheduler | Epoch {epoch} ({phase}) | λ={sched_lambda:.4f}, Loss_W={sched_loss_w:.4f}")
            
        pl_module.log('sched/lambda', sched_lambda, on_epoch=True, logger=True)
        pl_module.log('sched/loss_w', sched_loss_w, on_epoch=True, logger=True)

# ==========================================
# 核心改动 2: PL_LoFTR_V2 (恢复 Mask 输入)
# ==========================================
class PL_LoFTR_V2(PL_LoFTR):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, use_domain_rand=True, result_dir=None):
        super().__init__(config, pretrained_ckpt, profiler, dump_dir)
        self.use_domain_rand = use_domain_rand
        self.saved_batches = 0
        self.result_dir = result_dir
        # 初始化 curriculum 参数
        self.curriculum_loss_weight = 1.0 
        
    def on_train_epoch_start(self):
        # 覆盖父类 v2.1 的逻辑，不做任何“强制清零”操作
        # 让 CurriculumScheduler 来全权管理
        pass
        
    def training_step(self, batch, batch_idx):
        if self.use_domain_rand and self.training:
            if self.saved_batches < 2:
                img0_orig = batch['image0'].clone()
                img1_orig = batch['image1'].clone()
            
            batch['image0'] = apply_domain_randomization(batch['image0'])
            batch['image1'] = apply_domain_randomization(batch['image1'])
            
            if self.saved_batches < 2:
                if self.result_dir is None:
                    result_dir = Path(f"results/{self.config.DATASET.TRAINVAL_DATA_SOURCE}/{self.config.TRAINER.EXP_NAME}")
                else:
                    result_dir = Path(self.result_dir)
                result_dir.mkdir(parents=True, exist_ok=True)
                
                vessel_mask = batch.get('mask0', None)
                if self.trainer.sanity_checking:
                    epoch_label = 0
                else:
                    epoch_label = self.current_epoch + 1
                
                save_batch_visualization(
                    img0_orig, img1_orig, batch['image0'], batch['image1'],
                    str(result_dir), epoch=epoch_label, step=self.saved_batches + 1, 
                    batch_size=batch['image0'].shape[0], vessel_mask=vessel_mask
                )
                self.saved_batches += 1

        # 【关键改动】：不再移除 mask0/mask1
        # 我们需要保留 mask 供模型使用
        
        # 将 curriculum_loss_weight 注入到 batch 中或其他方式传递给 Loss
        # 这里我们hack一下：假设 Loss 计算逻辑可以使用 batch 中的某个字段
        # 或者我们直接修改 Loss 对象的权重（如果它支持）
        # 查看 LoFTR 的 Loss 计算逻辑，通常 LoFTRLoss 接受 batch
        # 如果要动态加权，最好是传入参数。
        # 暂时方案：我们不改 Loss 代码，而是依赖 vessel_soft_lambda 改变 Attention
        # 至于 loss_weight (vessel vs background)，如果 Loss 类没暴露接口，可能改起来复杂
        # 假如 LoFTRLoss 只是计算 coarse/fine loss，且不区分 vessel/bg 的话，那 sched_loss_w 可能只用于 log?
        # 不，Plan说 loss_bias: "Control how much penalty... for missing vessel matches".
        # 如果现有的 LoFTRLoss 不支持 pixel-wise weighting map，那我们需要修改 LoFTRLoss。
        # 考虑到时间，我们先主要依赖 Attention Bias (Lambda)，这已经是最强的引导了。
        
        return super().training_step(batch, batch_idx)

        # --- 新增：提取所有候选匹配点 (用于判断是取点有问题还是匹配有问题) ---
        all_candidates = []
        if 'conf_matrix' in batch:
            conf_matrix = batch['conf_matrix'] # [N, L, S]
            thr = self.matcher.coarse_matching.thr
            hw0c = batch['hw0_c']
            hw1c = batch['hw1_c']
            # 注意：hw0_i 可能在 batch 里是 [H, W]
            scale0 = batch['image0'].shape[2] / hw0c[0]
            scale1 = batch['image1'].shape[2] / hw1c[0]
            
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
        self.best_val = float('inf')
        self.result_dir = Path(f"results/{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []
        self.epoch_maces = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_maces = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            sub_dir = "step0"
        else:
            sub_dir = f"epoch{trainer.current_epoch + 1}"
        
        if self.args.val_on_real:
            if dataloader_idx == 0:
                epoch_dir = self.result_dir / f"{sub_dir}_generated"
            else:
                epoch_dir = self.result_dir / f"{sub_dir}_real"
        else:
            epoch_dir = self.result_dir / sub_dir
            
        epoch_dir.mkdir(parents=True, exist_ok=True)
        batch_mses, batch_maces = self._save_batch_results(trainer, pl_module, batch, outputs, epoch_dir)
        
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
        if trainer.sanity_checking: return
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        avg_mace = sum(self.epoch_maces) / len(self.epoch_maces) if self.epoch_maces else float('inf')
        
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
            
        # 更新最优模型（改为使用 MACE 监控）
        metric_monitor = avg_mace
        if metric_monitor < self.best_val:
            self.best_val = metric_monitor
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\nBest MACE: {avg_mace:.6f}\nBest MSE: {avg_mse:.6f}")
            loguru_logger.info(f"发现新的最优模型! Epoch {epoch}, MACE: {avg_mace:.4f}")

    def _save_batch_results(self, trainer, pl_module, batch, outputs, epoch_dir):
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
            sample_name = f"{Path(pair_names0[i]).stem}_vs_{Path(pair_names1[i]).stem}"
            save_path = epoch_dir / sample_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            ref_key = 'image1_gt' if 'image1_gt' in batch else 'image1_origin'
            img1_gt = (batch[ref_key][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            H_est = H_ests[i]
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
    parser = argparse.ArgumentParser(description="LoFTR V2.2 Gentle Mask Supervision")
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'])
    # 默认名改为 v2.2
    parser.add_argument('--name', '-n', type=str, default='loftr_multimodal_v2_2_gentle', help='训练名称') 
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--vessel_sigma', type=float, default=6.0)
    parser.add_argument('--pretrained_ckpt', type=str, default='weights/outdoor_ds.ckpt')
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
    
    model = PL_LoFTR_V2(
        config, 
        pretrained_ckpt=args.pretrained_ckpt, 
        use_domain_rand=args.use_domain_randomization,
        result_dir=str(result_dir)
    )
    
    data_module = MultimodalDataModule(args, config)
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.name)
    val_callback = MultimodalValidationCallback(args)
    curriculum_callback = CurriculumScheduler() # 新增
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = DelayedEarlyStopping(start_epoch=100, monitor='val_mace', patience=5, min_delta=0.01)
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        min_epochs=20,
        check_val_every_n_epoch=5,
        num_sanity_val_steps=3,
        callbacks=[curriculum_callback, val_callback, lr_monitor, early_stop_callback],
        logger=tb_logger,
        plugins=DDPPlugin(find_unused_parameters=False) if _n_gpus > 1 else None,
        replace_sampler_ddp=True
    )
    
    loguru_logger.info(f"开始训练 V2.2 (Gentle Supervision): {args.name}")
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
