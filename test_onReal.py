import sys
import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import torch.utils.data as data

from src.config.default import get_cfg_defaults
from src.lightning.lightning_loftr import PL_LoFTR
from src.utils.plotting import make_matching_figures
from measurement import calculate_metrics

# 导入数据集类
from data.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# 数据集根目录 (保持与 train_multimodal_onReal.py 一致)
DATA_ROOTS = {
    'cffa': "/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cffa",
    'cfoct': "/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cfoct",
    'octfa': "/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_octfa",
    'cfocta': "/data/student/Fengjunming/LoFTR/data/CF_OCTA_v2_repaired"
}

class LoFTRTestDatasetWrapper(data.Dataset):
    """
    通过 Dataset 提供的 get_raw_sample 接口获取原始数据。
    测试时直接使用未对齐的原始图像（fix 和 moving），不施加额外变换。
    """
    def __init__(self, dataset, img_size=512):
        self.dataset = dataset
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 调用统一接口获取原始图像和关键点
        # img0_raw: fix 图（已归一化到 [0,1]）
        # img1_raw: moving 图（未对齐，已归一化到 [0,1]）
        # pts0, pts1: 人工标注的关键点对（用于计算 GT）
        img0_raw, img1_raw, pts0, pts1, path0, path1 = self.dataset.get_raw_sample(idx)
        
        # 如果图像是 uint8 格式，归一化到 [0, 1]
        if img0_raw.dtype == np.uint8:
            img0_raw = img0_raw.astype(np.float32) / 255.0
        if img1_raw.dtype == np.uint8:
            img1_raw = img1_raw.astype(np.float32) / 255.0
        
        h0, w0 = img0_raw.shape
        h1, w1 = img1_raw.shape

        # Resize 图像到模型输入尺寸（保持 [0, 1] 范围）
        img0 = cv2.resize(img0_raw, (self.img_size, self.img_size))
        img1 = cv2.resize(img1_raw, (self.img_size, self.img_size))
        
        # 同步缩放关键点坐标
        pts0_res = pts0.copy()
        pts1_res = pts1.copy()
        if len(pts0_res) > 0:
            pts0_res[:, 0] *= (self.img_size / w0)
            pts0_res[:, 1] *= (self.img_size / h0)
        if len(pts1_res) > 0:
            pts1_res[:, 0] *= (self.img_size / w1)
            pts1_res[:, 1] *= (self.img_size / h1)

        return {
            'image0': torch.from_numpy(img0).float()[None],  # [1, H, W], [0, 1]
            'image1': torch.from_numpy(img1).float()[None],  # [1, H, W], [0, 1]
            'ctrl_pts0': pts0_res,
            'ctrl_pts1': pts1_res,
            'pair_names': (os.path.basename(path0), os.path.basename(path1)),
            'dataset_name': 'MultiModal'
        }

def filter_valid_area(img1, img2):
    """完全对齐 train_multimodal_onGen.py 的筛选逻辑"""
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
    mask1 = img1 > 0
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

def create_chessboard(img1, img2, grid_size=4):
    """
    创建棋盘图，将两张图交替拼接成4x4的棋盘
    """
    H, W = img1.shape
    assert img2.shape == (H, W), "Two images must have the same size"
    cell_h = H // grid_size
    cell_w = W // grid_size
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * cell_h
            y_end = (i + 1) * cell_h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    return chessboard

def draw_matches(img0, img1, mkpts0, mkpts1):
    """
    手动绘制匹配连线图
    """
    h0, w0 = img0.shape
    h1, w1 = img1.shape
    canvas = np.zeros((max(h0, h1), w0 + w1), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0+w1] = img1
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for p0, p1 in zip(mkpts0, mkpts1):
        pt0 = (int(p0[0]), int(p0[1]))
        pt1 = (int(p1[0] + w0), int(p1[1]))
        cv2.line(canvas, pt0, pt1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0, 2, (0, 0, 255), -1)
        cv2.circle(canvas, pt1, 2, (0, 0, 255), -1)
    return canvas

def main():
    parser = argparse.ArgumentParser(description="LoFTR 多模态测试脚本")
    parser.add_argument('--mode', type=str, required=True, choices=['cffa', 'cfoct', 'octfa', 'cfocta'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--savedir', type=str, default='test_run')
    args = parser.parse_args()

    # 1. 路径准备
    ckpt_path = Path(f"results/{args.mode}/{args.name}/best_checkpoint/model.ckpt")
    if not ckpt_path.exists():
        logger.error(f"未找到模型权重: {ckpt_path}")
        return

    save_root = Path(f"test_results/{args.mode}/{args.savedir}")
    save_root.mkdir(parents=True, exist_ok=True)
    
    log_file = save_root / "log.txt"
    logger.add(log_file, rotation="10 MB")
    logger.info(f"开始测试模式: {args.mode}, 名称: {args.name}")

    # 2. 加载模型
    config = get_cfg_defaults()
    model = PL_LoFTR.load_from_checkpoint(str(ckpt_path), config=config)
    model.cuda().eval()

    # 3. 准备数据集
    mode = args.mode
    root = DATA_ROOTS[mode]
    if mode == 'cffa':
        base_ds = CFFADataset(root_dir=root, split='test', mode='cf2fa')
    elif mode == 'cfoct':
        base_ds = CFOCTDataset(root_dir=root, split='test', mode='cf2oct')
    elif mode == 'octfa':
        base_ds = OCTFADataset(root_dir=root, split='test', mode='oct2fa')
    elif mode == 'cfocta':
        base_ds = CFOCTADataset(root_dir=root, split='test', mode='cf2octa')
    
    test_ds = LoFTRTestDatasetWrapper(base_ds)
    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # 4. 测试循环
    all_metrics = []
    
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        pair_name = f"{Path(batch['pair_names'][0][0]).stem}_vs_{Path(batch['pair_names'][1][0]).stem}"
        sample_dir = save_root / pair_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 提取控制点并预计算 H_gt (作为 T_0to1 喂给指标函数)
        ctrl_pts0 = batch['ctrl_pts0'][0].numpy()
        ctrl_pts1 = batch['ctrl_pts1'][0].numpy()
        H_gt_mat = np.eye(3, dtype=np.float32)
        if len(ctrl_pts0) >= 4 and len(ctrl_pts1) >= 4:
            H_gt_found, _ = cv2.findHomography(ctrl_pts0, ctrl_pts1, cv2.RANSAC, 5.0)
            if H_gt_found is not None:
                H_gt_mat = H_gt_found.astype(np.float32)
        
        # 将 H_gt 塞入 batch 伪装成 T_0to1，确保 compute_homography_errors 不报错且能算出正确误差
        batch_gpu['T_0to1'] = torch.from_numpy(H_gt_mat).unsqueeze(0).to(batch_gpu['image0'].device)
        batch_gpu['dataset_name'] = ['MultiModal']

        with torch.no_grad():
            model.matcher(batch_gpu)
            
            # 【关键修正】完全对齐验证流程，调用源码函数获取 H_est
            from src.utils.metrics import compute_homography_errors
            compute_homography_errors(batch_gpu, model.config)
            
        # 提取 H_est (由 compute_homography_errors 按照 model.config.TRAINER.RANSAC_PIXEL_THR 算出)
        H_est = batch_gpu['H_est'][0] 
        mkpts0 = batch_gpu['mkpts0_f'].cpu().numpy()
        mkpts1 = batch_gpu['mkpts1_f'].cpu().numpy()

        # 提取图像
        img0 = (batch['image0'][0, 0].numpy() * 255).astype(np.uint8)
        img1 = (batch['image1'][0, 0].numpy() * 255).astype(np.uint8)

        h, w = img0.shape
        # 此时 H_gt 已经计算过了
        H_gt = H_gt_mat
        
        # 1. moving_result: 用模型预测的 H_est 把 img1 变换到 img0 空间
        if H_est is not None and not np.allclose(H_est, np.eye(3)):
            try:
                H_est_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_est_inv, (w, h))
            except:
                img1_result = np.zeros_like(img0)
        else:
            img1_result = np.zeros_like(img0)

        # 2. moving_origin: 用 GT 关键点把 img1 变换到 img0 空间
        if H_gt is not None:
            try:
                H_gt_inv = np.linalg.inv(H_gt)
                img1_origin = cv2.warpPerspective(img1, H_gt_inv, (w, h))
            except:
                img1_origin = np.zeros_like(img0)
        else:
            img1_origin = np.zeros_like(img0)

        # 3. 为内置绘图补齐字段
        if 'm_bids' not in batch_gpu:
            batch_gpu['m_bids'] = torch.zeros(len(mkpts0), dtype=torch.long).to(batch_gpu['image0'].device)
        if 'conf_matrix_gt' not in batch_gpu:
            batch_gpu['conf_matrix_gt'] = torch.zeros(1, 1, 1).to(batch_gpu['image0'].device)

        # 计算指标
        metrics = calculate_metrics(
            img_origin=img0, img_result=img1_result,
            mkpts0=mkpts0, mkpts1=mkpts1,
            ctrl_pts0=ctrl_pts0, ctrl_pts1=ctrl_pts1,
            H_gt=H_gt
        )
        all_metrics.append(metrics)

        # 保存结果
        cv2.imwrite(str(sample_dir / "fix.png"), img0)
        cv2.imwrite(str(sample_dir / "moving.png"), img1)
        cv2.imwrite(str(sample_dir / "moving_result.png"), img1_result)
        cv2.imwrite(str(sample_dir / "moving_origin.png"), img1_origin)
        
        chessboard = create_chessboard(img1_result, img0)
        cv2.imwrite(str(sample_dir / "chessboard.png"), chessboard)
        
        # 匹配可视化
        try:
            import matplotlib.pyplot as plt
            figures = make_matching_figures(batch_gpu, model.config, mode='validation')
            if 'validation' in figures:
                fig = figures['validation'][0]
                fig.savefig(str(sample_dir / "matches_loftr.png"), bbox_inches='tight')
                plt.close(fig)
        except: pass
            
        try:
            match_img = draw_matches(img0, img1, mkpts0, mkpts1)
            cv2.imwrite(str(sample_dir / "matches.png"), match_img)
        except Exception as e:
            logger.warning(f"无法保存匹配图: {e}")

        logger.info(f"Sample: {pair_name} | SR_ME: {metrics['SR_ME']} | MeanErr: {metrics['mean_error']:.2f}")

    # 5. 汇总统计
    if all_metrics:
        total_samples = len(all_metrics)
        # 成功的样本定义：mean_error 是有限数值（非 inf/nan）
        success_metrics = [m for m in all_metrics if m.get('mean_error') is not None and np.isfinite(m['mean_error'])]
        num_success = len(success_metrics)
        num_failed = total_samples - num_success
        failure_rate = (num_failed / total_samples) * 100

        logger.info("="*30)
        logger.info(f"测试整体总结 (总样本: {total_samples}, 成功: {num_success}, 失败: {num_failed}):")
        
        if num_success > 0:
            # 只对成功的样本计算各指标的平均值
            avg_metrics = {k: np.mean([m[k] for m in success_metrics if m.get(k) is not None and np.isfinite(m[k])]) 
                          for k in all_metrics[0].keys()}
            for k, v in avg_metrics.items():
                logger.info(f"  Overall {k}: {v:.4f}")
        else:
            logger.warning("  没有成功的匹配样本，无法计算平均误差指标。")

        logger.info(f"  Overall failure_rate: {failure_rate:.2f}%")
        logger.info("="*30)

if __name__ == "__main__":
    main()
