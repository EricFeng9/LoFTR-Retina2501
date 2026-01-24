"""
基于v3_2, 添加以下改进:
1. 域随机化增强 (Domain Randomization) - 增强版
   - Intensity Bias Field (空间渐变亮度不均, 模拟光照不均匀)
   - Speckle Noise (斑点噪声, OCT/超声成像常见)
   - Poisson Noise (泊松噪声, 低光条件)
   - Random Downsampling (3%概率, 模拟低分辨率扫描)
2. InfoNCE Loss 替代 Triplet Loss - 使用 batch 内所有点作为负样本, 防止描述子坍塌

参考论文: "Synthetic data in generalizable, learning-based neuroimaging"
对应模型文件: model/super_retina_multimodal.py (descriptor_loss_warmup 使用 InfoNCE)
"""

import torch
import os
import sys
import yaml
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import random

# 添加本地模块路径
sys.path.append(os.getcwd())

# 使用新数据集脚本
from dataset.FIVES_extract_v2.FIVES_extract_v2 import MultiModalDataset
from dataset.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset
from model.super_retina_multimodal import SuperRetinaMultimodal
from common.train_util import value_map_load, value_map_save, affine_images
from common.common_util import nms, sample_keypoint_desc
from common.vessel_keypoint_extractor import extract_vessel_keypoints, extract_vessel_keypoints_fallback
from torchvision import transforms


# ============================================================================
# 域随机化增强 (Domain Randomization) - 增强版
# 目的: 打破生成数据中 CF 和 FA/OCT 之间的纹理相关性
#       迫使模型学习真正的跨模态不变性特征 (几何结构) 而非表面纹理
# 
# 参考: "Synthetic data in generalizable, learning-based neuroimaging"
# 增加: 1. Intensity Bias Field (空间渐变亮度不均)
#       2. Speckle/Poisson Noise (斑点/泊松噪声, 模拟真实成像)
#       3. Random Downsampling (小概率随机降采样, 模拟低质量扫描)
# ============================================================================

def generate_intensity_bias_field(H, W, device, num_control_points=4, strength_range=(0.7, 1.3)):
    """
    生成随机 Intensity Bias Field (空间渐变亮度场)
    模拟真实眼底成像中的光照不均匀性 (如边缘比中心暗)
    
    Args:
        H, W: 图像尺寸
        device: torch 设备
        num_control_points: 控制点数量 (越少越平滑)
        strength_range: 亮度乘数范围
    
    Returns:
        bias_field: [1, 1, H, W] 亮度乘法场
    """
    # 生成低分辨率随机场
    low_res = torch.rand(1, 1, num_control_points, num_control_points, device=device)
    # 缩放到 strength_range
    low_res = low_res * (strength_range[1] - strength_range[0]) + strength_range[0]
    # 上采样到原始分辨率 (双三次插值产生平滑渐变)
    bias_field = F.interpolate(low_res, size=(H, W), mode='bicubic', align_corners=False)
    return bias_field


def apply_speckle_noise(img, intensity_range=(0.0, 0.15)):
    """
    添加斑点噪声 (Speckle Noise)
    常见于 OCT、超声等相干成像系统
    
    公式: out = img + img * noise
    即噪声强度与局部亮度成正比
    """
    intensity = random.uniform(*intensity_range)
    if intensity > 0:
        noise = torch.randn_like(img) * intensity
        img = img + img * noise
    return img


def apply_poisson_noise(img, scale_range=(50, 200)):
    """
    添加泊松噪声 (Shot Noise)
    模拟低光条件下光子计数噪声
    
    Args:
        scale_range: 缩放因子范围, 越小噪声越强
    """
    scale = random.uniform(*scale_range)
    # 转换到 numpy 处理
    img_np = img.cpu().numpy()
    # 缩放到 [0, scale] 范围
    img_scaled = img_np * scale
    # 应用泊松噪声
    noisy = np.random.poisson(img_scaled.clip(0, None)).astype(np.float32)
    # 缩放回 [0, 1]
    noisy = noisy / scale
    return torch.from_numpy(noisy).to(img.device)


def apply_random_downsample(img, factor_range=(2, 4)):
    """
    随机降采样后再上采样回原尺寸
    模拟低分辨率或老旧设备产生的模糊图像
    """
    _, _, H, W = img.shape
    factor = random.randint(*factor_range)
    # 降采样
    low_res = F.interpolate(img, size=(H // factor, W // factor), mode='bilinear', align_corners=False)
    # 上采样回原尺寸
    restored = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=False)
    return restored


def apply_domain_randomization(img_tensor, gamma_range=(0.7, 1.5), 
                                contrast_range=(0.7, 1.3),
                                brightness_range=(-0.15, 0.15),
                                noise_std_range=(0.0, 0.05),
                                blur_prob=0.3, blur_kernel_range=(3, 7),
                                bias_field_prob=0.5,
                                speckle_prob=0.3,
                                poisson_prob=0.2,
                                downsample_prob=0.03):
    """
    对输入图像张量应用域随机化增强 (增强版)
    
    Args:
        img_tensor: [B, 1, H, W] 形状的图像张量，值域 [0, 1]
        gamma_range: Gamma 变换范围
        contrast_range: 对比度调整范围
        brightness_range: 亮度偏移范围
        noise_std_range: 高斯噪声标准差范围
        blur_prob: 应用模糊的概率
        blur_kernel_range: 模糊核大小范围 (奇数)
        bias_field_prob: 应用 Intensity Bias Field 的概率
        speckle_prob: 应用斑点噪声的概率
        poisson_prob: 应用泊松噪声的概率
        downsample_prob: 应用随机降采样的概率 (默认3%)
    
    Returns:
        增强后的图像张量 [B, 1, H, W]
    """
    B, _, H, W = img_tensor.shape
    device = img_tensor.device
    augmented = img_tensor.clone()
    
    for b in range(B):
        img = augmented[b:b+1]  # [1, 1, H, W]
        
        # 0. [新增] Intensity Bias Field (空间渐变亮度不均)
        if random.random() < bias_field_prob:
            bias_field = generate_intensity_bias_field(H, W, device)
            img = img * bias_field
        
        # 1. Gamma 变换 (模拟不同曝光条件)
        gamma = random.uniform(*gamma_range)
        img = torch.pow(img.clamp(min=1e-8), gamma)
        
        # 2. 对比度调整 (围绕均值缩放)
        contrast = random.uniform(*contrast_range)
        mean_val = img.mean()
        img = (img - mean_val) * contrast + mean_val
        
        # 3. 亮度偏移
        brightness = random.uniform(*brightness_range)
        img = img + brightness
        
        # 4. 高斯噪声
        noise_std = random.uniform(*noise_std_range)
        if noise_std > 0:
            noise = torch.randn_like(img) * noise_std
            img = img + noise
        
        # 5. [新增] 斑点噪声 (Speckle Noise)
        if random.random() < speckle_prob:
            img = apply_speckle_noise(img)
        
        # 6. [新增] 泊松噪声 (Shot Noise)
        if random.random() < poisson_prob:
            img = apply_poisson_noise(img)
        
        # 7. 高斯模糊 (随机应用)
        if random.random() < blur_prob:
            kernel_size = random.choice(range(blur_kernel_range[0], blur_kernel_range[1] + 1, 2))  # 只选奇数
            # 使用 OpenCV 进行模糊 (更高效)
            img_np = img[0, 0].cpu().numpy()
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
            img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        
        # 8. [新增] 随机降采样+上采样 (模拟低分辨率扫描, 小概率)
        if random.random() < downsample_prob:
            img = apply_random_downsample(img)
        
        # 裁剪到有效范围
        augmented[b:b+1] = img.clamp(0, 1)
    
    return augmented


def save_batch_visualization(img0_orig, img1_orig, img0_aug, img1_aug, 
                              vessel_mask, save_dir, epoch, step, batch_size):
    """
    保存一个 batch 的可视化结果,用于检查域随机化效果
    
    Args:
        img0_orig: 原始 fix 图像 [B, 1, H, W]
        img1_orig: 原始 moving 图像 [B, 1, H, W]
        img0_aug: 增强后 fix 图像 [B, 1, H, W]
        img1_aug: 增强后 moving 图像 [B, 1, H, W]
        vessel_mask: 血管掩码 [B, 1, H, W]
        save_dir: 保存目录
        epoch: 当前 epoch
        step: 当前 step (batch index)
        batch_size: batch 大小
    """
    vis_dir = os.path.join(save_dir, f'epoch{epoch}_visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    for b in range(min(batch_size, img0_orig.shape[0])):
        sample_dir = os.path.join(vis_dir, f'step{step}_sample{b}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 转换为 numpy 并保存
        fix_orig = (img0_orig[b, 0].cpu().numpy() * 255).astype(np.uint8)
        mov_orig = (img1_orig[b, 0].cpu().numpy() * 255).astype(np.uint8)
        fix_aug = (img0_aug[b, 0].cpu().numpy() * 255).astype(np.uint8)
        mov_aug = (img1_aug[b, 0].cpu().numpy() * 255).astype(np.uint8)
        vessel = (vessel_mask[b, 0].cpu().numpy() * 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(sample_dir, 'fix_original.png'), fix_orig)
        cv2.imwrite(os.path.join(sample_dir, 'moving_original.png'), mov_orig)
        cv2.imwrite(os.path.join(sample_dir, 'fix_augmented.png'), fix_aug)
        cv2.imwrite(os.path.join(sample_dir, 'moving_augmented.png'), mov_aug)
        cv2.imwrite(os.path.join(sample_dir, 'vessel_mask.png'), vessel)
        
        # 创建对比图: 左边原始,右边增强
        comparison_fix = np.hstack([fix_orig, fix_aug])
        comparison_mov = np.hstack([mov_orig, mov_aug])
        cv2.imwrite(os.path.join(sample_dir, 'comparison_fix_orig_vs_aug.png'), comparison_fix)
        cv2.imwrite(os.path.join(sample_dir, 'comparison_moving_orig_vs_aug.png'), comparison_mov)


def draw_matches(img1, kps1, img2, kps2, matches, save_path):
    """
    在两张图像之间绘制匹配连线
    """
    if torch.is_tensor(img1): img1 = (img1.cpu().numpy() * 255).astype(np.uint8)
    if torch.is_tensor(img2): img2 = (img2.cpu().numpy() * 255).astype(np.uint8)
    
    if img1.ndim == 3: img1 = img1.squeeze()
    if img2.ndim == 3: img2 = img2.squeeze()
    
    kp1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps1]
    kp2_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps2]
    
    out_img = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, matches, None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imwrite(save_path, out_img)

def validate(model, val_dataset, device, epoch, save_dir, log_file, train_config, mode):
    """
    验证函数:评估模型在真实数据集上的表现
    使用与 test_on_real.py 相同的评估流程
    """
    from measurement import calculate_metrics
    
    model.eval()
    all_metrics = []
    
    # 从配置中统一读取阈值
    nms_thresh = train_config.get('nms_thresh', 0.01)
    content_thresh = train_config.get('content_thresh', 0.7) # Lowe's Ratio
    geometric_thresh = train_config.get('geometric_thresh', 0.7) # RANSAC re-projection error
    
    epoch_save_dir = os.path.join(save_dir, f'epoch{epoch}')
    os.makedirs(epoch_save_dir, exist_ok=True)
    
    log_f = open(log_file, 'a')
    log_f.write(f'\n--- Validation Epoch {epoch} ---\n')
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc=f"Val Epoch {epoch}"):
            raw_data = val_dataset.get_raw_sample(i)
            
            # 根据不同模态解包数据
            if mode == 'cfocta': # (cf, octa, pts_cf, pts_octa, path_cf, path_octa)
                img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
            elif mode == 'cffa': # (fa, cf, pts_fa, pts_cf, path_fa, path_cf)
                img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
            elif mode == 'cfoct': # (cf, oct, pts_cf, pts_oct, path_cf, path_oct)
                img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
            elif mode == 'octfa': # (fa, oct, pts_fa, pts_oct, path_fa, path_oct)
                img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
            
            # 确保灰度图用于模型输入
            if img_fix_raw.ndim == 3:
                img_fix_gray = cv2.cvtColor(img_fix_raw, cv2.COLOR_RGB2GRAY) if img_fix_raw.shape[2] == 3 else img_fix_raw.squeeze()
            else:
                img_fix_gray = img_fix_raw

            if img_mov_raw.ndim == 3:
                img_mov_gray = cv2.cvtColor(img_mov_raw, cv2.COLOR_RGB2GRAY) if img_mov_raw.shape[2] == 3 else img_mov_raw.squeeze()
            else:
                img_mov_gray = img_mov_raw
            
            sample_id = os.path.basename(path_fix).split('.')[0]
            
            # 准备模型输入
            img0_tensor = transform(img_fix_gray).unsqueeze(0).to(device)
            img1_tensor = transform(img_mov_gray).unsqueeze(0).to(device)
            
            # 提取跨模态特征
            det_fix, desc_fix = model.network(img0_tensor)
            det_mov, desc_mov = model.network(img1_tensor)
            
            # 有效区域屏蔽,防止边缘伪影干扰关键点提取
            valid_mask = (img1_tensor > 0.05).float()
            valid_mask = -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)
            det_mov_masked = det_mov * valid_mask
            
            # 提取关键点
            kps_fix = nms(det_fix, nms_thresh=nms_thresh, nms_size=5)[0]
            kps_mov = nms(det_mov_masked, nms_thresh=nms_thresh, nms_size=5)[0]
            
            # 兜底策略:如果关键点太少,强制取响应最高的前100个点
            if len(kps_fix) < 10:
                flat_det = det_fix[0, 0].view(-1)
                _, idx = torch.topk(flat_det, min(100, flat_det.numel()))
                y = idx // det_fix.shape[3]; x = idx % det_fix.shape[3]
                kps_fix = torch.stack([x, y], dim=1).float()

            if len(kps_mov) < 10:
                flat_det = det_mov_masked[0, 0].view(-1)
                if flat_det.max() > 0:
                    _, idx = torch.topk(flat_det, min(100, flat_det.numel()))
                    y = idx // det_mov.shape[3]; x = idx % det_mov.shape[3]
                    kps_mov = torch.stack([x, y], dim=1).float()
            
            good_matches = []
            if len(kps_fix) >= 4 and len(kps_mov) >= 4:
                # 采样描述子并进行特征匹配
                desc_fix_samp = sample_keypoint_desc(kps_fix[None], desc_fix, s=8)[0]
                desc_mov_samp = sample_keypoint_desc(kps_mov[None], desc_mov, s=8)[0]
                
                d1 = desc_fix_samp.permute(1, 0).cpu().numpy()
                d2 = desc_mov_samp.permute(1, 0).cpu().numpy()
                
                matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
                
                # 使用配置中的 content_thresh 作为 Ratio Test 阈值
                for m, n in matches:
                    if m.distance < content_thresh * n.distance:
                        good_matches.append(m)
            
            # 映射回原始尺度
            h_f, w_f = img_fix_raw.shape[:2]
            h_m, w_m = img_mov_raw.shape[:2]
            
            kps_f_orig = kps_fix.cpu().numpy() * [w_f / 512.0, h_f / 512.0]
            kps_m_orig = kps_mov.cpu().numpy() * [w_m / 512.0, h_m / 512.0]
            
            mkpts0 = np.array([kps_f_orig[m.queryIdx] for m in good_matches]) if good_matches else np.array([])
            mkpts1 = np.array([kps_m_orig[m.trainIdx] for m in good_matches]) if good_matches else np.array([])
            
            # 为了统一评估，将 moving 图像 resize 到和 fix 相同尺寸
            # 这样匹配点和控制点都在同一尺寸空间中
            if (h_m, w_m) != (h_f, w_f):
                # Resize moving 图像到 fix 尺寸
                img_mov_resized = cv2.resize(img_mov_raw, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
                
                # 调整 moving 侧的关键点和匹配点坐标
                scale_x = w_f / w_m
                scale_y = h_f / h_m
                
                # 只对非空数组进行缩放操作
                if len(mkpts1) > 0:
                    mkpts1 = mkpts1 * [scale_x, scale_y]
                if len(kps_m_orig) > 0:
                    kps_m_orig = kps_m_orig * [scale_x, scale_y]
                if len(pts_mov_gt) > 0:
                    pts_mov_gt = pts_mov_gt * [scale_x, scale_y]
            else:
                img_mov_resized = img_mov_raw
            
            # 使用 measurement.py 进行评估
            metrics = calculate_metrics(
                img_origin=img_fix_raw, img_result=img_mov_resized,
                mkpts0=mkpts0, mkpts1=mkpts1,
                kpts0=kps_f_orig, kpts1=kps_m_orig,
                ctrl_pts0=pts_fix_gt, ctrl_pts1=pts_mov_gt
            )
            
            all_metrics.append(metrics)
            
            log_f.write(f"ID: {sample_id} | SR_ME: {metrics['SR_ME']} | SR_MAE: {metrics['SR_MAE']} | "
                       f"Rep: {metrics['Rep']:.4f} | MIR: {metrics['MIR']:.4f} | "
                       f"MeanErr: {metrics['mean_error']:.2f} px\n")
            
            # 保存可视化结果
            sample_save_dir = os.path.join(epoch_save_dir, sample_id)
            os.makedirs(sample_save_dir, exist_ok=True)
            
            # 保存原始图像（moving 使用统一尺寸后的版本）
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fix.png'), img_fix_raw)
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving.png'), img_mov_resized)

            # 如果有足够的匹配点,进行配准
            if len(mkpts0) >= 4:
                H_pred, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, geometric_thresh)
                if H_pred is not None:
                    # 确保图像为灰度图（使用统一尺寸后的 moving 图像）
                    img_m_gray = cv2.cvtColor(img_mov_resized, cv2.COLOR_RGB2GRAY) if img_mov_resized.ndim == 3 else img_mov_resized
                    img_f_gray = cv2.cvtColor(img_fix_raw, cv2.COLOR_RGB2GRAY) if img_fix_raw.ndim == 3 else img_fix_raw
                    
                    # 将 moving 配准到 fix 的尺寸空间
                    reg_img = cv2.warpPerspective(img_m_gray, H_pred, (w_f, h_f))
                    cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving_result.png'), reg_img)
                    
                    # 计算棋盘格可视化
                    def compute_checkerboard(img1, img2, n_grid=4):
                        if img1.ndim == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.shape[2] == 3 else img1.squeeze()
                        if img2.ndim == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.shape[2] == 3 else img2.squeeze()
                        h, w = img1.shape[:2]
                        if img2.shape[:2] != (h, w): img2 = cv2.resize(img2, (w, h))
                        grid_h, grid_w = h // n_grid, w // n_grid
                        checkerboard = np.zeros_like(img1)
                        for i in range(n_grid):
                            for j in range(n_grid):
                                y_s, y_e = i * grid_h, (i + 1) * grid_h if i < n_grid - 1 else h
                                x_s, x_e = j * grid_w, (j + 1) * grid_w if j < n_grid - 1 else w
                                checkerboard[y_s:y_e, x_s:x_e] = img1[y_s:y_e, x_s:x_e] if (i + j) % 2 == 0 else img2[y_s:y_e, x_s:x_e]
                        return checkerboard
                    
                    checker = compute_checkerboard(img_f_gray, reg_img, n_grid=4)
                    cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_checkerboard.png'), checker)
            
            # 绘制匹配关系（使用统一尺寸后的图像）
            draw_matches(img_fix_raw, kps_f_orig, img_mov_resized, kps_m_orig, good_matches, 
                        os.path.join(sample_save_dir, f'{sample_id}_matches.png'))

    # 计算平均指标
    summary = {}
    for key in ['SR_ME', 'SR_MAE', 'Rep', 'MIR', 'mean_error', 'max_error']:
        vals = [m[key] for m in all_metrics if m[key] != float('inf')]
        summary[key] = np.mean(vals) if vals else 0.0
    
    log_f.write(f"\n--- Validation Summary ---\n")
    log_f.write(f"Overall SR_ME (Success Rate @5px):  {summary['SR_ME']*100:.2f}%\n")
    log_f.write(f"Overall SR_MAE (Success Rate @10px): {summary['SR_MAE']*100:.2f}%\n")
    log_f.write(f"Average Repeatability:              {summary['Rep']*100:.2f}%\n")
    log_f.write(f"Average Matching Inliers Ratio:     {summary['MIR']*100:.2f}%\n")
    log_f.write(f"Mean Registration Error:            {summary['mean_error']:.2f} px\n")
    log_f.write(f"Max Registration Error (Average):   {summary['max_error']:.2f} px\n")
    log_f.close()
    
    print(f'Validation Epoch {epoch} Finished. Mean Error: {summary["mean_error"]:.2f} px, SR_ME: {summary["SR_ME"]*100:.2f}%')
    return summary['mean_error']

def train_multimodal():
    """
    多模态训练主流程 - 使用新数据集和三阶段课程学习
    """
    # 加载配置
    config_path = './config/train_multimodal.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Command line args to override config
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, help='Experiment name', default=None)
    parser.add_argument('--mode', '-m', type=str, choices=['cffa', 'cfoct', 'octfa', 'cfocta'], 
                        help='Registration mode', default=None)
    parser.add_argument('--epoch', '-e', type=int, help='Number of training epochs', default=500)
    parser.add_argument('--batch_size', '-b', type=int, help='Batch size for training', default=4)
    parser.add_argument('--geometric_thresh', '-g', type=float, help='RANSAC geometric threshold for PKE', default=0.7)
    parser.add_argument('--content_thresh', '-c', type=float, help='Lowe ratio threshold for feature matching', default=0.8)
    args = parser.parse_args()
    
    if args.name:
        config['MODEL']['name'] = args.name
    if args.mode:
        config['DATASET']['registration_type'] = args.mode
    if args.epoch:
        config['MODEL']['num_epoch'] = args.epoch
    if args.batch_size:
        config['DATASET']['batch_size'] = args.batch_size
    if args.geometric_thresh is not None:
        config['PKE']['geometric_thresh'] = args.geometric_thresh
    if args.content_thresh is not None:
        config['PKE']['content_thresh'] = args.content_thresh
        
    train_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}
    
    exp_name = train_config.get('name', 'default_exp')
    reg_type = train_config['registration_type']
    save_root = f'./save/{reg_type}/{exp_name}'
    os.makedirs(save_root, exist_ok=True)
    
    log_file = os.path.join(save_root, 'validation_log.txt')
    train_log_file = os.path.join(save_root, 'train_log.txt')  # 新增：训练日志
    
    device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
    
    # 打开训练日志文件
    train_log = open(train_log_file, 'a', buffering=1)  # 行缓冲，实时写入
    
    def log_print(msg):
        """同时输出到控制台和日志文件"""
        print(msg)
        train_log.write(msg + '\n')
        train_log.flush()
    
    log_print(f"Using device: {device} | Experiment: {exp_name}")

    # 数据加载 - 使用新的FIVES数据集进行训练
    root_dir = train_config['root_dir']
    batch_size = train_config['batch_size']
    img_size = train_config.get('img_size', 512)
    df = train_config.get('df', 8)
    
    train_set = MultiModalDataset(
        root_dir=root_dir, 
        mode=reg_type, 
        split='train', 
        img_size=img_size, 
        df=df
    )
    
    # 验证集使用真实数据集 (与 test_on_real.py 一致)
    if reg_type == 'cfocta':
        val_set = CFOCTADataset(root_dir='dataset/CF_OCTA_v2_repaired', split='val', mode='cf2octa')
    elif reg_type == 'cffa':
        val_set = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='val', mode='fa2cf')
    elif reg_type == 'cfoct':
        val_set = CFOCTDataset(root_dir='dataset/operation_pre_filtered_cfoct', split='val', mode='cf2oct')
    elif reg_type == 'octfa':
        val_set = OCTFADataset(root_dir='dataset/operation_pre_filtered_octfa', split='val', mode='fa2oct')
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 初始化多模态 SuperRetina 模型
    model = SuperRetinaMultimodal(train_config, device=device)
    
    if train_config['load_pre_trained_model']:
        path = train_config['pretrained_path']
        if os.path.exists(path):
            log_print(f"Loading pretrained model from {path}")
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['net'])
            
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = train_config['num_epoch']
    pke_start_epoch = train_config['pke_start_epoch']
    
    is_value_map_save = train_config['is_value_map_save']
    value_map_save_dir = train_config['value_map_save_dir']
    
    if is_value_map_save:
        if os.path.exists(value_map_save_dir):
            shutil.rmtree(value_map_save_dir)
        os.makedirs(value_map_save_dir)
        
    value_maps_running = {} if not is_value_map_save else None
    best_mean_error = float('inf')
    
    # 早停机制变量 (仅在epoch >= 100后启用)
    patience = 5  # 验证损失连续5次不下降则早停
    patience_counter = 0
    best_val_error = float('inf')

    # 初始验证
    log_print("Running initial validation...")
    _ = validate(model, val_set, device, 0, save_root, log_file, train_config, reg_type)

    pke_start_epoch = train_config.get('pke_start_epoch', 40) # 默认40以后开启PKE
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # Determine Training Phase
        if epoch <= 20:
            phase = 1 # Warmup
            model.PKE_learn = False
            phase_name = "Phase 1: Warmup (Desc + GT Det)"
        elif epoch <= 40:
            phase = 2 # Geo Consistency
            model.PKE_learn = False
            phase_name = "Phase 2: Geo Consistency (Desc + GT Det + Geo Det)"
        else:
            phase = 3 # Joint PKE
            model.PKE_learn = True
            phase_name = "Phase 3: Joint PKE"
            
        log_print(f'Epoch {epoch}/{num_epochs} | {phase_name}')
        model.train()
            
        running_loss_det = 0.0
        running_loss_desc = 0.0
        total_samples = 0
        
        for step_idx, data in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            img0_orig = data['image0'].to(device)
            img1_orig = data['image1'].to(device)
            
            # ===== 域随机化增强: 对 fix 和 moving 分别应用独立的随机扰动 =====
            # 关键: 两边使用完全独立的随机参数,打破它们之间的纹理相关性
            img0 = apply_domain_randomization(img0_orig)
            img1 = apply_domain_randomization(img1_orig)
            
            # ===== 可视化: 保存第一个 epoch 的前两个 batch =====
            if epoch == 1 and step_idx < 2:
                save_batch_visualization(
                    img0_orig, img1_orig, img0, img1,
                    data['vessel_mask0'].to(device),
                    save_root, epoch, step_idx + 1, batch_size
                )
            # ===== 关键修改：从完整血管分割图中提取稀疏的分叉点 =====
            # 这些分叉点将作为训练时的监督信号，引导模型学习独特的关键点
            vessel_mask_full = data['vessel_mask0']  # [B, 1, H, W]
            vessel_keypoints_batch = []
            
            for b in range(vessel_mask_full.shape[0]):
                # 转换为 numpy 格式 (H, W)
                mask_np = (vessel_mask_full[b, 0].cpu().numpy() * 255).astype(np.uint8)
                
                # 提取关键点
                try:
                    keypoints = extract_vessel_keypoints(mask_np, min_distance=8)
                except:
                    try:
                        keypoints = extract_vessel_keypoints_fallback(mask_np, min_distance=8)
                    except:
                        # 如果提取失败，使用原始掩码（但这会导致之前的问题）
                        keypoints = (mask_np > 127).astype(np.float32)
                        print("点位提取失败")
                vessel_keypoints_batch.append(torch.from_numpy(keypoints).float())

            # 转换回 tensor [B, 1, H, W]
            vessel_keypoints = torch.stack(vessel_keypoints_batch, dim=0).unsqueeze(1).to(device)
            
            
            # 准备 PKE 训练所需参数
            batch_size = img0.size(0)
            input_with_labels = torch.ones(batch_size, dtype=torch.bool).to(device)
            learn_index = torch.where(input_with_labels)
            
            # 加载动态 Value Maps (记录每个像素点的历史置信度)
            names = data['pair_names'][0] # 使用固定图名称作为 key
            value_maps = value_map_load(value_map_save_dir, names, input_with_labels, 
                                      img0.shape[-2:], value_maps_running)
            value_maps = value_maps.to(device)
            
            # 读取真值几何变换矩阵 H_0to1 (image0 -> image1)，用于描述子热身阶段
            H_0to1 = data.get('T_0to1', None)
            if H_0to1 is not None:
                H_0to1 = H_0to1.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                # 调用模型 forward 方法，传入关键点掩码（而不是完整血管掩码）作为初始标签
                # 同时传入完整血管掩码 vessel_mask_full 用于 PKE 候选点过滤
                loss, number_pts, loss_det_item, loss_desc_item, enhanced_kp, enhanced_label, det_pred, n_det, n_desc = \
                    model(img0, img1, vessel_keypoints, value_maps, learn_index,
                          phase=phase, vessel_mask=vessel_mask_full, H_0to1=H_0to1)
                    
                loss.backward()
                optimizer.step()
                
            # 更新持久化的 Value Maps
            if len(learn_index[0]) > 0:
                value_maps = value_maps.cpu()
                value_map_save(value_map_save_dir, names, input_with_labels, value_maps, value_maps_running)
                    
            running_loss_det += loss_det_item
            running_loss_desc += loss_desc_item
            total_samples += img0.size(0)

        epoch_loss = (running_loss_det + running_loss_desc) / total_samples
        log_print(f'Train Total Loss: {epoch_loss:.4f} (Det: {running_loss_det/total_samples:.4f}, Desc: {running_loss_desc/total_samples:.4f})')
        
        # 每 5 个 Epoch 进行一次验证并保存模型
        if epoch % 5 == 0:
            mean_error = validate(model, val_set, device, epoch, save_root, log_file, train_config, reg_type)
            
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'mean_error': mean_error
            }
            
            # 保存最新模型
            latest_dir = os.path.join(save_root, 'latestpoint')
            os.makedirs(latest_dir, exist_ok=True)
            torch.save(state, os.path.join(latest_dir, 'checkpoint.pth'))
            # 保存epoch信息
            with open(os.path.join(latest_dir, 'checkpoint_info.txt'), 'w') as f:
                f.write(f'Latest Checkpoint\nEpoch: {epoch}\nMean Error: {mean_error:.4f} px\n')
            
            # 保存 Mean Error 表现最好的模型 (越小越好)
            if mean_error < best_mean_error:
                log_print(f"New Best Mean Error: {mean_error:.4f} px (Previous: {best_mean_error:.4f} px)")
                best_mean_error = mean_error
                best_dir = os.path.join(save_root, 'bestcheckpoint')
                os.makedirs(best_dir, exist_ok=True)
                torch.save(state, os.path.join(best_dir, 'checkpoint.pth'))
                # 保存epoch信息
                with open(os.path.join(best_dir, 'checkpoint_info.txt'), 'w') as f:
                    f.write(f'Best Checkpoint\nEpoch: {epoch}\nMean Error: {mean_error:.4f} px\n')
            
            # 早停机制 (仅在 epoch >= 100 后启用)
            if epoch >= 100:
                if mean_error < best_val_error:
                    best_val_error = mean_error
                    patience_counter = 0
                    log_print(f'[Early Stopping] Validation Mean Error improved to {best_val_error:.4f} px. Reset patience counter.')
                else:
                    patience_counter += 1
                    log_print(f'[Early Stopping] Validation Mean Error did not improve. Patience: {patience_counter}/{patience}')
                
                if patience_counter >= patience:
                    log_print(f'Early stopping triggered at epoch {epoch}. Best validation Mean Error: {best_val_error:.4f} px')
                    break
    
    # 训练结束，关闭日志文件
    train_log.close()

if __name__ == '__main__':
    train_multimodal()
