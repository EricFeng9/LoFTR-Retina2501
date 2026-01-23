import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone0 = build_backbone(config)
        self.backbone1 = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        (feat_c0, feat_f0) = self.backbone0(data['image0'])
        (feat_c1, feat_f1) = self.backbone1(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. 基于血管软掩码对 coarse 特征进行加权（高斯扩展，见 plan.md）
        if 'vessel_weight0' in data and 'vessel_weight1' in data:
            # 将软掩码下采样到 coarse 分辨率，使用双线性插值保持平滑
            v_w0 = data['vessel_weight0']
            v_w1 = data['vessel_weight1']
            if v_w0.dim() == 4:
                v_w0 = v_w0.squeeze(1)
                v_w1 = v_w1.squeeze(1)
            if v_w0.shape[1:] != data['hw0_c']:
                v_w0_c = F.interpolate(v_w0.unsqueeze(1), size=data['hw0_c'], mode='bilinear', align_corners=False).squeeze(1)
                v_w1_c = F.interpolate(v_w1.unsqueeze(1), size=data['hw1_c'], mode='bilinear', align_corners=False).squeeze(1)
            else:
                v_w0_c, v_w1_c = v_w0, v_w1
            data.update({'vessel_weight0_c': v_w0_c, 'vessel_weight1_c': v_w1_c})

            # 使用 (alpha * W + beta) 对 coarse 特征做通道共享的缩放
            alpha = self.config['coarse'].get('vessel_alpha', 1.0)
            beta = self.config['coarse'].get('vessel_beta', 0.3)
            # 【方案 B 改进】取消在 Backbone 阶段直接乘以掩码，让模型看到完整图像以保持全局位置编码的正确性
            # feat_c0 = feat_c0 * (alpha * v_w0_c.unsqueeze(1) + beta)
            # feat_c1 = feat_c1 * (alpha * v_w1_c.unsqueeze(1) + beta)

        # 3. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            # 方案B: 同时处理普通掩码和血管掩码的下采样
            mask_c0, mask_c1 = self._process_mask(data['mask0'], data['mask1'], data)
            data.update({'mask0': mask_c0, 'mask1': mask_c1})
            
            if 'vessel_mask0' in data:
                v_mask_c0, v_mask_c1 = self._process_mask(data['vessel_mask0'], data['vessel_mask1'], data)
                data.update({'vessel_mask0': v_mask_c0, 'vessel_mask1': v_mask_c1})
            
            mask_c0, mask_c1 = mask_c0.flatten(-2), mask_c1.flatten(-2)

        # 若存在 coarse 级别的血管软权重，则展平成 [N, L] 供 coarse matching 使用
        if 'vessel_weight0_c' in data and 'vessel_weight1_c' in data:
            v_w0_flat = data['vessel_weight0_c'].flatten(-2)
            v_w1_flat = data['vessel_weight1_c'].flatten(-2)
            data.update({
                'vessel_weight0_c_flat': v_w0_flat,
                'vessel_weight1_c_flat': v_w1_flat
            })
            
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('matcher.'):
                k = k.replace('matcher.', '', 1)
            
            # 处理单流 Backbone 到双流 Backbone 的权重复制
            if k.startswith('backbone.'):
                # 复制给 backbone0
                new_state_dict[k.replace('backbone.', 'backbone0.', 1)] = v
                # 复制给 backbone1
                new_state_dict[k.replace('backbone.', 'backbone1.', 1)] = v
            else:
                new_state_dict[k] = v
        
        return super().load_state_dict(new_state_dict, *args, **kwargs)

    def _process_mask(self, mask0, mask1, data):
        """ 辅助函数：处理掩码的维度并进行下采样 """
        if mask0.dim() == 4:  # (N, 1, H, W)
            mask0 = mask0.squeeze(1)
            mask1 = mask1.squeeze(1)
        
        if mask0.shape[1:] != data['hw0_c']:
            mask0 = F.interpolate(mask0.unsqueeze(1).float(), size=data['hw0_c'], mode='nearest').squeeze(1).bool()
            mask1 = F.interpolate(mask1.unsqueeze(1).float(), size=data['hw1_c'], mode='nearest').squeeze(1).bool()
        return mask0, mask1
