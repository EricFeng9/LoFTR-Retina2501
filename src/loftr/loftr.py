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
        self.backbone = build_backbone(config)
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

        (feat_c0, feat_f0) = self.backbone(data['image0'])
        (feat_c1, feat_f1) = self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 3. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            # 方案B: 同时处理普通掩码和血管掩码的下采样
            mask_c0, mask_c1 = self._process_mask(data['mask0'], data['mask1'], data)
            data.update({'mask0': mask_c0, 'mask1': mask_c1})
            
            mask_c0, mask_c1 = mask_c0.flatten(-2), mask_c1.flatten(-2)
        
        # [V2.3 Clean Up] We removed all vessel mask injection logic.
        # forward() is now purely vision-based.
            
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
