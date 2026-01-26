from loguru import logger

import torch
import torch.nn as nn


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['loftr']['loss']
        self.match_type = self.config['loftr']['match_coarse']['match_type']
        self.sparse_spvs = self.config['loftr']['match_coarse']['sparse_spvs']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']
        
        # 【方案 B 改进】梯度保底权重 (默认 0.1)
        self.background_weight = 0.1
        
        # [V2.3] Spatial Loss Weight Scale (Updated by Schedule)
        self.vessel_loss_weight_scaler = 1.0

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ 粗级匹配损失 (对应 plan.md 中的 Lc)
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1) 预测的置信度矩阵
            conf_gt (torch.Tensor): (N, HW0, HW1) 真值匹配矩阵 (0 或 1)
            weight (torch.Tensor): (N, HW0, HW1) 样本权重（用于处理 padding）
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # 处理没有正样本的边缘情况
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Cross-entropy 不支持稀疏监督'
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            # 计算负对数似然损失
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                
                loss =  c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))
        
    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        精级细化损失 (对应 plan.md 中的 Lf)
        使用方差的倒数作为权重，方差越大（不确定性越高）的点权重越低。
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask 标识哪些点需要计算精级损失
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # 使用方差的倒数作为权重 (plan.md: 1/sigma^2)
        std = expec_f[:, 2]
        inverse_var = 1. / torch.clamp(std**2, min=1e-10) # sigma^2
        weight = (inverse_var / torch.mean(inverse_var)).detach()  # 归一化权重以防止通过增大方差来减小损失

        # 处理没有正确匹配的情况
        if not correct_mask.any():
            if self.training:
                logger.warning("没有发现正确的 coarse 匹配，分配一个伪监督以避免 DDP 死锁")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # 计算加权 L2 损失
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss
    
 

    @torch.no_grad()
    def compute_c_weight(self, data):
        """ 
        compute element-wise weights for computing coarse-level loss. 
        [V2.3] Spatial Loss Weighting Engine
        """
        # 1. 获取 coarse 级别的血管高斯软掩码
        weight0_c = data.get('vessel_weight0_c', None)
        weight1_c = data.get('vessel_weight1_c', None)

        if weight0_c is not None and weight1_c is not None:
            # 2. 构建基础血管概率图 W_pair (N, L, S) [0.0 ~ 1.0]
            w0 = weight0_c.flatten(-2) # [N, L]
            w1 = weight1_c.flatten(-2) # [N, S]
            W_pair = (w0[..., None] * w1[:, None, :]).float()
            
            # 3. 应用动态 Scaler 构建最终权重图
            # Formula: Weight = 1.0 + Mask * (Scaler - 1.0)
            # Background (Mask=0) -> Weight = 1.0
            # Vessel (Mask=1) -> Weight = Scaler (e.g., 10.0)
            scaler = self.vessel_loss_weight_scaler
            c_weight = 1.0 + W_pair * (scaler - 1.0)
            
            return c_weight

        # 回退逻辑：如果没有软掩码（比如纯背景图），尝试使用硬掩码
        # 但在 V2.3 中，只要有 vessel_mask 就会生成 soft mask，所以这里主要是防崩
        mask0 = data.get('mask0')
        mask1 = data.get('mask1')
        
        if mask0 is not None:
            # 如果没有血管权重但有 Padding Mask，则只处理 Padding
            # 这种情况下无法进行血管加权，只能返回 Padding Mask (0 or 1)
            c_weight = (mask0.flatten(-2)[..., None] * mask1.flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                else data['conf_matrix'],
            data['conf_matrix_gt'],
            weight=c_weight)
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
