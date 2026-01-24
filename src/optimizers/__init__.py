import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


def build_optimizer(model, config):
    """
    构建优化器。
    
    【最简方案】统一学习率，所有参数使用相同的学习率。
    让 Transformer 自己学习跨模态匹配，不做特殊处理。
    """
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR

    # 【方案 B 改进】分层学习率
    # Backbone 使用 0.1x LR 微调
    # Transformer 及其他部分使用 1.0x LR 从头学习
    
    # 获取需要优化的参数
    backbone_params = []
    transformer_params = []
    
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 根据参数名判断属于 backbone 还是 transformer
        # LoFTR 中 backbone 参数通常以 'matcher.backbone' 开头
        if 'backbone' in param_name:
            backbone_params.append(param)
        else:
            transformer_params.append(param)
            
    param_groups = [
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': transformer_params, 'lr': lr * 1.0}
    ]

    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.TRAINER.SCHEDULER

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config.TRAINER.MSLR_MILESTONES, gamma=config.TRAINER.MSLR_GAMMA)})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    else:
        raise NotImplementedError()

    return scheduler
