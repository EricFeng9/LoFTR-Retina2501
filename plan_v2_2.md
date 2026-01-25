[text](plan_v2.md)我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。我们目前在用fractMorph模型进行2D化改造后的模型，但是fractMorph2D只能针对一些微小血管形变进行对齐和修复，但是针对真实情况不对齐图像中出现的血管大尺度旋转（如15 30度的旋转角度）、大的位置偏移的情况下配准效果较差，1是原先的逐像素点移动的模式很差2是我们修改成让模型直接预测放射参数，效果也很差，几乎不收敛。我现在的想法是改造loftr，学习cf图像和oct/fa图像之间的关键点匹配，然后利用这些匹配出来的关键点计算放射矩阵，对跨模态图像进行配准。

# Plan v2.2: Gentle Mask Supervision (Curriculum Learning)

## 核心动机：解决 Mode Collapse
*   **现象**：在 v2.1 (无 Mask) 训练中，真实数据的匹配点全部坍塌到图像上部 (Mode Collapse)。
*   **原因**：由于 Sim-to-Real 的域差异，真实图像下半部分的弱纹理区域无法激活 LoFTR 特征。Transformer 在无引导情况下，学会了将所有点映射到 FA 图像顶部唯一的高响应区域。
*   **对策**：必须在训练初期强制模型“看血管”，利用血管的拓扑结构作为唯一可靠的跨模态锚点。

## 核心机制：三阶段课程学习 (Curriculum Learning)

通过动态调整 `vessel_soft_lambda` (注意力偏置) 和 `loss_weight` (血管损失权重)，实现从“强监督”到“自适应”的过渡。

### 阶段一：教学期 (Teaching Phase) —— "强制关注"
*   **Epoch 0 - 25**
*   `vessel_soft_lambda = 2.0` (强注意力偏置)
*   `loss_weight = 5.0` (强惩罚)
*   **目的**：强迫 Transformer 忽略背景噪声，锁定血管结构。打破“瞎猜”的局部最优解。

### 阶段二：断奶期 (Weaning Phase) —— "逐渐放手"
*   **Epoch 25 - 50**
*   `vessel_soft_lambda`: 线性衰减 2.0 -> 0.0
*   `loss_weight`: 线性衰减 5.0 -> 1.0
*   **目的**：随着外部辅助减弱，模型必须把“关注血管”的知识内化到权重中。

### 阶段三：独立期 (Independence Phase) —— "自由泛化"
*   **Epoch 50+**
*   `vessel_soft_lambda = 0.0`
*   `loss_weight = 1.0`
*   **目的**：完全依靠学习到的特征进行匹配，允许模型探索背景上下文（如果有用的话），此时模型已有良好的初始化状态。

## 代码实现计划

### 1. 新增训练脚本 `train_multimodal_onGen_v2_2.py`
基于 v2.1 修改，主要改动如下：

#### A. 恢复 Mask 输入
*   移除 v2.1 中 `batch.pop('mask0')` 等代码。
*   确保 DataModule 正常输出 `mask0` / `mask1`。

#### B. 实现 `CurriculumScheduler` Callback
```python
class CurriculumScheduler(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        # 计算当前参数
        if epoch < 30:
            sched_lambda = 2.0
            sched_loss_w = 5.0
        elif epoch < 80:
            progress = (epoch - 30) / (80 - 30)
            sched_lambda = 2.0 * (1.0 - progress)
            sched_loss_w = 5.0 - (4.0 * progress)
        else:
            sched_lambda = 0.0
            sched_loss_w = 1.0
            
        # 注入模型
        # 注意：PL_LoFTR 内部可能有 matcher 属性
        if hasattr(pl_module, 'matcher') and hasattr(pl_module.matcher, 'coarse_matching'):
            # 这里的 vessel_soft_lambda 位于 coarse_matching 模块
            pl_module.matcher.coarse_matching.vessel_soft_lambda = sched_lambda
            
        # Loss weight 注入
        if hasattr(pl_module.loss, 'background_weight'):
             # v2.1 是 background_weight=1.0, 我们需要调整它
             # 假设 loss logic 是: loss = w_vessel * loss_v + w_bg * loss_bg
             # 或者 mask weight 逻辑. 之前代码是 vessel_loss_weight
             # 检查 LoFTRLoss 定义. 假设我们通过调整 matcher.loss_weight 来控制
             pass 

        # 记录日志
        pl_module.log('sched/lambda', sched_lambda, on_epoch=True)
        pl_module.log('sched/loss_w', sched_loss_w, on_epoch=True)
```

#### C. 调整 `PL_LoFTR`
*   在脚本中定义 `PL_LoFTR_V2` 继承自 `PL_LoFTR`。
*   覆盖 `on_train_epoch_start`，避免硬编码的 0.0 覆盖了 Scheduler 的值。

## 预期结果
*   **极早期 (Epoch 5)**：真实数据 MACE 应该依然很高（因为刚开始），但 visualization 的红线应该不再是“一束全都指向上方”，而是开始散布在血管周围（虽然乱，但分布应该均匀）。
*   **中期 (Epoch 40)**：随着 Mask 引导生效，红线应该能大致对齐粗血管。
