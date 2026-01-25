[text](plan_v2.md)我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。我们目前在用fractMorph模型进行2D化改造后的模型，但是fractMorph2D只能针对一些微小血管形变进行对齐和修复，但是针对真实情况不对齐图像中出现的血管大尺度旋转（如15 30度的旋转角度）、大的位置偏移的情况下配准效果较差，1是原先的逐像素点移动的模式很差2是我们修改成让模型直接预测放射参数，效果也很差，几乎不收敛。我现在的想法是改造loftr，学习cf图像和oct/fa图像之间的关键点匹配，然后利用这些匹配出来的关键点计算放射矩阵，对跨模态图像进行配准。

@data/CF_260116_expand 数据集每一个文件夹下面是一组多模态数据集，格式为[编号]_[模态]_[其他备注].png。配准模式共有cffa（cf为fix），cfoct（cf为fix），octfa（oct为fix），cfocta（cf为fix）四种模式。_cf_clip_512(或者_cf_gen_clip_512)和_octa_gen用于cfocta配准的训练，剩下的_cf_512(_cf_gen_512)、_fa(_fa_gen)、_oct(_oct_gen)分别用于对应的cffa和cfoct和faoct配准。
**重要更新：v2.1 (No Mask Strategy)**
本版本完全移除血管掩码引导策略。
- `vessel_mask = None`
- `vessel_soft_lambda = 0`
- `loss_weight = 1.0`
模型将完全依靠图像特征进行自主学习，不再依赖任何辅助的血管分割信息。

---

为了实现基于 LoFTR 的眼底图像跨模态配准，你需要将原有的“基于变形场”或“直接参数回归”的思路转变为“**基于特征匹配的几何解算**”思路 。

以下是为您整理的详细实现与训练计划，包含核心公式与步骤：

---

## 核心改进总结：完全自主学习 (Autonomous Learning)

**设计理念：** 完全摒弃人工先验（血管掩码），让 Transformer 完全依靠数据驱动的方式学习跨模态特征对应关系。

改进后的 LoFTR 模型采用以下**端到端学习**策略：

1.  **全局无偏置学习：** 不使用任何掩码偏置（Bias），让 Transformer 自动捕捉图像中的显著特征（无论是血管还是其他纹理）。
2.  **统一损失权重：** 所有区域的损失权重均为 `1.0`，不进行加权。模型必须自己学会区分哪些区域是可靠的匹配点。
3.  **保留完整图像输入：** 依然输入完整图像（含背景），不进行预过滤。
4.  **空间均匀化采样：** 推理时保持全图 8×8 空间均匀化采样，依靠模型输出的置信度来筛选高质量点。

---

## 第零阶段：权重初始化 (Hybrid Initialization)

采用**混合初始化策略**：保留 LoFTR 预训练的 Backbone 权重，但对 Transformer 进行随机初始化。

> [!IMPORTANT]
> LoFTR outdoor_ds 权重是在自然场景（MegaDepth）上训练的同模态匹配模型，**不具备跨模态匹配能力**。
> 因此，Transformer 部分学到的注意力模式对跨模态眼底配准**完全不适用**，需要从头学习。

### 初始化策略

| 模块 | 初始化方式 | 学习率 | 理由 |
|------|-----------|--------|------|
| **Backbone (ResNetFPN)** | LoFTR 预训练权重 | 低 (0.1x) | 低层视觉特征（边缘、纹理）可迁移，只需微调 |
| **Transformer (Coarse+Fine)** | 随机初始化 | 高 (1x) | 从零学习跨模态匹配逻辑 |

### 实现代码

```python
# 加载权重时，只加载 Backbone 部分
state_dict = torch.load('weights/outdoor_ds.ckpt')['state_dict']

# 过滤掉 Transformer 相关权重
backbone_only = {k: v for k, v in state_dict.items() 
                 if k.startswith('backbone')}

model.load_state_dict(backbone_only, strict=False)
```

### 分层学习率优化器

```python
optimizer = torch.optim.AdamW([
    # Backbone: 小学习率，微调
    {'params': model.matcher.backbone.parameters(), 'lr': lr * 0.1},
    # Transformer: 大学习率，从头学习
    {'params': model.matcher.loftr_coarse.parameters(), 'lr': lr},
    {'params': model.matcher.loftr_fine.parameters(), 'lr': lr},
    {'params': model.matcher.coarse_matching.parameters(), 'lr': lr},
    {'params': model.matcher.fine_preprocess.parameters(), 'lr': lr},
    {'params': model.matcher.fine_matching.parameters(), 'lr': lr},
], weight_decay=0.1)
```

---

## 第一阶段：模型架构改造 (Architecture)

LoFTR 的核心是舍弃特征检测器，直接在稠密特征上通过 Transformer 建立联系 。

### 1. 双流特征提取 (Backbone)

针对 CF 与 OCT/FA 的巨大模态差异，建议将原本共享权重的 ResNet-18 修改为**双流不共享权重**的结构 。
输入完整图像（含背景）。

### 2. 线性 Transformer 模块 (Linear Attention)

由于眼底图分辨率高，必须使用**线性注意力机制**。

**移除注意力偏置：**
- `vessel_soft_lambda = 0`
- 注意力计算不再包含 `vessel_bias` 项。
- 公式简化为原始 LoFTR 形式：$sim(Q, K) = \phi(Q) \cdot \phi(K)^T$

---

## 第二阶段：训练数据生成 (GT Label Generation)

利用你现有的**已对齐生成数据集**，你可以自动生成完美的训练标签 。

### 1. 模拟放射变换

在训练时，随机生成一个放射变换矩阵 $T$（包含 $\pm 90^\circ$ 的旋转、平移和缩放），并作用于 $I^B$ 得到 $I_{warped}^B$。

**增强策略：**
* **旋转：** 均匀采样 $[-90^\circ, 90^\circ]$
* **翻转：** 20% 概率水平/垂直翻转

### 2. 建立坐标映射 (Ground Truth)

对于 $I^A$ 中的任意像素点 $i = (x, y)$，其对应的真值点 $j_{gt}$ 在 $I_{warped}^B$ 中的位置为：
$$j_{gt} = T \cdot [x, y, 1]^T$$

**移除掩码权重：**
- 所有点对的权重均为 1.0（前提是 mutual_nearest 相互最近邻或在阈值范围内）。
- 不再根据血管掩码调整正样本权重。

---

## 第三阶段：损失函数设计 (Supervision)

训练过程是端到端的，强制模型忽略模态差异 。

### 1. 粗级匹配损失 ($\mathcal{L}_c$)

使用**双软最大值 (Dual-softmax)** 建立概率矩阵 $\mathcal{P}_c$。

**移除梯度保底策略：**
- `loss_weight = 1.0`（全图均匀权重）
- 不再对背景区域进行 `clamp(min=0.1)` 降权，也不对血管区域加权。

### 2. 精级细化损失 ($\mathcal{L}_f$)

在选定的粗匹配点周围截取 $5 \times 5$ 的窗口，通过期望值预测亚像素偏移。
损失计算不使用任何 mask 加权。

---

## 第四阶段：训练策略：完全自主学习

> [!NOTE]
> 移除所有课程学习策略 (Schdule)，因为模型已经简化。
> **新增 Early Stopping 冷启动策略**：为了防止模型在 Transformer 权重调整初期（loss震荡期）被错误终止，前 100 epoch 不进行早停检查。

- **始终保持：** `vessel_soft_lambda = 0`
- **始终保持：** `loss_weight = 1.0`
- **早停策略：** `DelayedEarlyStopping(start_epoch=100, patience=5)`

---

## 第五阶段：配准解算 (Inference & Registration)

同 v2.0，模型输出点对后：
1. **空间均匀化采样 (Spatial Binning)**：全图 8×8 网格，每格选 Top-K。
2. **RANSAC**：剔除误匹配，解算放射矩阵。

---

## 第六阶段：数据增强策略 (Data Augmentation)

仍需对图像进行放射变换增强，但**不需要**再同步处理血管掩码（因为模型不再使用）。

---

## 总结

本方案 (v2.1) 是 v2.0 的极简版本，核心在于**相信 Transformer 的特征学习能力**，移除了所有人为设计的血管先验。这是一次“回归本源”的尝试，旨在验证 LoFTR 架构在不依赖掩码的情况下，能否通过纯大数据驱动学会跨模态配准。
