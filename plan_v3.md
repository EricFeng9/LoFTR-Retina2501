# Plan v3: Light-weight Fine-tuning & Vessel-Guided Geometry Learning
我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。我们目前在用fractMorph模型进行2D化改造后的模型，但是fractMorph2D只能针对一些微小血管形变进行对齐和修复，但是针对真实情况不对齐图像中出现的血管大尺度旋转（如15 30度的旋转角度）、大的位置偏移的情况下配准效果较差，1是原先的逐像素点移动的模式很差2是我们修改成让模型直接预测放射参数，效果也很差，几乎不收敛。我现在的想法是改造loftr，学习cf图像和oct/fa图像之间的关键点匹配，然后利用这些匹配出来的关键点计算放射矩阵，对跨模态图像进行配准。
**目标**：解决 MINIMA 预训练权重在生成数据上微调后出现的“灾难性遗忘”问题，利用生成数据集的“完美血管掩码”优势，让模型专注学习眼底血管的几何拓扑关系。

---

## 核心策略变更 (Strategy Shift)

| 特性 | Plan v2 (Current) | Plan v3 (New) | 理由 |
| :--- | :--- | :--- | :--- |
| **Backbone** | **Dual-Stream (Split)** <br> $I^A, I^B$ 走不同网络 | **Shared (Frozen)** <br> $I^A, I^B$ 走同一网络 | 避免双流网络在简单合成数据上发生权重漂移，保持 MINIMA 通用特征空间一致性。 |
| **权重状态** | 全量微调 (Full Fine-tuning) | **冻结骨干 (Freeze Backbone)** <br> 仅训练 Transformer | MINIMA 特征已足够强大；冻结它防止过拟合合成数据的纹理伪影。 |
| **损失权重** | 软引导 (`clamp(mask, 0.1)`) | **硬约束 (Strict Vessel)** <br> Mask=1, Background=0.001 | 合成数据的背景毫无价值。强制模型**只看血管**，忽略一切背景噪声。 |
| **训练目标** | 像素级纹理匹配 | **几何拓扑匹配** | 即使图像是合成的，血管的几何形状（分叉/交叉）是真实的。迫使 Transformer 学习拓扑对齐。 |

---

## 1. 模型架构改造 (Architecture)

### 1.1 回归共享骨干 (Shared Backbone)
*   **操作**：撤销 `backbone0` 和 `backbone1` 的拆分，回归原始 LoFTR 的单 `backbone` 结构。
*   **输入**：
    *   $I^A (Fix)$: 完整图像
    *   $I^B (Moving)$: 完整图像
*   **特征提取**：
    *   $F^A = Backbone(I^A)$
    *   $F^B = Backbone(I^B)$
    *   由于 Backbone 冻结且共享，$F^A$ 和 $F^B$ 必定在同一个特征空间内（解决了 v2 中特征空间分离的问题）。

### 1.2 冻结策略 (Freezing Strategy)
在初始化模型后，遍历参数并设置 `requires_grad`：

*   **🚫 Frozen (不更新)**:
    *   `model.backbone` (ResNet-18 + FPN)
    *   `model.pos_encoding` (位置编码)
    *   `model.fine_preprocess` (精细特征预处理，通常也是简单的 Conv)
*   **✅ Trainable (仅更新)**:
    *   `model.loftr_coarse` (Linear Transformer)
    *   `model.loftr_fine` (Transformer / Attention)
    *   `model.coarse_matching` / `model.fine_matching` (如果是可学习的层)

---

## 2. 损失函数设计 (Vessel-Guided Loss)

利用生成数据集中**完美的血管掩码（Ground Truth Mask）**作为“上帝视角”，告诉 Transformer 哪里才是应该关注的地方。

### 2.1 极度严格的掩码权重
$$ W(i, j) = M^A(i) \cdot M^B(j) + \epsilon $$

*   **$M^A, M^B$**: 血管二值掩码（血管=1，背景=0）。
*   **$\epsilon$ (Epsilon)**: 设为极小值（如 `1e-4` 甚至 `0`），而不是之前的 `0.1`。
    *   **意图**：如果模型在背景区域预测错误，Loss 权重几乎为 0，不产生梯度。
    *   **后果**：模型不需要浪费容量去拟合背景（背景在合成数据中太简单，在真实数据中太复杂且无意义）。

### 2.2 几何一致性导向
由于我们只训练 Transformer，模型将不得不学习：
> *"虽然这两张图纹理不一样（一个是生成的，一个是真实的），但这个‘Y’字形分叉在这个位置，那个‘十字’交叉在那个位置，它们应该匹配。"*

这就是我们想要的**几何拓扑泛化能力**。

---

## 3. 训练流程 (Training Pipeline)

### 3.1 数据加载 (DataLoader)
*   **输入**：`img0`, `img1` (经过随机大规模仿射变换), `mask0`, `mask1` (同步变换)。
*   **增强**：
    *   (已移除光度增强，仅依赖几何结构监督)

### 3.2 训练脚本逻辑
1.  加载 MINIMA 预训练权重。
2.  冻结 Backbone 及相关层。
3.  开始训练 Loop：
    *   Forward Pass (得到 Coarse/Fine 匹配预测)。
    *   计算 Loss (通过 Strict Vessel Mask 加权)。
    *   Backward Pass (梯度只更新 Transformer)。

---

## 4. 预期收益 (Expected Outcome)

1.  **解决灾难性遗忘**：MINIMA 的特征提取能力 100% 保留。
2.  **提升几何匹配精度**：Transformer 专精于“连连看”游戏（血管对血管），不再被背景干扰。
3.  **极佳的泛化性**：因为模型学的是“形状匹配”而非“纹理匹配”，迁移到真实数据（Real Data）时，只要血管形状还在，就能对齐。
4.  **训练速度极快**：参数量减少 80% 以上，显存占用降低，收敛速度加快。
