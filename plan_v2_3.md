# Plan v2.3: Vessel-Aware LoFTR for Multimodal Retina Registration

## 1. 核心改进理念 (Core Philosophy)
本方案旨在将 LoFTR (Local Feature Transformer) 适配于**多模态眼底图像配准 (Multimodal Retinal Registration)**。针对眼底图像纹理单一、血管结构细微且模态间差异巨大的挑战，我们对原始 LoFTR 进行了以下针对性改进：

**核心策略**：通过**训练端空间引导 (Training-time Spatial Guidance)** 强迫 Transformer 学习血管结构，结合**全量预训练先验 (Full Pretrained Priors)** 和**强健的评估指标 (AUC)**，实现端到端的高精度配准。

---

## 2. 关键技术改进 (Key Implementation Details)

### 2.1 血管感知损失引导 (Vessel-Aware Loss Guidance)
原始 LoFTR 对所有匹配区域一视同仁，但在眼底图像中，**血管交叉点和纹理**才是配准的关键，大面积的平坦背景（如视网膜背景）往往提供错误的低频信息。
*   **改进机制**：引入 **Static Spatial Loss Weighting**。
*   **实现细节**：
    *   利用训练数据中的 Ground Truth 血管分割掩码 (Vessel Mask)。
    *   在计算 Coarse-Level Loss 时，构建像素级权重图 $W$。
    *   **In-Vessel (血管区域)**: 权重设为 $scaler = 10.0$。
    *   **Background (背景区域)**: 权重设为 $1.0$。
*   **效果**：通过梯度的差异化，隐式驱动 Transformer 的 Attention Head 在反向传播时更多地更新与血管特征相关的参数，使其学会"关注血管"。

### 2.2 强化数据预处理 (Enhanced Preprocessing)
原始 LoFTR 假设输入为自然场景图像，而眼底图像通常对比度极低，且不同模态（CF, FA, OCT）间灰度分布差异巨大。
*   **CLAHE 对比度增强**: 
    *   在输入网络前，强制对所有图像应用 **CLAHE (Contrast Limited Adaptive Histogram Equalization)**。
    *   参数: `clip_limit=3.0`, `tile_grid_size=(8,8)`。
    *   **目的**: 显著增强微细血管与背景的对比度，使特征更易被 Backbone 提取。
*   **强域随机化 (Strong Domain Randomization)**:
    *   在训练阶段，对生成数据（Generated Data）进行大幅度的亮度、对比度、Gamma 扰动。
    *   **目的**: 模拟真实临床数据的复杂分布，提高模型的泛化能力 (Sim-to-Real)。

### 2.3 初始化策略 (Initialization Strategy)
*   **操作**: 强制加载 **Full MegaDepth Pretrained Weights** (Backbone + Transformer)。
*   **原因**: 原始 LoFTR 在自然场景（MegaDepth）上学到了非常通用的几何变换规则。虽然应用场景变成了眼底，但底层的“角点检测”和“几何一致性”逻辑是通用的。这比随机初始化能更快地让模型输出有意义的匹配。

---

## 3. 训练策略 (Training Strategy)

### 3.1 模型架构微调
*   **Backbone**: 保持 ResNet-FPN 结构不变。
*   **Positional Encoding**: 保持不变。
*   **Coarse/Fine Module**: 保持不变。
*   **改动点**: 仅修改了 `LoFTRLoss` 模块，使其支持接收外部传入的 `vessel_loss_weight_scaler` 并应用于 Coarse Loss 计算。

### 3.2 优化器与调度
*   **Optimizer**: AdamW, `weight_decay=0.01`。
*   **Batch Size**: 4 (受限于显存，配合梯度累积)。
*   **Monitor**: 监控 `auc@10` (Area Under Curve @ 10px error)。

---

## 4. 验证与评估体系 (Validation & Metrics)

鉴于眼底图像无法像自然图像那样通过肉眼直观判断匹配质量（尤其是重叠不准确时），我们建立了一套基于 **几何一致性** 的评估体系。

### 4.1 核心指标: AUC@10
*   **定义**: 计算配准误差小于 10 像素的样本比例及其分布。
*   **取代 MACE**: 放弃传统的平均误差 (MACE)，因为在存在大量 Outliers（配准完全失败）的情况下，MACE 均值没有参考价值。AUC 能过滤掉失败样本，只统计成功配准的比例，指标更硬核。

### 4.2 可视化辅助: RANSAC Filtering
*   **机制**: 在验证回调中，增加 RANSAC 几何校验步骤。
*   **输出**: 
    *   **Raw Matches (原始输出)**: 显示模型的所有预测，通常包含噪声。
    *   **RANSAC Inliers (绿线)**: 显示经几何变换矩阵过滤后的点。如果绿线稀少或杂乱，说明模型未收敛；如果绿线整齐平行，说明模型学会了。

---

## 5. 总结
本版本 (V2.3 Robust) 是 LoFTR 在眼底配准领域的**定制化适配版**。它保留了 LoFTR 强大的 Dense Matching 能力，通过 **CLAHE 增强** 让它“看清”血管，通过 **Loss 加权** 强迫它“记住”血管，最后通过 **全量预训练** 赋予它几何常识。这是一个不依赖推理时 Mask 输入的、端到端的鲁棒配准方案。
