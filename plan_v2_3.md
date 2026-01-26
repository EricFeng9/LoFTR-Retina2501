# Plan v2.3: Minimalist Vessel-Aware LoFTR (Simpler is Better)

## 核心理念 (Philosophy)
**"Occam's Razor": 如无必要，勿增实体。**
相信 Transformer 的特征提取能力，移除人为的复杂规则（Mask注入、GT清洗、课程学习），回归端到端学习的本质。

## 1. 核心策略：Training-Only Loss Guidance
我们不再修改模型结构，不再干扰推理过程，仅在**训练损失函数**层面施加血管结构约束。

*   **唯一机制**：**Spatial Loss Weighting (空间损失加权)**
    *   在计算 Coarse Loss 时，根据 Ground Truth 的血管 Mask 构建权重图。
    *   **In-Vessel (血管区域)**: Weight = `10.0` (固定值，全程不变)。
    *   **Background (背景区域)**: Weight = `1.0`。
    *   **目的**：通过梯度的差异化，隐式驱动 Transformer 的 Attention Head 自动聚焦于高权重的血管纹理，而非通过硬编码喂给它。

## 2. 移除的组件 (Removed Components)
为了净化模型学习过程，我们移除了 V2.1/V2.2 中引入的过度工程化组件：
*   **[移除] Curriculum Scheduler**: 不再需要复杂的 Epoch 调度，全程策略统一。
*   **[移除] Mask Injection**: 推理时不再需要 Mask 输入，前向传播 (Forward) 保持原汁原味的 LoFTR 逻辑。
*   **[移除] Attention Bias**: 不再在前向 Attention Map 上叠加硬偏置，让模型自己学 Attention。
*   **[移除] GT Sharpening**: 不再清洗 GT，保留模型对 Context 的理解能力。

## 3. 保留的优化 (Retained Optimizations)
*   **CLAHE Preprocessing**: 这是一个合理的域适配预处理，增强输入图像的对比度，帮助模型“看清”血管。
*   **Input Normalization**: 强制 [0, 1] 范围，保证数值稳定性。

## 4. 预期效果
模型在训练初期可能收敛稍慢（因为没有 Teacher 手把手带），但最终学出的特征应该更鲁棒，能够真正理解血管结构，且**完全不依赖分割 Mask 进行推理**，实现真正的端到端配准。
