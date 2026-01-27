# Plan v2.4: Scratch Training LoFTR for Multimodal Retina Registration

## 1. 核心改进理念 (Core Philosophy)
本方案旨在将 LoFTR (Local Feature Transformer) 适配于**多模态眼底图像配准 (Multimodal Retinal Registration)**。针对眼底图像纹理单一、血管结构细微且模态间差异巨大的挑战，我们对原始 LoFTR 进行了以下针对性改进：

**核心策略**：基于 V2.3 的血管感知引导 (Vessel-Aware Guidance) 和强化数据增强，**放弃使用 Megadepth 预训练权重 (Random Initialization)**。通过从零开始的实验，验证模型在眼底图像这一特定域内，是否能通过纯血管损失引导学习到更纯粹、更适配的几何特征，避免受自然场景先验的干扰。

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
*   **CLAHE 对比度增强 (全流程一致性)**: 
    *   在输入网络前，强制对所有图像应用 **CLAHE (Contrast Limited Adaptive Histogram Equalization)**。
    *   参数: `clip_limit=3.0`, `tile_grid_size=(8,8)`。
    *   **核心修复**: 通过重写 `_trainval_inference`，确保 **训练 (Training)** 和 **验证 (Validation)** 过程应用完全相同的 CLAHE。消除了之前验证集使用原图导致的分布漂移 (Distribution Shift)。
*   **强域随机化 (Consolidated Domain Randomization)**:
    *   在训练阶段，对生成数据（Generated Data）应用包含强度偏移、斑点噪声、泊松噪声、平滑 Dropout 的混合增强。
    *   **架构修复**: 合并了冗余的 `training_step` 定义，确保增强逻辑在每一个训练 Step 都能切实执行，而非仅作“回调”。

### 2.3 初始化策略 (Initialization Strategy - Scratch Training)
*   **操作**: **完全随机初始化 (No Pretrained Weights)**。
*   **原因**: 
    1.  **消除域偏见**: Megadepth 权重在由直线、平面构成的城市场景中训练，其学到的局部特征可能过于依赖边缘信息，而眼底血管的弯曲纹理与此不同。
    2.  **验证 Loss 效率**: 测试 V2.3 引入的血管感知 Loss $W$ 是否强大到足以引导一个完全白纸的模型学会配准。
    3.  **模型纯净化**: 探索在不依赖大规模预训练权重的情况下，LoFTR 架构在眼底数据上的收敛潜力。

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
*   **Monitor**: 监控 **Combined AUC** (AUC@5, 10, 20 的算术平均值)。
*   **Early Stopping**: 考虑到从随机初始化开始，模型需要较长的时间预热，**早停机制延迟至 100 epoch 后触发**。

---

## 4. 验证与评估体系 (Validation & Metrics)

鉴于眼底图像无法像自然图像那样通过肉眼直观判断匹配质量（尤其是重叠不准确时），我们建立了一套基于 **几何一致性** 的评估体系。

### 4.1 核心指标: 综合 AUC (Combined AUC)
*   **定义**: 取 `(auc@5 + auc@10 + auc@20) / 3`。
*   **优化理由**: 单一的 `auc@10` 容易受到局部运气成分影响（如某次验证刚好有几个点落在10px内）。综合指标要求模型在不同精度层级上均有提升，筛选出的 `Best Model` 在真实场景下更稳健，肉眼对齐效果更佳。

### 4.2 几何鲁棒性: RANSAC Threshold 调整
*   **设置**: 将 `RANSAC_PIXEL_THR` 从默认的 `0.5` 放宽至 **`2.0`**。
*   **物理考量**: 跨模态眼底图像（如 CF vs OCTA）中，血管中心点往往由于物理成像原理存在约 1 像素的天然偏移。过严的阈值会导致 RANSAC 误杀大量正确匹配，使指标虚低。`2.0` 像素是针对多模态配准的宽容度修正。

### 4.3 实时调试: 训练首批可视化 (Early-Batch Visualization)
*   **机制**: 在第 1 个 Epoch 的前 2 个 batch，自动输出：
    *   **增强对比图**: `comparison_orig_vs_aug.png` 系列，用于确认增强强度是否合理。
    *   **注册全流程可视化**: 自动运行一轮 Evaluation，生成棋盘图和匹配热图，确保模型在“带毒训练”初期依然具备基本的几何感官。

### 4.2 可视化辅助: RANSAC Filtering
*   **机制**: 在验证回调中，增加 RANSAC 几何校验步骤。
*   **输出**: 
    *   **Raw Matches (原始输出)**: 显示模型的所有预测，通常包含噪声。
    *   **RANSAC Inliers (绿线)**: 显示经几何变换矩阵过滤后的点。如果绿线稀少或杂乱，说明模型未收敛；如果绿线整齐平行，说明模型学会了。

---

## 5. 总结
本版本 (V2.4 Scratch) 是对 V2.3 方案的**消融验证与极限探索**。它保留了 V2.3 的 **CLAHE 增强**、**血管 Loss 加权** 和 **强域随机化**，但由于**撤销了预训练权重**，这要求模型必须完全依靠血管分割掩码提供的空间引导来建立特征感知。这是一个纯粹的“眼底原生 (Retina-Native)”训练实验方案。
