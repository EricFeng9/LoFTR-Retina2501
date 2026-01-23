我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。我们目前在用fractMorph模型进行2D化改造后的模型，但是fractMorph2D只能针对一些微小血管形变进行对齐和修复，但是针对真实情况不对齐图像中出现的血管大尺度旋转（如15 30度的旋转角度）、大的位置偏移的情况下配准效果较差，1是原先的逐像素点移动的模式很差2是我们修改成让模型直接预测放射参数，效果也很差，几乎不收敛。我现在的想法是改造loftr，学习cf图像和oct/fa图像之间的关键点匹配，然后利用这些匹配出来的关键点计算放射矩阵，对跨模态图像进行配准。

@data/CF_260116_expand 数据集每一个文件夹下面是一组多模态数据集，格式为[编号]_[模态]_[其他备注].png。配准模式共有cffa（cf为fix），cfoct（cf为fix），octfa（oct为fix），cfocta（cf为fix）四种模式。_cf_clip_512(或者_cf_gen_clip_512)和_octa_gen用于cfocta配准的训练，剩下的_cf_512(_cf_gen_512)、_fa(_fa_gen)、_oct(_oct_gen)分别用于对应的cffa和cfoct和faoct配准。

**重要更新：** 每个数据集文件夹现在包含一张血管分割掩码图（格式如 [编号]_vessel_mask.png），用于表示该组数据中所有模态（CF/OCT/FA/CFCLIP/OCTA）共同的血管结构。该掩码**仅用于损失函数加权和注意力偏置**，不直接输入给模型作为特征通道，避免模型过度依赖掩码。由@data/CF_260116_expand/cf_260116_expand.py 根据需要返回给训练和测试脚本

为了实现基于 LoFTR 的眼底图像跨模态配准，你需要将原有的“基于变形场”或“直接参数回归”的思路转变为“**基于特征匹配的几何解算**”思路 。

以下是为您整理的详细实现与训练计划，包含核心公式与步骤：

---

## 核心改进总结：从"硬约束"转为"端到端学习" (参考 RoMa 方案)

**设计理念：** 不将血管掩码直接输入给模型，而是通过软引导机制让模型端到端学习血管区域的重要性。

改进后的 LoFTR 模型采用以下**端到端学习**策略：

1.  **保留完整图像输入：** 不在数据加载时对图像进行有效区域过滤（背景置零），让 Backbone 看到完整图像（含背景）。LoFTR 需要全局上下文来理解旋转和大尺度偏移，背景区域也提供位置编码所需的信号。
2.  **注意力解耦与偏置 (Attention Biasing)：** Transformer 注意力机制采用**加性偏置 (Additive Bias)**，通过课程学习动态调整 $\lambda$。这保证了梯度的连续性，引导模型向血管区域倾斜而不直接关闭背景通道。
3.  **损失监督与梯度保底 (Loss Weighting)：** 损失函数使用 `torch.clamp(mask, min=0.1)` 给背景区域预留 0.1 的"低保"权重。这确保即使匹配点落在背景上，也能产生微弱梯度推动模型学习大尺度的几何变换。
4.  **移除推理时的硬过滤：** 不在匹配点选择时根据掩码硬过滤（丢弃背景点），让模型的置信度预测决定点的质量。
5.  **空间均匀化无掩码约束：** 在 RANSAC 前进行全图 8×8 空间均匀化采样，不使用掩码限制网格范围，让模型自动学习给血管区域高置信度。
6.  **评估时过滤有效区域：** 只在计算 MSE 等评估指标时使用有效区域过滤，避免背景噪声影响评估结果。

---

## 第零阶段：权重初始化 (Initialization with MINIMA)

为了加速收敛并利用大规模数据预训练的优势，我们将**加载 MINIMA 在 LoFTR 上的预训练权重**作为初始状态。

*   **预训练权重来源：** MINIMA (LoFTR 改进版) 提供的预训练模型。
*   **加载策略：** 
    *   由于后续我们将改造为双流骨干网络（Dual-Stream Backbone），而预训练权重通常基于单流（Shared Backbone）。
    *   **权重迁移：** 需要将预训练的 Backbone 权重**复制两份**，分别初始化 $I^A$ 和 $I^B$ 的独立特征提取网络。
    *   **Transformer 权重：** 直接加载 Transformer 部分的权重。
    *   **严格模式：** 使用 `strict=False` 加载，确保兼容性。

---

## 第一阶段：模型架构改造 (Architecture)

LoFTR 的核心是舍弃特征检测器，直接在稠密特征上通过 Transformer 建立联系 。

### 1. 双流特征提取 (Backbone)

针对 CF 与 OCT/FA 的巨大模态差异，建议将原本共享权重的 ResNet-18 修改为**双流不共享权重**的结构 。

* **输入：** 图像对 $I^A$ (固定图 CF) 和 $I^B$ (待配准图 OCT/FA)，**完整图像（含背景）**。
* **输出：** 提取两层特征。
* **粗级特征：** $\tilde{F} \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times D}$，用于全局匹配 。
* **精级特征：** $\hat{F} \in \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times d}$，用于亚像素微调 。

* **关键原则（参考 RoMa）：**
  * **❌ 不将血管掩码 $M^{vessel}$ 作为输入通道拼接到图像**
  * **❌ 不在 Backbone 中使用 `feat_c *= (α W + β)` 进行特征截断**
  * **✅ Backbone 只提取纯净的图像特征，保持特征空间的完整性**
  * **✅ 血管掩码信息仅通过 Transformer 的加性偏置和损失加权进入模型**
  * **物理意义：** 让 Backbone 看到完整图像，有助于 Transformer 建立正确的全局位置编码，处理大尺度旋转和偏移。





### 2. 线性 Transformer 模块 (Linear Attention with Masking)

由于眼底图分辨率高，必须使用**线性注意力机制**以保证计算量在 $O(N)$ 级别 。其核心公式如下：
$$sim(Q, K) = \phi(Q) \cdot \phi(K)^T$$

其中 $\phi(x) = \text{elu}(x) + 1$。
通过矩阵乘法的结合律，计算流程变为：
$$\text{Output} = (\phi(Q) \cdot (\phi(K)^T \cdot V)) / (\phi(Q) \cdot \sum \phi(K)^T)$$

**引入解剖掩码的注意力加性偏置（训练时软引导）：**

采用**加性偏置 (Additive Bias)** 在训练时软性引导模型关注血管区域，而非硬约束：

* **注意力加性偏置（仅训练时启用）：**
  - 在 coarse 级别下采样血管掩码得到 $\tilde{M}^A, \tilde{M}^B \in [0,1]^{\frac{H}{8}\times\frac{W}{8}}$
  - 构造匹配对的血管偏置项：
  \[
  vessel\_bias(i,j) = \tilde{M}^A(i) \cdot \tilde{M}^B(j)
  \]
  - 在相似度计算中使用加性偏置：
  \[
  S_{biased}(i,j) = S(i,j) + \lambda \cdot vessel\_bias(i,j)
  \]
  - **课程学习动态调整 $\lambda$：**
    - 阶段1 (0-25 epochs)：$\lambda = 0$（全图感知期）
    - 阶段2 (25-50 epochs)：$\lambda = 0.2$（软掩码引导期）
    - 阶段3 (50+ epochs)：$\lambda = 0.05$（精度冲刺期）
  - **推理/测试时：$\lambda = 0$** （完全关闭血管掩码偏置，避免对掩码依赖）

* **物理意义：**  
  - 加性偏置允许非血管区域保留微弱的响应，保证了梯度的连续性
  - 引导注意力机制向血管区域倾斜，而不是直接关闭非血管通道
  - 通过课程学习，模型从全图感知逐步过渡到血管关注，最终在推理时不依赖掩码

这种方式让 $I^A$ 的每一个像素都能感知 $I^B$ 的全局血管拓扑结构：既能处理大尺度旋转（±90°），又避免了过度依赖人工掩码，提升泛化能力。

---

## 第二阶段：训练数据生成 (GT Label Generation)

利用你现有的**已对齐生成数据集**，你可以自动生成完美的训练标签 。

### 1. 模拟放射变换

在训练时，随机生成一个放射变换矩阵 $T$（包含 $\pm 90^\circ$ 的旋转、平移和缩放），并作用于 $I^B$ 得到 $I_{warped}^B$。

**增强策略：**
* **旋转：** 均匀采样 $[-90^\circ, 90^\circ]$ 范围内的角度，覆盖大角度旋转情况
* **翻转：** 以小概率（10%）随机应用水平翻转或垂直翻转，模拟不同采集方向
  * 水平翻转概率：10%
  * 垂直翻转概率：10%
  * 翻转与旋转可组合使用

### 2. 建立坐标映射 (Ground Truth with Vessel Constraint)

对于 $I^A$ 中的任意像素点 $i = (x, y)$，其对应的真值点 $j_{gt}$ 在 $I_{warped}^B$ 中的位置为：
$$j_{gt} = T \cdot [x, y, 1]^T$$

**引入血管掩码的GT标注（用于损失权重，不做硬筛选）：**

* **粗级标签 $\mathcal{M}_c^{gt}$：** 对于所有满足重投影距离小于阈值（如 1 个网格单位）的点对 $(\tilde{i}, \tilde{j})$：
  1. 标记为正样本对（不根据掩码值丢弃）
  2. 为每个点对分配权重 $w(\tilde{i}, \tilde{j}) = \tilde{M}^A(\tilde{i}) \cdot \tilde{M}^B(\tilde{j})$
  3. **不丢弃背景点对**，保留所有有效的几何对应关系
  
* **精级标签 $\mathcal{M}_f^{gt}$：** 直接使用 $j_{gt}$ 作为 $L_2$ 损失的目标坐标，**不做掩码筛选，保留所有点对**。

**作用：** 
- ✅ 保留完整的几何对应关系，让模型学习全图的大尺度变换
- ✅ 通过损失权重引导模型关注血管区域，而非硬性丢弃背景点
- ✅ 背景区域的微弱梯度（通过 `clamp(min=0.1)`）有助于模型学习旋转和偏移

---

## 第三阶段：损失函数设计 (Supervision)

训练过程是端到端的，通过两个损失函数强制模型忽略模态差异 。

### 1. 粗级匹配损失 ($\mathcal{L}_c$) with Background Gradient Guard

使用**双软最大值 (Dual-softmax)** 建立概率矩阵 $\mathcal{P}_c$ ：
$$\mathcal{P}_c(i, j) = \text{softmax}(S_{biased}(i, \cdot))_j \cdot \text{softmax}(S_{biased}(\cdot, j))_i$$

其中 $S_{biased}(i, j)$ 使用了**加性偏置**。

**权重设计 (Gradient Guard)：**
使用 `torch.clamp(mask, min=0.1)` 给非血管区域保底权重。

$$\mathcal{L}_c = -\frac{1}{|\mathcal{M}_c^{gt}|} \sum_{(\tilde{i}, \tilde{j}) \in \mathcal{M}_c^{gt}} \text{clamp}(vessel\_weight(\tilde{i}, \tilde{j}), \min=0.1) \cdot \log \mathcal{P}_c(\tilde{i}, \tilde{j})$$

其中 $vessel\_weight(\tilde{i}, \tilde{j}) = \tilde{M}^A(\tilde{i}) \cdot \tilde{M}^B(\tilde{j})$

* **梯度保底机制：** 将权重最小值限制为 **0.1**（血管区域权重为 1.0）
* **物理意义：** 即使点对落在背景上，也会产生微弱的梯度（0.1）推动模型学习大尺度的几何变换。血管区域的高权重（1.0）则负责后期的精细对齐。
* **实现：** `loss_weight = torch.clamp(vessel_mask_pair, min=0.1)`

**意义：** 引导 Transformer 优先学习血管分叉点、交叉点等具有显著拓扑特征的关键点匹配 。

### 2. 精级细化损失 ($\mathcal{L}_f$) with Soft Weighting

在选定的粗匹配点周围截取 $w \times w$ 的窗口，通过期望值预测亚像素偏移 $\hat{j}'$ 。

**软权重策略（不做硬筛选）：**
* **不剔除窗口：** 保留所有通过置信度阈值的粗匹配点窗口，不根据窗口内血管占比丢弃
* **软权重计算：** 为每个窗口分配权重 $w_{window} = \text{clamp}(\text{mean}(\tilde{M}_{window}), \min=0.1)$
* **端到端学习：** 让模型学习在不同区域的精细匹配能力，通过权重自然引导

损失函数为加权 $L_2$ 距离：
$$\mathcal{L}_f = \frac{1}{|\mathcal{M}_f|} \sum_{(\hat{i}, \hat{j}') \in \mathcal{M}_f} w_{window}(\hat{i}) \cdot \frac{1}{\sigma^2(\hat{i})} ||\hat{j}' - \hat{j}_{gt}'||_2$$

其中：
* $\sigma^2(\hat{i})$ 代表热图的方差（不确定性），方差越大的点权重越低
* $w_{window}(\hat{i})$ 是窗口的血管权重（最小值为 0.1）
* $\mathcal{M}_f$ 包含所有有效的精细匹配点（不做掩码筛选）

**亚像素位置计算：** 对窗口内所有像素进行加权和计算期望值 $\hat{j}'$，不做掩码过滤，保持端到端学习。

---

## 第四阶段：训练策略：三阶段课程学习 (Curriculum Learning)

让模型先学"简单的全图对齐"，再学"难的血管对齐"，最后减弱引导确保泛化。

* **阶段 1 (前 20% Epochs)：全图感知期**
  - 设置 $\lambda = 0$，不使用血管掩码注意力偏置
  - 损失函数仍使用 `clamp(mask, min=0.1)` 的梯度保底权重
  - 让模型先利用全图信息学会处理 $\pm 90^\circ$ 的旋转和大尺度偏移
  
* **阶段 2 (20%-70% Epochs)：软掩码引导期**
  - 设置 $\lambda = 0.2$，引入弱的血管掩码加性偏置
  - 模型开始向血管聚焦，但在非血管区域保留梯度
  - 平衡全局感知和局部精细对齐
  
* **阶段 3 (70%-100% Epochs)：精度冲刺期**
  - 设置 $\lambda = 0.05$，使用更弱的血管权重
  - 避免模型过度依赖掩码，提升泛化能力
  - **早停机制**：只在此阶段开启早停，patience=8，min_epochs=70% of total

**验证/测试阶段：**
- 设置 $\lambda = 0$，完全关闭血管分割图的注意力偏置
- 确保模型在纯图像特征下进行推理，避免对掩码的依赖
- 评估时使用 `filter_valid_area()` 只在有效区域计算指标

---

## 第五阶段：配准解算 (Inference & Registration)

模型训练完成后，配准不再是预测变形场，而是解算矩阵。

### 推理配置

**重要：验证/测试时关闭血管掩码偏置**
- 设置 $\lambda = 0$，完全关闭 Transformer 注意力机制中的血管掩码偏置
- 原因：
  1. 避免模型对掩码产生依赖，确保在没有精确掩码的真实场景下也能工作
  2. 测试模型在纯图像特征下的泛化能力
  3. 课程学习已经让模型学会了血管区域的重要性，推理时不需要显式引导

### 配准流程

1. **特征匹配（$\lambda = 0$，无掩码过滤）：** 
   * 输入未对齐的完整图像 $I^A, I^B$（包含背景，不做预过滤）
   * 掩码在推理时不参与注意力计算（$\lambda = 0$）
   * LoFTR 输出点对集合 $\{(x_k, y_k) \leftrightarrow (x'_k, y'_k)\}$ 及其置信度
   * **❌ 不使用掩码硬过滤匹配点**，让模型的置信度预测决定点的质量

2. **空间均匀化采样 (Spatial Binning - 全图无掩码约束)：**
   * **问题：** 如果匹配点全部挤在图像中心的几根主血管上，解出来的放射矩阵对边缘区域会非常不准
   * **操作：** 从 LoFTR 的输出中，基于置信度进行**全图均匀采样**
   * **网格约束：**
     - 将 $512 \times 512$ 的图像划分为 $8 \times 8 = 64$ 个网格分区（**全图范围**）
     - 在每个分区中只选取置信度最高的前 5 个匹配点
     - 最多保留 $8 \times 8 \times 5 = 320$ 个匹配点
     - **不使用掩码过滤**：让模型的置信度预测决定点的质量
   * **理由：**
     - 强制匹配点在空间上均匀分布
     - 极大提高放射变换矩阵求解的稳定性和精度
     - 防止模型退化为单位矩阵 $I$
     - 端到端学习：模型自动学习给背景区域低置信度

3. **RANSAC 求解放射矩阵：**
   * 使用 **RANSAC** 算法剔除误匹配，并求解放射矩阵 $M$：
   $$\min_M \sum_k ||M \cdot [x_k, y_k, 1]^T - [x'_k, y'_k, 1]^T||^2$$

4. **大尺度转换处理：** 由于课程学习的训练策略，模型即使在 $\lambda = 0$ 的情况下，也能在极大旋转角度（±90°）下依靠学到的血管拓扑特征建立准确匹配。

5. **重采样与评估：**
   * 利用 $M$ 对 $I^B$ 进行 Warp，完成配准
   * **评估时使用有效区域过滤**：计算 MSE 等指标时调用 `filter_valid_area()`，只在眼底圆形有效区域内计算，避免背景噪声影响评估结果

---

## 第五阶段：数据增强策略 (Data Augmentation)

在训练过程中，对图像进行放射变换的同时，**同步对血管掩码进行相同的几何变换**，以保证掩码与变换后的图像对齐（用于损失函数权重计算）。

### 增强策略

1. **同步放射变换：**
   * 生成随机放射矩阵 $T$（旋转 $\pm 90^\circ$，平移 $\pm 20\%$，缩放 $0.8 \sim 1.2$）
   * **随机翻转：** 10% 概率水平/垂直翻转（可与旋转组合）
   * 同时对 $I^B$ 和 $M^{vessel}_B$ 应用相同的变换得到 $I^B_{warped}$ 和 $M^{vessel}_{B,warped}$
   * **插值方法：** 图像使用双线性插值，掩码使用最近邻插值（保持二值性）
   * **不对图像进行掩码过滤**：保留完整图像（含背景），让模型端到端学习

2. **血管结构保持增强：**
   * 亮度、对比度调整（仅影响图像，不影响掩码）
   * 轻微高斯噪声（$\sigma < 0.02$，避免破坏血管边缘）
   * **禁止使用：** Elastic Deformation（会破坏掩码与图像的几何对应关系）

3. **掩码使用说明：**
   * 血管掩码 $M^{vessel}$ 仅用于：
     - **损失函数权重**：粗级/精级损失中的 `torch.clamp(mask, min=0.1)` 梯度保底
     - **注意力偏置**（训练时）：$\lambda \cdot vessel\_bias$ 软引导
   * **不用于**：
     - ❌ 图像预过滤（不将背景置零）
     - ❌ 匹配点硬过滤（不根据掩码丢弃点）
     - ❌ 空间均匀化约束（不限制网格范围）

---

## 第六阶段：实现细节与模块设计 (Implementation Details)

### 数据加载器修改 (DataLoader)

在 `cf_260116_expand.py` 中添加掩码返回逻辑：

```python
def __getitem__(self, idx):
    # 原有逻辑：加载 I^A, I^B
    img_A = load_image(self.cf_paths[idx])
    img_B = load_image(self.oct_fa_paths[idx])
    
    # 新增：加载血管掩码
    mask_vessel = load_mask(self.vessel_mask_paths[idx])  # 二值图，0/1 或 0/255
    mask_vessel = (mask_vessel > 0.5).astype(np.float32)  # 归一化为 0/1
    
    # 生成随机放射变换（旋转范围扩大到 ±90°）
    T = generate_random_affine(rotation=(-90, 90), translation=(-0.2, 0.2), scale=(0.8, 1.2))
    
    # 随机翻转（小概率）
    flip_h = np.random.rand() < 0.1  # 10% 概率水平翻转
    flip_v = np.random.rand() < 0.1  # 10% 概率垂直翻转
    
    # 同步变换图像和掩码（图像不做过滤，保留完整）
    img_B_warped = cv2.warpAffine(img_B, T[:2], (W, H), flags=cv2.INTER_LINEAR)
    mask_B_warped = cv2.warpAffine(mask_vessel, T[:2], (W, H), flags=cv2.INTER_NEAREST)
    
    # 应用翻转（图像和掩码同步）
    if flip_h:
        img_B_warped = cv2.flip(img_B_warped, 1)  # 水平翻转
        mask_B_warped = cv2.flip(mask_B_warped, 1)
    if flip_v:
        img_B_warped = cv2.flip(img_B_warped, 0)  # 垂直翻转
        mask_B_warped = cv2.flip(mask_B_warped, 0)
    
    # 构建 3x3 单应矩阵（包含翻转）
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = T
    if flip_h:
        H_flip_h = np.array([[-1, 0, W-1], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        H = H_flip_h @ H
    if flip_v:
        H_flip_v = np.array([[1, 0, 0], [0, -1, H-1], [0, 0, 1]], dtype=np.float32)
        H = H_flip_v @ H
    
    return {
        'image0': img_A / 255.0,              # 归一化到 [0, 1]，不做掩码过滤
        'image1': img_B_warped / 255.0,       # 保留完整图像（含背景）
        'image1_origin': img_B / 255.0,       # 原始图像（用于MSE计算）
        'vessel_mask0': mask_vessel,          # 血管掩码（用于损失权重）
        'vessel_mask1': mask_B_warped,        # 变换后的血管掩码
        'T_0to1': H,                          # 真值变换矩阵（3x3）
    }
```

### LoFTR 模型修改要点

1. **Backbone 输出修改：**
   * **❌ 不在 `ResNetFPN` 中拼接掩码作为输入通道**
   * **✅ Backbone 只提取纯净的图像特征**
   * 掩码单独传递给 Transformer 和损失函数

2. **Transformer 模块修改：**
   * 在 `LinearAttention` 的 `forward()` 中添加 `vessel_mask` 和 `lambda_weight` 参数
   * **加性偏置实现（不在 Backbone 拼接掩码）：**
   ```python
   # 相似度计算 (Additive Bias)
   # 在 Transformer 的注意力计算中，不要在 Backbone 拼接 mask
   # 而是在 Transformer 的 Attention Matrix 中引入偏置
   scores = compute_similarity(feat_A, feat_B)  # 基于纯净图像特征
   
   # 加性偏置引导，不截断梯度
   if lambda_weight > 0 and vessel_mask_pair is not None:
       # vessel_mask_pair: [B, N_A, N_B] = vessel_mask_A.unsqueeze(-1) * vessel_mask_B.unsqueeze(-2)
       scores = scores + lambda_weight * vessel_mask_pair
   
   attn = softmax(scores, dim=-1)
   ```

3. **损失函数修改：**
   * **梯度保底实现：**
   ```python
   # 损失计算 (Gradient Guard)
   # 粗级损失
   conf_loss = F.nll_loss(log_pt, gt_mask, reduction='none')
   
   # 梯度保底：给背景留 0.1 的权重
   # 使用 torch.clamp(mask, min=0.1) 作为损失权重系统
   loss_weight = torch.clamp(vessel_mask_pair, min=0.1)  # [B, N]
   loss_coarse = (conf_loss * loss_weight).sum() / (loss_weight.sum() + 1e-8)
   
   # 精级损失（同样使用梯度保底）
   fine_loss = L2_loss(pred, gt)
   window_weight = torch.clamp(window_vessel_ratio, min=0.1)
   loss_fine = (fine_loss * window_weight).mean()
   ```

### 训练超参数建议

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 8e-4 | AdamW 优化器，带 Cosine 衰减 |
| Batch Size | 4-8 | 取决于 GPU 显存（512×512 图像） |
| 训练轮数 | 100-200 epochs | 监控验证集匹配精度 |
| 最小轮数 | 70% of total epochs | 前70% epoch不触发早停 |
| 早停 patience | 8 | 验证损失连续8个epoch不改善则停止 |
| 粗级阈值 | 1.0 (网格单位) | 1/8 分辨率下的距离阈值 |
| 精级窗口 | $w = 5$ | 5×5 像素窗口 |
| 课程学习 λ | 0 → 0.2 → 0.05 | 阶段1(0-20%)→阶段2(20%-70%)→阶段3(70%+) |
| 验证/测试 λ | 0 | 推理时关闭血管掩码偏置 |
| 梯度保底 ε | 0.1 | 背景区域的最小损失权重 |

---

## 修改总结：引入血管掩码后的完整流程

### 数据流 (Data Pipeline)

1. **输入：** 
   - 固定图 $I^A$ (CF)
   - 待配准图 $I^B$ (OCT/FA)
   - 血管分割掩码 $M^{vessel}$（二值图，表示所有模态共同的血管结构）

2. **数据增强：**
   - 对 $I^B$ 和 $M^{vessel}$ 同步应用随机放射变换 $T$
   - 掩码使用最近邻插值保持二值性
   - **不对图像进行掩码过滤，保留完整图像（含背景）**

3. **特征提取：**
   - 双流 Backbone 提取粗/精两级特征
   - **❌ 不在 Backbone 中拼接或融合掩码信息**
   - **✅ 掩码单独传递给 Transformer**
   - 掩码下采样到 1/8 分辨率用于注意力偏置

4. **解剖偏置注意力：**
   - 在 Transformer 的注意力分数矩阵上应用**加性偏置 (Additive Bias)**
   - 通过 $scores = scores + \lambda \cdot vessel\_bias$ 引导注意力向血管区域倾斜
   - **不截断梯度**，非血管区域保留微弱响应，确保梯度连续性

5. **损失计算：**
   - **粗级：** 对所有有效点对计算损失，使用 `clamp(mask, min=0.1)` 梯度保底
   - **精级：** 对所有窗口计算损失，使用软权重而非硬筛选

6. **推理与配准：**
   - 输入完整的待配准图像对（含背景，$\lambda=0$）
   - LoFTR 输出匹配点对及置信度
   - **❌ 不使用掩码过滤匹配点**，让置信度决定点的质量
   - 全图 8×8 空间均匀化采样（无掩码约束）
   - RANSAC 求解放射矩阵
   - 应用变换完成配准
   - **评估时使用有效区域过滤**计算指标

### 关键创新点

| 模块 | 原始 LoFTR | 引入掩码后的改进（端到端学习方案） |
|------|-----------|----------------|
| **输入** | 仅图像对 | 图像对 + 血管掩码（掩码不拼接到Backbone） |
| **图像预处理** | 无 | **保留完整图像（含背景）**，不做过滤 |
| **特征提取** | ResNet-18 | **双流 ResNet-18**，不拼接掩码 |
| **注意力机制** | 全局无约束匹配 | **加性偏置 (+λ·vessel_bias)**，课程学习动态调整 |
| **损失函数** | 均匀权重 | **梯度保底 (torch.clamp(mask, min=0.1))**，保证早期收敛 |
| **训练策略** | 单一阶段 | **三阶段课程学习**：λ=0→0.2→0.05 |
| **推理配置** | - | **λ=0，关闭掩码偏置**，避免依赖 |
| **匹配点过滤** | 置信度阈值 | **不使用掩码硬过滤**，让置信度决定 |
| **RANSAC** | 均匀采样 | **全图 8×8 空间均匀化**，无掩码约束 |
| **评估** | 全图计算 | **filter_valid_area()**，只在有效区域计算指标 |

### 预期效果

通过引入"端到端学习"策略（参考 RoMa 方案），模型将能够：

1. **处理极大角度旋转（±90°）和翻转：** 保留全图背景有助于 Transformer 建立正确的全局位置编码。血管拓扑结构在旋转下保持不变。
2. **处理大位置偏移：** 即使点对落在背景上，微弱的梯度保底（ε=0.1）也能推动模型学习大尺度的旋转信息。
3. **避免模型退化：** 全图 8×8 空间均匀化（无掩码约束）强制匹配点散开，防止解算的放射矩阵退化为单位矩阵 $I$。
4. **提升关键点质量：** 通过三阶段课程学习，模型从全图感知逐步过渡到血管区域关注，在训练后期自动聚焦于血管交叉点和分叉点。
5. **加速收敛与稳定训练：** 
   - 阶段1（λ=0）：快速学习大尺度几何变换
   - 阶段2（λ=0.2）：引入弱血管引导，平衡全局和局部
   - 阶段3（λ=0.05）：精细调优，避免过度依赖掩码
   - 前70% epoch保护期：确保模型充分学习基础特征后再启用早停
6. **推理时的鲁棒性：** 测试时设置 λ=0，确保模型不依赖掩码，具有更好的泛化能力。
7. **端到端学习的优势：** 
   - 模型自动学习血管区域的重要性（通过注意力偏置和损失权重）
   - 模型对背景区域的匹配给出低置信度（通过端到端学习）
   - 避免对人工规则的过度依赖，提升泛化能力

