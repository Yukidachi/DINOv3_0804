 # A 实验的具体实施方案：Cityscapes 固定低秩特征迁移

版本：2026-08-08  
适用范围：A0-A6，首轮只在 `MobileNetV2 + R-ASPP` 上执行  
依据：`知识蒸馏实验分析与后续实验方向.md`、`plan_markdown/Cityscapes知识蒸馏实验详单.md`、`plan_markdown/基线训练部分总结.md`

本文把 A0-A6 从“实验编号”细化为可执行的采样、参数拟合、投影、训练、验收和产物约定。不提供完整训练程序；实现时可以拆成 PCA 统计脚本、投影模块、无标签特征训练入口和冻结 backbone probe 入口。

---

## 1. 当前基线与 A 组要回答的问题

### 1.1 已锁定的基线事实

当前基线已经满足启动 A 组的条件：

| 实验 | `dev_local` mIoU | 解释 |
|---|---:|---|
| T0：DINOv3 ConvNeXt-T + R-ASPP，冻结 backbone | `0.681911 ± 0.001092` | 仅作教师冻结表征参考 |
| T1：head warm-up 后解冻最后 stage | `0.776024 ± 0.002086` | 后续 A/K 组的教师候选 |
| S2-F：随机 V2 backbone 冻结，只训 R-ASPP | `0.019125` | A 组 probe 下界 |
| S2-0：V2 scratch 端到端 CE | `0.495653 ± 0.005220` | V2 主监督基线 |
| S2-P：当前本地 ImageNette-10（方案中简称 ImageNette2-320）预训练后端到端 CE | `0.494730 ± 0.008502` | 资源受限的辅助基线，单独报告 |

T1 的三个 seed 中，`seed=3407` 的 `dev_local mIoU=0.778346` 最高。因此首轮固定使用：

```text
result/T1_DINOv3_RASPP/seed_3407/t1_dinov3_raspp_teacher.pth
```

锁定教师后必须保存 checkpoint SHA-256、配置、DINOv3 权重 SHA-256 和教师 shape audit。后续 A0-A6 不得根据单次结果重新选择教师，也不得查看 `test_local` 后替换教师。

### 1.2 A0-A6 的共同科学问题

A 组只研究“教师稠密特征怎样进入轻量学生”，暂不加入教师像素 logits、关系损失、GAN 或类别权重。每个编号只改变一个投影因素：

1. PCA 本身是否优于任意低秩投影；
2. `StandardScaler` 是否必要；
3. 固定投影是否比可学习投影稳定；
4. 把目标压到学生通道数是否比强迫学生拟合完整教师空间容易；
5. 学生侧是否需要一个可学习坐标变换。

---

## 2. 所有 A 组都必须固定的协议

### 2.1 数据与增强

- 只使用 `datasets/cityscapes/train_local.txt` 的 2530 张图拟合 PCA、统计量和训练特征目标。
- 数据清单组合 SHA-256 必须为：

  ```text
  033161572be28a6de295e0c5dfb62d83cd4d0a18b6039321347c58ab28b9d3c2
  ```

- 特征训练沿用基线：随机缩放 `[0.5, 2.0]`、随机裁剪 `512×1024`、水平翻转；教师和学生接收同一张增强后的图像。
- `dev_local` 只用于选择 checkpoint；评估使用原分辨率 `1024×2048`、单尺度、无翻转。
- A 组的特征预训练不使用语义标签计算 loss。数据加载器可以携带 target 以复用现有 Cityscapes 管线，但 target 不进入 PCA、投影或特征损失；不得按类别、像素比例或 mask 选择 PCA token。
- 训练 crop 全部为 ignore 时，沿用基线的重新采样规则；该规则只用于得到有效输入，不得形成类别均衡采样。

### 2.2 模型和特征 tap

所有模型采用 `output_stride=16`。V2 当前 shape audit 已锁定为：

| 层 | 模块 | 学生通道/形状（`512×1024` crop） | 教师对应层 | 教师通道/形状 |
|---|---|---|---|---|
| OS=4 | `backbone.3` | `C_s=24`，`[B,24,128,256]` | ConvNeXt stage 0 | `C_t=96`，`[B,96,128,256]` |
| OS=8 | `backbone.6` | `C_s=32`，`[B,32,64,128]` | ConvNeXt stage 1 | `C_t=192`，`[B,192,64,128]` |
| OS=16 | `backbone.17` | `C_s=320`，`[B,320,32,64]` | ConvNeXt `os16` stage 3 | `C_t=768`，`[B,768,32,64]` |

教师的 `os16_mid=[B,384,32,64]` 只保留作诊断，不进入 A0-A6 主损失；否则会引入第四个教师层和额外的层位变量。每次运行开始都要做 shape audit：

```text
输入 logits: [B,19,512,1024]
teacher/student 的三个 KD 层 H、W 完全相同
align_corners=False（仅 logits 评估需要上采样）
```

如果后续修改教师或学生的具体模块，必须生成新的 `feature_taps.json` 和新的实验编号。

### 2.3 初始化、预算和优化

- 学生 backbone 使用 `weights=None`，A0-A6 不从 S2-0 或 S2-P checkpoint 继续训练。
- 为保证对比公平，先用 `seed=42` 生成一份 `student_backbone_init.pth`，A0-A6 全部从这份相同初始化开始；正式三 seed 使用 `42/3407/260805` 各自固定的初始化。
- 特征预训练预算：40,000 optimizer steps；只优化学生 backbone，以及 A2/A5/A6 中明确存在的训练期 adapter。
- probe 预算：冻结学生 backbone 后训练同一个 19 类 R-ASPP head 40,000 optimizer steps。probe 的初始化、global batch、学习率和数据顺序对所有 A 组一致。
- global batch 固定为 8。服务器上使用 `batch_size_per_gpu × accumulation_steps × world_size=8`；不得把 local batch 误当 global batch。
- 优化器、poly 学习率、AMP、deterministic、worker 和退出顺序沿用服务器 S2-0 约定。首轮只使用已经验证过的 `SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, poly_power=0.9)`；不要因某个 A 编号单独调学习率。
- 特征损失在前 5% step 线性 warm-up，之后保持固定权重。首轮 `lambda_feat=1.0`。

---

## 3. PCA 统计的正式方案

### 3.1 为什么不能直接取前若干 token

Cityscapes 每张图包含大量 road、building、sky 等大面积区域。如果按 DataLoader 顺序把展平 token 累加到上限，OS=4 层可能只覆盖很少图片，PCA 学到的是少数场景和背景的方差。A 组需要比较子空间，而不是比较 DataLoader 顺序。

因此采用“全图覆盖优先 + 每图固定配额 + 空间分层选择”的无标签采样。它比随机 token reservoir 更容易审计，也不依赖 worker 数量、batch 顺序或目录遍历顺序。

### 3.2 PCA 输入视图

PCA 统计使用独立的、确定性的 PCA view，不使用随机增强：

1. 按 `train_local.txt` 的字典序读取图像；
2. 将原始 `1024×2048` RGB 图像双线性缩放到 `512×1024`，不裁剪、不翻转；
3. 使用与训练相同的 ImageNet mean/std 归一化；
4. 教师 `eval()`、`inference_mode()`，不计算梯度；
5. 记录该 view 的完整 transform signature。

这个 view 保留每张 Cityscapes 图的完整城市构图，同时与训练 crop 的张量大小一致。PCA view 只影响统计量，不改变正式 A 组的随机 crop 训练协议。

### 3.3 每层 token 数和每图配额

首轮每层固定抽取 `N=200,000` 个 token，三个层分别拟合，不能把三层拼在一起。2530 张图片的配额严格按下式分配：

```text
base = floor(200000 / 2530) = 79
remainder = 200000 - 2530 × 79 = 330
```

- 每张图片先得到 79 个 token；
- 用 `SHA256(seed / relative_image_path / "extra_quota")` 排序，最小的 330 张图片各增加 1 个 token；
- 因此 2200 张图贡献 79 个 token、330 张图贡献 80 个 token，总数严格为 200,000；
- 每图上限 128 的约束自然满足；
- 三个层使用相同的图片配额，减少层间“抽到了不同图片”这一额外变量，但每层的空间位置独立生成。

若未来更改 `N`，必须重新生成 manifest，重新计算 `base/remainder`，并创建新的实验编号；不能在旧 PCA manifest 上追加 token。

### 3.4 空间位置的确定性选择

对每个层的特征图 `H_l×W_l` 单独选位置。为避免 79/80 个点集中在一小块区域，将特征图划分为 `8×8` 个空间格子，按 token slot 循环访问格子：

```text
cell_id = token_slot mod 64
cell_row = cell_id // 8
cell_col = cell_id % 8
```

在当前三层分辨率下，每个格子都有足够的空间位置。格子内的具体 `(row,col)` 由稳定哈希确定：

```text
key = seed / relative_image_path / layer_name / token_slot / retry_index
u = int(SHA256(key), 16)
local_position = u mod (cell_height × cell_width)
```

若生成的位置与同一图片同一层已经选中的位置重复，则递增 `retry_index` 后重算。最终 flatten index 为 `row×W_l+col`。必须保存每个 token 的 `path、layer、token_slot、row、col、flat_index`，不能只保存随机 seed。

这不是按标签做的类别均衡；它只是让每张图和每个空间区域都有机会进入统计。任何按 `trainIds` 过滤、按类别补采或按稀有类别加权的做法都属于后续 supervised PCA 扩展，不得混入 A0-A6 主线。

### 3.5 两遍式拟合，避免一次性保存大矩阵

每个层单独执行以下两遍流程，特征批次按 manifest 顺序读取：

**第一遍：拟合 StandardScaler**

1. 提取当前层全部固定位置 token，按 `[token, channel]` 组织；
2. 分块大小固定为 4096 或 8192 token，使用 `float64` 累积均值和方差；
3. 调用 `StandardScaler.partial_fit`（或等价的在线 Welford 统计）；
4. 保存 `mean_、var_、scale_`。若某通道方差为 0，`scale_` 按 sklearn 约定置为 1，不得除以 0。

**第二遍：拟合 PCA**

1. 重新按完全相同的 manifest 提取 token；
2. 用第一遍的 `mean_、scale_` 标准化；
3. 用 `IncrementalPCA(n_components=d_l, batch_size=8192)` 逐块 `partial_fit`；
4. 关闭 whitening：`whiten=False`；
5. 保存 `components_、mean_、explained_variance_、explained_variance_ratio_、singular_values_、n_samples_seen_`。

`d_l` 首轮等于学生对应层通道数：

```text
d_os4  = 24
d_os8  = 32
d_os16 = 320
```

实现可以用一次性 memmap + `PCA(svd_solver="randomized", random_state=42)` 做交叉验证，但正式主结果必须固定一种算法和顺序。不能 A0 用 randomized PCA、A1 重新用另一个 PCA 对象，否则 A1 的差异不能解释为 Conv 等价性。

### 3.6 PCA 产物与哈希

每层至少保存：

```text
pca_manifest_<layer>.json
scaler_<layer>.npz
pca_<layer>.npz
feature_input_<layer>.sha256
pca_parameters_<layer>.sha256
```

manifest 必须包含：数据清单组合哈希、T1 checkpoint 哈希、PCA view、采样 seed、总 token 数、每图配额直方图、层名、`C_t/H/W`、所有 token 位置及 token 清单哈希。参数哈希应按固定字段顺序、dtype、shape 和 C-order 原始字节计算。

禁止使用 `dev_local` 或 `test_local` 拟合 scaler/PCA。可以用 `dev_local` 检查投影后 CKA 和 probe mIoU，但不能用它重新估计 PCA 参数。

---

## 4. 从 PCA 参数到 `1×1 Conv` 的精确转移

### 4.1 矩阵约定

对某一层，教师特征的单个空间位置写成列向量 `x∈R^{C_t}`。设：

- `mu_s = scaler.mean_`，形状 `[C_t]`；
- `sigma_s = scaler.scale_`，形状 `[C_t]`；
- `mu_p = pca.mean_`，形状 `[C_t]`，注意它是在 StandardScaler 输出空间中的均值；
- `V = pca.components_`，形状 `[d_l,C_t]`，第 `o` 行是第 `o` 个主成分。

sklearn 的 `StandardScaler + PCA.transform` 为：

\[
y = \left((x-\mu_s)\oslash\sigma_s-\mu_p\right)V^T,
\]

其中 `y∈R^{d_l}`，`⊘` 为逐通道除法。

### 4.2 融合为一个卷积层

用 `Conv2d(C_t, d_l, kernel_size=1, bias=True)` 逐位置执行上述变换。卷积权重和偏置为：

\[
W_{o,i}=\frac{V_{o,i}}{\sigma_{s,i}},
\]

\[
b_o=-\sum_i V_{o,i}\left(\frac{\mu_{s,i}}{\sigma_{s,i}}+\mu_{p,i}\right).
\]

对应的实现顺序应是：

```python
W = pca.components_ / scaler.scale_[None, :]       # [d_l, C_t]
b = -((scaler.mean_ / scaler.scale_) + pca.mean_) @ pca.components_.T
conv.weight.copy_(torch.from_numpy(W[:, :, None, None]).float())
conv.bias.copy_(torch.from_numpy(b).float())
```

这里 `pca.components_` 不能转置后再当作 `[C_t,d_l]` 使用；`Conv2d.weight` 的第一维必须是输出通道。`pca.mean_` 是标准化后的 PCA 中心，不能误用原始特征均值 `scaler.mean_`。

### 4.3 A4 和 A3 的融合公式

A4 不使用 StandardScaler。此时只有 `pca.mean_` 和 `components_`：

\[
W=V,\qquad b=-\mu_pV^T.
\]

A3 使用标准化后的随机行正交矩阵 `Q∈R^{d_l×C_t}`：

\[
y=((x-\mu_s)\oslash\sigma_s)Q^T,
\]

\[
W_{o,i}=Q_{o,i}/\sigma_{s,i},\qquad
b_o=-\sum_i Q_{o,i}\mu_{s,i}/\sigma_{s,i}.
\]

`Q` 的生成方式固定为：以 `seed=42+layer_id` 生成高斯矩阵 `[C_t,d_l]`，QR 分解后取 `Q_full.T[:d_l]`，得到行正交矩阵；保存 `Q^TQ` 的误差，要求 `||QQ^T-I||_F≤1e-5`。A3 不额外减去 PCA 均值。

### 4.4 数值等价性验收（A1 必做）

正式训练前必须执行 CPU `float32` 等价性测试：

1. 为每层构造固定随机输入 `[2,C_t,32,64]`；OS=4/8 可用对应 H/W，或统一使用任意合法 H/W；
2. 分别计算 numpy/sklearn 路径和融合 Conv 路径；
3. 比较输出的 `max_abs_error、mean_abs_error、relative_l2_error`；
4. 再用真实 T1 特征做一次比较。

验收阈值：

```text
max_abs_error <= 1e-5
relative_l2_error <= 1e-6
```

若未通过，先检查 `components_` 方向、NCHW/`[B,N,C]` 展平顺序、偏置公式、`float64→float32` 转换和 scaler 的零方差处理；不得直接开始 A1 训练。A1 的固定 Conv 只需保存 `weight/bias`，但仍要保留原始 scaler/PCA 参数和哈希以便审计。

### 4.5 状态、梯度与推理边界

- A0/A1/A3/A4 的教师投影固定：`requires_grad=False`，在 optimizer 中不出现；
- A2 的 Conv 从 A1 权重初始化后允许训练，adapter 学习率为学生主学习率的 `0.1×`，其余优化器设置不变；
- A5/A6 的学生侧 Conv 只在特征预训练阶段存在，初始化方式和学习率必须记录；
- probe、端到端 fine-tune、参数量、MACs、显存和延迟统计前，移除所有训练期 adapter，使用原始学生 backbone + R-ASPP；
- 如果某个 adapter 的输出会进入 R-ASPP，则必须把“保留 adapter 的模型”和“移除 adapter 的学生模型”分开命名，不能把训练期参数量冒充部署参数量。

---

## 5. 特征损失和训练流程

### 5.1 三层对齐损失

对每个训练 crop，冻结 T1，在同一几何视图上得到教师特征 `t_l`；学生得到 `s_l`。教师投影或学生 adapter 输出记为 `p_l(t_l)`、`a_l(s_l)`。首轮损失为：

\[
L_{feat}=\frac{1}{3}\sum_{l\in\{4,8,16\}}
\operatorname{mean}\left(s_l^{*}-t_l^{*}\right)^2,
\]

其中 `s_l*` 和 `t_l*` 是该 A 编号定义的对齐张量。mean 必须同时除以 batch、channel、height、width，不能先对高分辨率层求和再与低分辨率层平均。

教师张量使用 `detach()`；不使用 teacher logits，不使用 `trainIds` 加权。日志必须分别记录三个层的 loss 和梯度范数：

```text
feat_loss_os4 / feat_loss_os8 / feat_loss_os16
student_grad_l2_os4 / student_grad_l2_os8 / student_grad_l2_os16
```

如果辅助项梯度长期大于 CE 参考梯度的 2 倍，先停止该运行，检查 loss reduction 和梯度累积；不能靠改实验编号掩盖尺度问题。

### 5.2 A 组的两个训练阶段

**阶段 A-pretrain：无标签稠密表征训练**

1. 固定 T1 为 `eval()`，所有参数 `requires_grad=False`；
2. 学生从统一 scratch 初始化开始；
3. 读取随机 crop，前向得到 OS=4/8/16 学生特征；
4. 计算该编号的固定/可学习投影和 `L_feat`；
5. 训练 40,000 optimizer steps，保存每 5,000 steps 的特征 loss、梯度和权重；
6. 按 `dev_local` 的冻结 head probe mIoU 选出 A-pretrain checkpoint，不根据训练 loss 单独选模型。

**阶段 A-probe：冻结 backbone 的统一分割 probe**

1. 移除 A2/A5/A6 的训练期 adapter；
2. 冻结学生 backbone 参数和 BatchNorm 运行统计，使用同一个 R-ASPP head 配置（`in_channels=1280`、19 类、`inter_channels=256`、dropout=0.1）；
3. 只训练 R-ASPP head 40,000 optimizer steps；
4. 每 5,000 steps 在 `dev_local` 评估，按 mIoU 选 checkpoint；
5. 输出每类 IoU、small-object mIoU、boundary F1、CKA 和投影残差。

probe 必须与 S2-F 共享评估实现，才能判断 A 组的提升来自 backbone 表征而不是 head 代码差异。A-probe 与 S2-0 的端到端 CE 结果不能混为同一训练预算。

### 5.3 可选的统一 fine-tune

A0、A5、A6 以及 seed=42 的领先方案通过 probe 后，再从其 probe checkpoint 继续端到端 CE fine-tune 80,000 optimizer steps。fine-tune 时：

- 不再计算 `L_feat`，除非另建 K 组编号；
- 使用与 S2-0 完全相同的 SGD/poly、增强和 global batch；
- 从 `dev_local` 选择 checkpoint；
- 只在所有候选冻结后统一查看 `test_local`。

这样可以区分“固定 backbone 的稠密表征质量”和“该表征经过监督适配后的最终分割质量”。

---

## 6. A0-A6 的逐项实施定义

下表中 `T→S` 表示教师特征先变换到学生对应通道数，`S→T` 表示学生侧变换到教师通道数。未写出的条件全部与第 2 节相同。

| 编号 | 教师侧变换 | 学生侧变换 | 是否可训练 | 首轮目的 |
|---|---|---|---|---|
| A0 | `StandardScaler + PCA`，输出 `[24,32,320]` | 无 | 固定 | PCA-T→S 的函数基线 |
| A1 | 将 A0 精确融合为三个固定 `1×1 Conv` | 无 | 固定 | 验证 PCA 与 Conv 的数值等价性 |
| A2 | A1 的 PCA-Conv 初始化 | 无 | PCA-Conv 可训练，LR=`0.1×` | 固定目标是否是关键正则 |
| A3 | `StandardScaler +` 固定随机行正交投影 | 无 | 固定 | 主方向是否真的比任意稳定低秩子空间重要 |
| A4 | 不做 StandardScaler，直接 PCA | 无 | 固定 | 尺度校准的独立贡献 |
| A5 | 固定 PCA 到学生通道数 | 学生 `C_s→C_s` 的 `1×1 Conv`，identity 初始化 | 学生 Conv 可训练，LR=`0.1×` | 学生坐标自由度是否有益 |
| A6 | `StandardScaler` 后保留完整教师空间 `[96,192,768]` | 学生 `C_s→C_t` 的 `1×1 Conv` | 学生 Conv 可训练，LR=`0.1×` | 完整教师空间是否对小学生过难 |

### A0：显式的固定 PCA-T→S

**输入/输出**

```text
teacher os4  [B,96,128,256]  -> [B,24,128,256]
teacher os8  [B,192,64,128]  -> [B,32,64,128]
teacher os16 [B,768,32,64]   -> [B,320,32,64]
```

实现时可以用逐通道标准化、减去 PCA mean、矩阵乘法，或用一个不参与优化的投影模块；推荐先保留显式路径作为 A1 的参考实现。学生直接与投影后的教师张量做 MSE。A0 是后续所有“固定 PCA 是否有效”判断的坐标基准。

**验收**

- 三层投影输出 shape 与学生完全一致；
- PCA 参数和 adapter 不出现在 optimizer；
- 训练前后 PCA 参数 hash 不变；
- 40k pretrain 后进行 40k probe，记录 `A0_probe_mIoU`、各层残差和 CKA。

### A1：把 A0 融合成固定 `1×1 Conv`

A1 不重新拟合 PCA、不重新抽样、不改变学生初始化。唯一变化是把 A0 的函数改写为三个 `Conv2d(C_t,d_l,1)`。先完成第 4.4 节的等价性测试，再开始 A1 的特征预训练。

**预期关系**

```text
A1 与 A0 的投影输出差异在数值误差范围内
A1 与 A0 的 probe mIoU 差异应小于 seed 波动
```

如果 A1 与 A0 明显不同，优先判定为实现错误，不得把它解释成“卷积比 PCA 更好/更差”。A1 只跑 seed=42 即可；它的主要产物是等价性报告和固定 Conv 权重 hash。

### A2：PCA 初始化后允许 adapter 移动

A2 从 A1 的三个 Conv 权重和偏置开始，唯一放开其梯度。PCA-Conv 采用独立 optimizer param group：

```text
student backbone: lr = 1.0 × base_lr
PCA adapter:      lr = 0.1 × base_lr
momentum、weight_decay、poly schedule 与主组一致
```

A2 的 adapter 位于教师目标侧；如果直接让它和学生一起最小化 MSE，`W,b→0` 会产生“教师目标和学生同时变零”的退化解。因此 A2 必须加入预注册的锚定项，而不能裸训：

\[
L_{A2}=L_{feat}+0.01\left(
\frac{\|W-W_0\|_F^2}{\|W_0\|_F^2+\epsilon}+
\frac{\|b-b_0\|_2^2}{\|b_0\|_2^2+\epsilon}
\right),
\]

其中 `W_0,b_0` 是 A1 的初始 PCA-Conv 参数，`epsilon=1e-12`。`0.01` 在 A0-A6 中固定，不为 A2 单独调参；若要研究锚定强度，另建 A2.x。每个 eval 点还要记录教师投影输出 RMS、`W/W_0` 范数比和最小奇异值；若输出 RMS 低于初始值的 0.5 倍或高于 2 倍，停止该运行并标记为退化。

每次评估保存 adapter 的权重变化量：

```text
||W_t-W_0||_F / ||W_0||_F
||b_t-b_0||_2
```

如果 adapter 很快偏离 PCA、feature loss 降低但 probe mIoU 下降，说明固定教师坐标可能提供了有效锚点；如果 adapter 变化很小且 A2≈A0，说明可学习自由度没有实际贡献。

### A3：固定随机正交低秩投影

A3 与 A0 保留相同的 StandardScaler、相同输出通道和相同训练预算，只将每层 PCA `V` 替换为固定随机行正交矩阵 `Q`。随机矩阵只在 seed=42 首次生成并保存；训练中不更新。

**必须报告**

- `||QQ^T-I||_F`；
- 投影后每层均值、方差和有效秩；
- A3-A0 的 probe mIoU、small-object mIoU 和 boundary F1 差值。

A3 接近 A0 时，结论只能写成“固定低秩和尺度校准可能是主要因素”，不能写成“PCA 主方向没有作用”；还需要结合 A4 和残差率判断。

### A4：取消 StandardScaler 的 PCA

A4 直接在原始教师特征上拟合 PCA：

```text
PCA.fit(raw_teacher_tokens)
PCA.transform(raw_teacher_feature)
```

不读取 A0 的 scaler，不复用 A0 的 components。每层仍使用 `d_l=[24,32,320]`，每层仍用同一份 200k-token 位置 manifest。融合 Conv 时使用第 4.3 节的无 scaler 公式。

A4 的重点不是只比较累计解释方差，而是检查大方差通道是否挤压 OS=4/8 的细粒度信息。必须另外报告每层投影输出的 channel std、feature loss 分层值和稀有类别/边界指标。

### A5：教师固定 PCA，学生侧增加可学习坐标变换

A5 的教师目标与 A0 完全相同。学生每层增加 `Conv2d(C_s,C_s,1)`：

- weight 初始化为 identity；
- bias 初始化为 0；
- 只在 A-pretrain 阶段训练；
- 学习率为 base LR 的 0.1 倍；
- probe 前删除该 Conv，使用原始学生 backbone 的 OS=4/8/16 特征。

这样 A5 测量的是“学生是否需要先学习一个坐标变换才能接近固定教师子空间”，而不是把额外 Conv 的部署开销计入学生。必须保存删除 adapter 前后的 backbone 参数 hash，确认 adapter 没有被错误地并入部署模型。

### A6：学生拟合完整的标准化教师空间

A6 不做 PCA。先按每层独立的 `StandardScaler` 标准化教师特征，目标维度为 `[96,192,768]`；学生侧增加：

```text
os4:  Conv2d(24, 96, 1)
os8:  Conv2d(32, 192, 1)
os16: Conv2d(320,768, 1)
```

由于输入输出维度不同时不能使用 identity 初始化，统一使用固定 seed 生成的正交初始化（矩阵按 `nn.init.orthogonal_`，bias=0），并使用 `0.1×` adapter LR。A6 的 adapter 训练结束后删除，再进入 probe。

A6 如果明显低于 A0/A5，只能说明在当前训练预算下完整教师坐标更难迁移；不能单独证明教师的高维信息没有价值。应结合 A6 的 adapter 梯度、loss 曲线、输出秩和学生 backbone 的 CKA 一起解释。

---

## 7. A 组筛选、复跑与统计

### 7.1 首轮 seed=42 筛选

按以下顺序执行，避免先跑大量重复实验：

1. PCA manifest、Scaler/PCA 参数和 A1 等价性测试；
2. A0、A1、A2、A3、A4、A5、A6 各跑一次 seed=42；
3. 每个编号都执行 40k pretrain + 40k probe；
4. 以 `dev_local probe mIoU` 为主要排序，检查 boundary F1、small-object mIoU 和训练稳定性；
5. A0、A5、A6 无论单次排名如何都扩展三 seed；另选一个单次领先方案扩展三 seed。

A1 的三 seed 不是优先算力；它首先是函数等价性验收。如果 A1 等价性通过且 A1/A0 结果相近，正式表格可以将 A1 作为 A0 的实现复现列单独列出。

### 7.2 “领先”不能只按单次最高 mIoU 定义

单 seed 的差异如果小于当前 S2-0 的 seed 标准差 `0.005220`，先描述为“性能相近”。扩展三 seed 后报告：

- `mean ± sample std`；
- 相对 A0 的成对 seed 差值；
- `dev_local` 逐图 bootstrap 95% CI；
- 19 类 IoU、small-object mIoU、boundary F1；
- 三层 CKA、有效秩和 PCA 残差率。

不要在看到 `test_local` 后修改 A 组候选、层位、PCA token 数、adapter 学习率或初始化。

### 7.3 统一 fine-tune 和最终 test

通过 probe 筛选的 A0、A5、A6 和领先组，再按第 5.3 节做 80k 端到端 CE fine-tune。所有 A 组候选冻结后，预注册最终 checkpoint 清单，才可在同一批次评估：

- T1 `seed=3407` 教师；
- S2-0 scratch 基线；
- 纳入报告的 S2-P；
- A 组最终候选。

`test_local` 结果只作为最终泛化报告，不参与任何选择；Cityscapes 官方 test 没有可用于本地 mIoU 的真值。

---

## 8. 必须做的诊断和单元测试

### 8.1 PCA/Conv 相关测试

- 采样 manifest 总 token 数严格为 200,000；每张图片 79 或 80 个；路径无重复；三层位置均在合法范围；
- 相同 seed、相同数据和教师 checkpoint 生成相同 manifest hash；改变 seed 必须生成不同 hash；
- 每层输入矩阵 shape 为 `[200000,C_t]`，无 NaN/Inf；
- scaler 的 `scale_` 无 0、NaN；PCA `components_` shape 为 `[d_l,C_t]`；
- A0/A1/A3/A4 的固定参数在训练前后 hash 不变；
- A1 通过随机张量和真实教师特征两次等价性测试；
- A5/A6 删除 adapter 后，probe 的 forward shape 与 S2-0 完全一致。

### 8.2 训练相关测试

- 单 batch smoke：教师无梯度、学生有梯度、三层 loss finite；
- 记录第一步各层 student gradient L2，确认 OS=4 不因像素数较多而独占总梯度；
- global batch、累计步数和 scheduler step 逐项写入 JSON；
- resume 后 optimizer step、LR、PCA/adapter hash 与中断前一致；
- 服务器 DDP 中只有 rank 0 写 checkpoint，所有 rank collective 顺序一致；
- 正常退出顺序沿用“关闭 worker → CUDA synchronize → 释放 DDP/optimizer → barrier → destroy process group”。

### 8.3 需要重点观察的曲线

每 500 step 或每个 eval 点保存：

```text
feat_loss_total
feat_loss_os4 / os8 / os16
gradient_l2_os4 / os8 / os16
student_feature_mean/std
adapter_weight_delta（A2/A5/A6）
dev_probe_mIoU、boundary F1、small-object mIoU
```

若 loss 很低但 probe mIoU 接近 S2-F，优先检查 student feature tap 是否错位、BatchNorm 是否在 probe 期间更新、adapter 是否被错误移除或 teacher/student 是否使用了不同 crop。

---

## 9. 结果目录和文件约定

建议 A 组按如下结构保存，避免覆盖 S2-0、T1 或后续 K 组结果：

```text
result/A_MobileNetV2_RASPP/
  pca_shared/
    sampling_manifest.json
    scaler_os4.npz / scaler_os8.npz / scaler_os16.npz
    pca_os4.npz / pca_os8.npz / pca_os16.npz
    pca_parameters_sha256.json
  A0/seed_42/
  A1/seed_42/
  A2/seed_42/
  A3/seed_42/
  A4/seed_42/
  A5/seed_42/
  A6/seed_42/
```

每次运行至少写入：

- `config.json`：实验编号、seed、数据/教师/PCA hash、crop、global batch、总 step、optimizer、AMP 和 deterministic；
- `feature_taps.json`：teacher/student 模块、OS、通道、H×W；
- `training_history.json`：各层 loss、梯度、LR、adapter 变化量；
- `best_probe_checkpoint.pth` 及 SHA-256；
- `dev_metrics.json`、19×19 confusion matrix、逐图 confusion；
- `projection_equivalence.json`（A1 必须存在）；
- `efficiency.json`：移除 adapter 后的参数量、MACs、峰值显存和 batch=1 延迟。

PCA 参数可以在 A0-A6 之间共享，但必须在每个实验的 `config.json` 中记录共享参数的 hash。改变教师 checkpoint、PCA view、token 数、采样 seed、层位或通道目标时，禁止复用旧目录。

---

## 10. 推荐的实际执行命令顺序（仅示意）

不要求命令名称与下列示意完全相同，但执行顺序和验收条件必须保持：

```text
1. verify locked Cityscapes splits
2. audit T1/V2 feature shapes at 512×1024
3. build deterministic 200k-token manifest for os4/os8/os16
4. fit scaler and PCA in two passes; save hashes
5. run A1 random/real-feature equivalence test
6. run A0-A6 with seed=42: 40k feature pretrain + 40k frozen-head probe
7. expand A0/A5/A6/A_best to seeds 42/3407/260805
8. run optional 80k CE fine-tune for the registered candidates
9. freeze candidate list and only then evaluate test_local
```

第一轮完成标准不是“某个 mIoU 最高”，而是：PCA 采样可复现、A1 数值等价、所有 A 组使用相同预算、adapter 边界清晰、probe 和效率指标可复核。达到这些条件后，才进入 K0-K3 的 feature/logits 2×2 实验。
