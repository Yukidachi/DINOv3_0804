# D 组实验的具体实施方案：Cityscapes 分布知识蒸馏

版本：2026-08-22
适用范围：D0-D4，首轮只在 `MobileNetV2 + R-ASPP`、Cityscapes 本地划分上执行  
依据：`plan_markdown/K组实验的具体实施方案.md`、`plan_markdown/K组训练部分总结.md`、`plan_markdown/R组实验的具体实施方案.md`、`plan_markdown/R组训练部分总结.md`、`plan_markdown/Cityscapes知识蒸馏实验详单.md`

本文把 D0-D4 从“分布约束是否有增量”的研究问题细化为可直接实现的初始化、教师前向、分布特征、损失、训练、验收、统计和产物约定。D 组只研究 feature 分布约束，不改变 K/R 组已经锁定的教师、学生、数据、优化器、评估和留出集协议。

---

## 1. D 组要回答的问题

### 1.1 已锁定的前置事实

| 项目 | 已完成结果 | 对 D 组的意义 |
|---|---:|---|
| K1 | `0.52203 ± 0.00219` mIoU，三 seed | D0 的协议锚点；D0 seed=42 必须复现 K1 seed=42 |
| R2 `lambda=0.3` | `0.53275 ± 0.00601`，三 seed | 图内空间关系已显示稳定方向增益 |
| R3 | `0.52839 ± 0.00347`，三 seed | R1+R2 当前未显示主指标互补 |
| R5 `seed=42` | `0.573653` | 当前最强 scratch 组合候选，但尚未完成三 seed |
| K4 | `0.57049 ± 0.00582` | A0 分阶段初始化协议，不是 D 组 matched 基线 |

K1 的三 seed 统计、R2/R3 的三 seed 统计和 R5 的单 seed 结果已由各自训练总结核对；单次运行的指标仍以对应 `metrics.json` 为权威来源。D 组不得加载 R5、K4、A0-FT、R2 或其他训练完成的学生 checkpoint 作为初始化。

### 1.2 D 组共同科学问题

D 组回答“在硬标签 CE 和固定 A0 feature KD 已存在时，教师与学生表征的分布统计约束能否提供额外分割增量” ：

1. `D0`：D 组新入口能否严格复现 K1；
2. `D1`：教师/学生 feature 的一阶、二阶统计对齐是否提供超出 pointwise feature MSE 的信息；
3. `D2`：基于随机投影的全局分布距离是否提供增量；
4. `D3`：小型分布判别器作为辅助项时，是否比显式 CORAL/SWD 更有效；
5. `D4`：去除 feature MSE 后，对抗分布约束是否仍能提供可用监督，还是只会得到欠约束解。

D 组不回答：

- R2 关系矩阵与分布距离谁在所有任务上更优；
- R5 中 logits KD、R2 和 feature MSE 的联合最优权重；
- K4 A0 初始化与分布蒸馏的交互；
- PCA、层位、学生结构、增强、教师 checkpoint 或分割头的优劣；
- MMD、SWD、CORAL、GAN 的完整超参数搜索；
- `test_local` 查看后的任何再调参问题。

---

## 2. D0-D4 实验矩阵

首轮只选择一个显式分布距离：D2 固定使用 SWD，不在 D2 中同时实现 MMD 和 SWD。D3 只有在 D1、D2 均通过实现验收但没有形成有意义增量时才启动。

| 编号 | 硬标签 CE | A0 feature KD | CORAL | SWD | 对抗分布项 | 目的 |
|---|---|---|---|---|---|---|
| D0 | 是 | 是 | 否 | 否 | 否 | K1 受控复现锚点 |
| D1 | 是 | 是 | 是 | 否 | 否 | 测量均值/协方差分布增量 |
| D2 | 是 | 是 | 否 | 是 | 否 | 测量随机投影分布距离增量 |
| D3 | 是 | 是 | 否 | 否 | 是 | 小判别器辅助分布对齐 |
| D4 | 是 | 否 | 否 | 否 | 是 | adversarial-only 欠约束诊断 |

统一损失锚点为 K1：

```text
D0 = CE + feature
D1 = CE + feature + CORAL
D2 = CE + feature + SWD
D3 = CE + feature + adversarial
D4 = CE + adversarial
```

所有 D 组运行均从对应 seed 的 K shared scratch initialization 独立启动。D1-D4 不加载 D0、K1 或 R5 的 checkpoint；D0 的作用是验证入口，不是给后续实验提供已训练初始化。

首轮执行顺序固定为：

```text
D0(seed=42) -> D1(seed=42) -> D2(seed=42)
-> 若 D1/D2 均通过实现验收但均无有效增量，再运行 D3(seed=42)
-> D4(seed=42，仅作 adversarial-only 诊断)
-> 只扩展通过筛选的候选到 seed=3407/260805
```

D0 只需在 D 入口完成 `seed=42` 的受控等价性验证，不为了 D0 重复运行完整三 seed；其余 seed 的 D0 锚点直接引用对应 K1 结果。D1/D2/D3 只有在 `seed=42` 筛选后才允许扩展三 seed，D4 永远保持单 seed 诊断，除非另行注册新的因果问题。

---

## 3. 所有 D 运行必须一致的协议

### 3.1 数据、标签和增强

- `train_local=2530`、`dev_local=445`、`test_local=500`；组合清单 SHA-256 必须为：

  ```text
  033161572be28a6de295e0c5dfb62d83cd4d0a18b6039321347c58ab28b9d3c2
  ```

- 训练视图：随机缩放 `[0.5,2.0]`、随机裁剪 `512×1024`、水平翻转；教师和学生必须接收同一增强后的 tensor。
- 标签：`labelIds -> trainIds 0..18`，其余为 `ignore_index=255`。
- dev：原分辨率 `1024×2048`、单尺度、无水平翻转。
- 分布损失只在训练 crop 的有效 feature token 上计算；ignore 像素不能进入均值、协方差、投影或判别器 batch。
- 不加入 class-mix、copy-paste、伪标签、类别权重、Dice、Focal、辅助分割头或多尺度推理。

分布损失不得跨 optimizer step 缓存 feature，也不得把不同 step 的 token 拼接成一个统计 batch。当前 step 的 global DDP batch 可以用于统计；统计中的 token 数必须记录。

### 3.2 留出集规则

所有 D 运行的 `metrics.json` 必须写入：

```json
{"test_local_evaluated": false}
```

`test_local` 不得用于训练、checkpoint 选择、分布损失权重选择、候选筛选、seed 扩展或解释分布损失。只有数据、代码、教师、候选和协议全部冻结后，才允许进行一次最终留出评估。

### 3.3 教师

所有 D 组固定使用：

```text
result/T1_DINOv3_RASPP/seed_3407/t1_dinov3_raspp_teacher.pth
```

checkpoint SHA-256：

```text
73cb1d3161c746d1b4ea30918ec6a1f0de5e3a4952c000cf85ddf95f3ccaddeb
```

要求：

1. 使用 `dino_t1.load_teacher_for_distillation()` 加载，不得误用 T0 loader；
2. 调用 `freeze_for_distillation()`，教师始终 `eval()` 且所有参数 `requires_grad=False`；
3. 教师不进入 optimizer、不包 DDP，每个 rank 各自持有同一冻结副本；
4. 教师前向置于 `torch.no_grad()`；
5. 教师与学生使用同一增强图像；
6. 教师 feature target、统计 target 和判别器 real sample 都必须 detach；
7. D1-D4 不使用 R5 checkpoint 中缓存的 teacher output，必须从当前同一增强图像重新计算。

### 3.4 学生、输出步长和 feature tap

学生固定为 `MobileNetV2 + R-ASPP`、`output_stride=16`、`weights=None`：

| 层 | 学生 tap | 学生形状（`512×1024`） | 教师形状 |
|---|---|---|---|
| OS=4 | `backbone.3` | `[B,24,128,256]` | `[B,96,128,256]` |
| OS=8 | `backbone.6` | `[B,32,64,128]` | `[B,192,64,128]` |
| OS=16 | `backbone.17` | `[B,320,32,64]` | `[B,768,32,64]` |
| head 输入 | `backbone.18` | `[B,1280,32,64]` | 不用于 D 分布损失 |

D 组的 feature distribution source 使用 A0 固定 projection 后的 student-space features：

```text
teacher distribution source = P_l(f_t^l).detach()
student distribution source = f_s^l
```

这样教师和学生在每一层具有相同通道数，同时 D1/D2/D3 的 target 仍然是分布统计，不是额外的 pointwise MSE。A0 projection 只作为固定通道对齐，不在 D 组训练中更新。

### 3.5 共同 scratch 初始化

D0-D4 复用 K 组已经生成并审计过的共同初始化：

```text
result/K_MobileNetV2_RASPP_server/shared_init/seed_<seed>/student_init.pth
```

规则：

1. D0-D4 `seed=42` 使用同一个 K shared init state；
2. 扩展到 `seed=3407/260805` 时使用对应 seed 的 K shared init；
3. 不生成 D-specific initialization；
4. 不加载 K1/K2/K3/K4、R2/R3/R5、A0/A0-FT 或其他学生 checkpoint；
5. 同一 seed 的 D0-D4 step=0 `student_state_sha256` 必须一致；
6. 不同 seed 的初始化 hash 必须不同；
7. D 组的 feature distribution target 不能改变学生 step=0 state。

D 组不从 R5 继续训练。R5 是当前最强的联合损失候选，用于最终候选上下文和后续比较，不是 D 组的初始化或 matched 因果基线。D 组的主问题是分布约束相对 K1 anchor 的独立增量。

### 3.6 batch、DDP 和优化器

| 项目 | 固定值 |
|---|---|
| world size | 2 |
| per-GPU batch | 2 |
| accumulation | 2 |
| effective optimizer global batch | 8 |
| optimizer | `SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)` |
| scheduler | poly，power=`0.9`，min LR ratio=`0.01` |
| optimizer steps | `80,000` |
| AMP | 开启 |
| deterministic | 开启 |
| checkpoint 选择 | 只按 dev mIoU |
| eval 间隔 | 每 5,000 optimizer steps 附近的首个 epoch 边界 |

分布统计使用当前同步的物理 global micro-batch，固定为 `per_gpu_batch × world_size = 4`；不得把梯度累积后的 batch=8 拼成统计 batch，也不得跨 micro-batch 或 optimizer step 缓存 token。这样与 R1 的物理 batch 口径一致。实现必须记录统计使用的 `physical_distribution_batch_size=4`、`global_token_count` 和每层有效 token 数。

### 3.7 warm-up

复用 K/R 组辅助项 warm-up：

```text
warmup_steps = 4000
warmup(s) = min(1, s / 4000)
```

CE 从第一步起使用完整权重；feature、CORAL、SWD 和 adversarial 项只使用 warm-up 权重。warm-up 的 `s` 必须按 optimizer step 计算，不得按 micro-batch 计算。

---

## 4. D0 的 K1 feature anchor

D0-D3 保留 K1 的固定 A0 feature KD，D4 才移除该项。复用：

```text
result/A_MobileNetV2_RASPP_server/pca_shared/
```

加载 `scaler_os4/os8/os16.npz` 和 `pca_os4/os8/os16.npz`，使用 A0 的显式 `FixedPCAProjection`：

\[
\hat f_t^l=P_l(f_t^l),
\]

\[
L_{feat}=\frac{1}{3}\sum_{l\in\{4,8,16\}}
\operatorname{mean}_{B,C,H,W}(f_s^l-\hat f_t^l)^2.
\]

要求：

- 三层等权；每层先对 BCHW 全元素取 mean，再除以 3；
- feature loss 不读取标签 mask；
- projection 是固定 buffer，不在 optimizer；
- 教师 features 使用同一增强图像并 detach；
- PCA 参数记录中的 teacher hash、data manifest hash 必须匹配；
- 首轮固定 `lambda_feat=1.0`；
- D0 的首 batch 和最终指标必须与 K1 `seed=42` 在预注册容差内一致。

---

## 5. 分布特征的共同约定

### 5.1 固定 A0 投影后的有效 token

对每个层 `l∈{os4,os8,os16}`：

1. 将原始 valid mask 用 nearest-neighbor resize 到 feature 分辨率；
2. 对 teacher native feature 使用冻结 A0 projection 得到 `\hat f_t^l`；
3. student feature `f_s^l` 直接使用对应 tap；
4. 只收集 mask 有效的位置；
5. 按固定上限进行无放回 token 抽样，默认每层每个 physical global micro-batch 最多 `256` 个 token；
6. 抽样使用由 `seed` 和 optimizer step 派生的确定性 generator，并记录抽样 seed、有效 token 数和 token hash；
7. 教师 token detach，学生 token 保留 autograd 路径。

抽样上限用于控制协方差和分布距离的计算量，不改变训练 batch 顺序。若有效 token 数少于上限，使用全部有效 token；若当前 physical global micro-batch 没有有效 token，必须按数据管线重采样或显式失败，不得返回静默零损失。

### 5.2 特征归一化

分布项在每层独立标准化：

\[
x'=(x-\mu_{t,l})/(\sigma_{t,l}+\epsilon),
\]

其中 `\mu_{t,l}` 和 `\sigma_{t,l}` 由当前 batch 的 teacher target 统计得到，并 detach。student 使用同一 teacher batch statistics，避免把 teacher/student 各自标准化后抹掉分布差异。`epsilon=1e-6`。

分布项的层间聚合固定为 OS=4/8/16 等权平均。不得因层通道数、token 数或矩阵元素数不同而额外加权。

### 5.3 不允许的分布实现

- 不得使用未 mask 的 ignore token；
- 不得把 teacher/student 分别用自身均值方差标准化后再比较；
- 不得在 D1/D2/D3 中重复加入未经登记的 pointwise MSE；
- 不得把 R2 cosine matrix 当作 D 组 distribution target；
- 不得使用 R5 checkpoint 或 R5 训练缓存 feature；
- 不得跨 optimizer step 缓存或混合 token；
- 不得在未记录抽样规则的情况下随机改变 token 数或 projection 数。

---

## 6. D1：CORAL 分布对齐

### 6.1 均值和协方差

对每层当前有效 token 矩阵 `X_s^l,X_t^l∈R^{N_l×C_l}`，使用 teacher batch 的归一化统计后计算：

\[
\mu_s^l=\frac{1}{N_l}\sum_i X_{s,i}^l,
\qquad
\mu_t^l=\frac{1}{N_l}\sum_i X_{t,i}^l,
\]

\[
C_s^l=\frac{(X_s^l-\mu_s^l)^T(X_s^l-\mu_s^l)}{\max(N_l-1,1)},
\]

\[
C_t^l=\frac{(X_t^l-\mu_t^l)^T(X_t^l-\mu_t^l)}{\max(N_l-1,1)}.
\]

教师均值、中心化 token 和协方差必须 detach。学生统计保留梯度。

### 6.2 CORAL 损失

\[
L_{CORAL}^l=\frac{1}{C_l^2}\|C_s^l-C_t^l\|_F^2
 +\frac{1}{C_l}\|\mu_s^l-\mu_t^l\|_2^2.
\]

\[
L_{CORAL}=\frac{1}{3}\sum_{l\in\{4,8,16\}}L_{CORAL}^l.
\]

均值项和协方差项各自记录 raw loss，不得只记录聚合值。协方差分母、通道归一化和层间等权必须写入 `config.json`。

### 6.3 D1 总损失

首轮固定 `lambda_coral=0.1`，只允许在短程梯度校准中按预注册候选比较 `0.03/0.1/0.3`，不得正式训练中动态改变：

```text
L_D1 = L_seg + warmup(s) * (1.0 * L_feat + lambda_coral * L_CORAL)
```

若协方差项比均值项大两个数量级以上，先报告该层尺度不均衡，不得事后只保留有利项。

---

## 7. D2：SWD 分布距离

### 7.1 固定随机方向

D2 首轮固定使用 Sliced Wasserstein Distance，不同时实现 MMD。每层在 student-space channel dimension `C_l` 上生成 `64` 个固定高斯方向：

```text
num_slices = 64
direction_seed = 20260821 + layer_index
```

每个方向生成后做 L2 normalization，并在 `config.json` 记录方向数量、seed、hash。方向只作为固定 buffer，不进入 optimizer。

### 7.2 SWD 定义

对归一化 token 矩阵投影：

\[
u_s^{l,k}=X_s^l r_l^k,
\qquad
u_t^{l,k}=X_t^l r_l^k.
\]

对每个方向分别排序，并对齐相同数量的经验分位点。若 `N_s=N_t`，直接计算：

\[
L_{SWD}^l=\frac{1}{K N_l}\sum_{k=1}^{K}
\|sort(u_s^{l,k})-sort(u_t^{l,k})\|_1.
\]

若 DDP rank 的 token 数不同，先通过 all-gather 收集当前 global batch 的 token，再由 rank 0/全 rank 使用相同的 global token 顺序和数量计算；不得分别计算 local SWD 后简单平均。student gather 必须保留梯度，teacher gather detach。

\[
L_{SWD}=\frac{1}{3}\sum_{l\in\{4,8,16\}}L_{SWD}^l.
\]

排序操作必须在 FP32 中执行。固定方向、有效 token 顺序、global denominator 和空 token 处理都必须有 reference test。

### 7.3 D2 总损失

首轮固定 `lambda_swd=0.1`：

```text
L_D2 = L_seg + warmup(s) * (1.0 * L_feat + lambda_swd * L_SWD)
lambda_swd = 0.1
```

若 SWD 的有效梯度低于 feature 梯度的 `0.05`，只允许按预注册候选 `0.03/0.1/0.3` 做一次短程校准；不得同时改变方向数和权重。

---

## 8. D3：对抗分布辅助

### 8.1 判别器输入

D3 使用一个小型、共享结构的分布判别器，不对每个 OS 层单独建立不同规模的网络。每层有效 token 先进行固定 A0 projection 后的 `LayerNorm(C_l)`，沿 token 维做 masked mean 得到 `[B,C_l]`；再沿通道维使用无参数的 `adaptive_avg_pool1d` 统一到 `d=64`，三层拼接为 `[B,192]` 的 image-level distribution descriptor。该操作只用于固定维度，不引入可学习 adapter。

判别器结构固定为：

```text
Linear(192, 128) -> LeakyReLU(0.2)
-> Linear(128, 64) -> LeakyReLU(0.2)
-> Linear(64, 1)
```

判别器参数约束：只允许判别教师/学生 descriptor 的分布，不得读取标签、学生 logits、R2 matrix 或 image path；教师 descriptor detach，学生 descriptor 保留梯度。

### 8.2 对抗损失

使用 BCE-with-logits：

\[
L_D= BCE(D(d_t),1)+BCE(D(d_s.detach()),0),
\]

\[
L_{adv}=BCE(D(d_s),1).
\]

判别器最小化 `L_D`，随后冻结判别器参数、保持其前向图可将梯度传给学生 descriptor，学生最小化：

```text
L_student = L_seg + warmup(s) * (1.0 * L_feat + lambda_adv * L_adv)
```

首轮固定 `lambda_adv=0.01`。判别器每个 optimizer step 更新一次，先更新判别器再更新学生；教师始终不更新。若实现采用 gradient reversal，必须用 reference test 证明与上述交替更新的学生梯度符号和系数一致；首轮优先使用显式交替更新，避免 optimizer 语义不清。

### 8.3 D3 稳定性门

记录 discriminator loss、real/fake accuracy、student adversarial loss、descriptor 均值/方差和三项梯度。以下任一情况连续 3 个固定审计点出现时，停止 D3：

- 判别器 accuracy `>0.99` 且 student adversarial loss 不下降；
- 判别器 accuracy 约 `0.50` 且判别器 loss 非有限或梯度长期为零；
- adversarial effective gradient 超过 CE 的 2 倍；
- teacher 参数出现 gradient 或 student/teacher descriptor 出现 non-finite。

D3 不得通过扩大判别器、增加更新次数或动态修改 lambda 来绕过停止门。

---

## 9. D4：adversarial-only 诊断

D4 复用 D3 完全相同的 descriptor、判别器、交替更新、lambda 和 warm-up，只移除 pointwise feature KD：

```text
L_D4 = L_seg + warmup(s) * lambda_adv * L_adv
lambda_adv = 0.01
```

D4 只有 `seed=42`，只用于回答“对抗分布监督是否在没有 feature anchor 时仍能形成可用分割监督”。D4 不作为主候选，不得与 D3 的性能差值解释为纯对抗项独立增益；二者同时涉及 feature MSE 的主效应与对抗交互。

---

## 10. 总损失和共同权重

硬标签 CE：

\[
L_{seg}=\frac{1}{|V|}\sum_{(b,h,w)\in V}
-\log softmax(z_s)_{b,y_{b,h,w},h,w}.
\]

D0：

```text
L_D0 = L_seg + warmup(s) * 1.0 * L_feat
```

D1：

```text
L_D1 = L_seg + warmup(s) * (1.0 * L_feat + 0.1 * L_CORAL)
```

D2：

```text
L_D2 = L_seg + warmup(s) * (1.0 * L_feat + 0.1 * L_SWD)
```

D3：

```text
L_D3 = L_seg + warmup(s) * (1.0 * L_feat + 0.01 * L_adv)
```

D4：

```text
L_D4 = L_seg + warmup(s) * 0.01 * L_adv
```

所有 raw component loss 和 weighted component loss 都必须记录。不同损失的绝对数值不可直接比较，必须联合报告有效梯度、梯度 cosine、分割指标和去项对照。

---

## 11. 分布损失的梯度尺度与验收门

在共同学生 OS=16 tap 记录：

```text
grad_l2_seg_os16
grad_l2_feat_os16
grad_l2_coral_os16       # D1
grad_l2_swd_os16         # D2
grad_l2_adv_os16         # D3/D4
grad_l2_total_student
```

定义：

\[
\rho_{dist/feat}=\frac{warmup\times lambda_{dist}\times g_{dist}}
{warmup\times lambda_{feat}\times g_{feat}+\epsilon}.
\]

显式分布项 D1/D2 的目标范围沿用 R 组关系项：

```text
0.05 <= rho_dist/feat <= 0.20
```

D3/D4 的 adversarial 项不强制使用上述 feature 比例，因为 D4 没有 feature KD，D3 还受判别器动态影响；二者必须使用 `dist/CE` 和判别器稳定性门。所有 D 运行均执行：

1. CE、feature、distribution、adversarial、total loss finite；
2. 教师和固定 projection 无 grad；
3. 学生 backbone/head 有 grad；
4. warm-up 在 step=1 为 `1/4000`，step=4000 达到 1；
5. 辅助项有效梯度连续三次超过 CE 的 2 倍时停止；
6. DDP global token count、分母、mask 和 student autograd gather 与 reference 实现一致；
7. 同 seed 的 D0-D4 first batch base fields 与 K1 在 `1e-6` 容差内一致。

正式 80k 运行前，先执行相同 seed、相同 batch 的短程校准。校准只能选择已登记的 lambda，不得边训练边动态修改。

---

## 12. 正式训练前的单元测试和 smoke test

### 12.1 初始化与数据配对

- D0-D4 同 seed step=0 state hash 相同；
- `42/3407/260805` 初始化 hash 两两不同；
- D0/D1/D2/D3/D4 前 N 个 batch 路径、label hash 和增强状态一致；
- 训练 effective global batch 为 `8`；分布统计 physical global micro-batch 固定为 `4`，且不跨 micro-batch 或 optimizer step；
- D0 的首 batch CE/feature 与 K1 容差内一致。

### 12.2 共同 distribution token 测试

- teacher projection 输出与 student tap shape 一致；
- ignore 像素不进入 token；
- 只修改 ignore 像素不改变 distribution loss；
- 修改 valid 像素会改变 distribution loss；
- teacher target detach，teacher 无 grad；
- token 抽样数量、seed、顺序和 hash 可复现；
- 空有效 token 显式失败或重采样，不返回 NaN/静默零损失；
- teacher-only batch statistics 可复现，student 不得使用自身独立均值方差。

### 12.3 CORAL 测试

- 完全相同的 teacher/student token 得到 `L_CORAL≈0`；
- 仅改变均值会改变 mean term；
- 仅改变协方差会改变 covariance term；
- 协方差分母等于 `max(N-1,1)`；
- 通道归一化和三层等权聚合与公式一致；
- DDP global token 聚合与单进程 reference 一致。

### 12.4 SWD 测试

- 固定方向数量、seed、归一化和 hash 一致；
- teacher/student 分布完全相同时 `L_SWD≈0`；
- 正比例缩放在 teacher statistics 标准化后不改变 reference 结果；
- 排序后的分位点配对与公式一致；
- DDP token 数不同仍使用 global token 顺序和 denominator；
- student gather 保留梯度，teacher gather detach；
- 修改 ignore token 不改变 SWD。

### 12.5 对抗测试

- teacher descriptor 不产生 gradient；
- 判别器 real/fake label 和 BCE 符号正确；
- 判别器更新不改变 student parameter；
- student adversarial update 不改变 discriminator 参数，除非进入显式下一步；
- D3 feature term 开关正确，D4 feature term 为 `false/lambda_feat=0`；
- 判别器 accuracy/loss、student adversarial loss 均 finite；
- 停止门在模拟连续 3 次超阈值输入时触发。

### 12.6 训练与 DDP 恢复测试

- accumulation 前 N-1 个 micro-batch 使用 `no_sync()`；
- scheduler 每 optimizer step 只前进一步；
- resume 恢复 step、LR、AMP scaler、generator state、discriminator state、best checkpoint 和 D 配置；
- rank 0 独占写文件，collective 顺序一致；
- smoke test 结束后所有 rank 返回码为 0。

---

## 13. 训练流程和评估

### 13.1 每个 seed 的启动流程

1. 验证数据锁与组合 SHA-256；
2. 加载并冻结 T1，验证 checkpoint hash；
3. 加载 A0 PCA bundle，验证 teacher/data hash；
4. 加载对应 seed 的 K shared init，验证 student init hash；
5. 构造 D spec，确认 distribution 类型、抽样规则和 lambda；
6. 进行 `512×1024` shape audit；
7. 执行 distribution、mask、DDP、梯度和判别器 smoke test；
8. 通过短程 lambda 校准门后，进入 80k optimizer-step 训练。

### 13.2 每个训练 step

1. student 提取 OS=4/8/16 features 和 R-ASPP 输入；
2. student 计算全分辨率 logits 和硬标签 CE；
3. teacher 在同一增强图像上提取 native features；
4. D0-D3 经固定 A0 projection 计算 feature MSE；
5. D1 计算 masked token 均值/协方差；
6. D2 计算 masked token SWD；
7. D3/D4 构造 teacher/student descriptor，先用 detached real/fake descriptor 更新 discriminator，再冻结 discriminator 参数计算 student adversarial loss；
8. 按 optimizer step 应用 warm-up，反向并更新学生及对应辅助网络；
9. 记录 token 数、分布分量、判别器状态、梯度范数和 finite 状态。

教师 feature 前向必须共享。D1/D2/D3 不应分别调用教师多次；一次 teacher native feature 前向应同时服务 A0 projection 和 distribution target。教师 target 必须 detach，不能改变教师状态。

### 13.3 dev 和 checkpoint

- dev 只使用学生模型，不加载教师、projection、分布统计或 discriminator；
- 输出 mIoU、mAcc、pixel accuracy、19 类 IoU、small-object mIoU、boundary F1 和 confusion matrix；
- checkpoint candidate key 固定为 `(mIoU, mAcc, pixel_accuracy, -loss)`；
- 固定跑满 80k，只按 dev mIoU 选择 best checkpoint；
- 训练结束重载 best checkpoint，必须复现保存的 dev 指标并生成逐图 confusion JSONL；
- efficiency 只测学生 MobileNetV2+R-ASPP，不计入 teacher、projection、distribution sampler 或 discriminator 的部署参数量和延迟。

---

## 14. 梯度和审计产物

沿用 K/R 组 `gradient_norms.jsonl`，增加：

```text
distribution_type
distribution_lambda
distribution_raw_loss
distribution_weighted_loss
distribution_global_token_count_os4
distribution_global_token_count_os8
distribution_global_token_count_os16
grad_l2_coral_os16
grad_l2_swd_os16
discriminator_loss
discriminator_real_accuracy
discriminator_fake_accuracy
student_adversarial_loss
```

固定审计点：

```text
optimizer steps = 1, 4000, 20000, 40000, 60000, 80000
```

普通日志建议每 500 optimizer steps 一条。每条记录还应包含 CE、feature、distribution/adversarial、total loss、warm-up、LR、梯度 cosine、finite 状态和抽样 hash。

---

## 15. seed 扩展和候选筛选

### 15.1 seed=42 阶段

固定执行：

1. D0 `seed=42`，先通过 K1 等价性验收；
2. D1 `seed=42`；
3. D2 `seed=42`；
4. 只有 D1/D2 均通过实现验收但均无有效增量时，才运行 D3 `seed=42`；
5. D4 `seed=42` 可在 D3 后运行，始终只作诊断；
6. R5 三 seed复验不是 D1-D4 的启动前置条件，R5 只作当前主候选上下文。

### 15.2 候选有效性门

D1/D2/D3 扩展到三 seed，至少满足以下之一：

**主指标路径：**

- seed=42 dev mIoU 高于 D0；
- mIoU 增益大于 K1 三 seed样本标准差 `0.00219`；
- 相同 dev 图像 paired bootstrap 95% CI 支持正向差值；
- loss、梯度、token 统计和 DDP 验收全部通过。

**机制路径：**

- mIoU 不劣于 D0 的 `-0.00219`；
- boundary F1 提升至少达到 K1 样本标准差 `0.00613`，或 small-object mIoU 提升至少达到 `0.00851`；
- distribution loss 有效下降；
- 显式 distribution/feature 比例处于 `0.05-0.20`，或 adversarial 判别器稳定性门通过；
- 结果直接回答均值/协方差、切片分布或对抗分布问题。

D4 只运行 seed=42。单次最高 mIoU、没有 paired 对照、改变初始化/teacher/PCA/数据流或查看 `test_local`，均不得触发三 seed 扩展。

### 15.3 停止规则

- D0 无法复现 K1：停止 D 组，先修复公共入口；
- D1/D2 都未超过 D0 且 distribution 梯度有效：停止显式分布主线，不继续搜索更多 distance；
- D1/D2 均无效后才解锁 D3；
- D3 判别器触发稳定性停止门：停止 D3，不增加判别器容量；
- D4 低于 D0：只报告 adversarial-only 欠约束，不否定 D3；
- 任意运行查看 `test_local` 后，不得继续调整 D 组设置。

---

## 16. 统计和结果解释

每个完成运行必须报告：

- 最佳 dev step；
- mIoU、mAcc、pixel accuracy；
- small-object mIoU、boundary F1、19 类 IoU；
- confusion matrix 和逐图 confusion JSONL；
- CE、feature、CORAL/SWD/adversarial、total loss；
- 每层有效 token 数、均值/协方差或 slice 统计；
- raw/effective gradient、distribution/CE 比例和 cosine；
- D3 判别器 loss/accuracy 和停止门状态；
- checkpoint reload 复现误差；
- `test_local_evaluated=false`。

三 seed 候选报告：

- `mean ± sample std`，ddof=1；
- 每个 seed 相对同 seed D0/K1 的 mIoU 差值；
- 445 张相同 dev 图像的 paired bootstrap 95% CI；
- boundary、小目标和每类 IoU 差值；
- 最佳 step 分布和训练稳定性；
- 与 R5/K4 只作协议明确的上下文比较，不作无配对因果宣称。

允许的表述：

| 结果 | 允许结论 |
|---|---|
| D1/D2/D3 相对 D0 三 seed 稳定为正 | 对应分布约束提供独立增量 |
| D1/D2 CI 跨 0 | 当前分布约束的性能证据不足 |
| D1/D2 均无增量 | 在当前表示和权重下，显式分布项未显示任务增益 |
| D3 高于 D1/D2 且判别器稳定 | 对抗分布监督可能提供额外增量，仍需报告判别器状态 |
| D4 有效 | 仅说明 adversarial-only 具有诊断性监督价值，不说明可替代 feature KD |
| D4 失败 | 说明 adversarial-only 可能欠约束，不否定 D3 |
| 差值小于 seed 波动或 CI 跨 0 | 统一写为“性能相近/证据不足” |

R5 `seed=42` 的高分不能被用作 D1/D2/D3 的 matched baseline；D 组首先比较 D0 与 D 候选。若最终候选需要比较 R5 与 D 候选，必须在相同 seeds、相同初始化协议和冻结候选名单后进行。

---

## 17. 超参数解锁与停止条件

D0-D2 首轮完成前，禁止同时搜索分布类型、抽样上限和 lambda。首轮固定：

```text
lambda_feat = 1.0
lambda_coral = 0.1
lambda_swd = 0.1
lambda_adv = 0.01
SWD slices = 64
distribution token cap = 256/layer/physical global micro-batch
```

只有满足以下条件，才允许单变量扩展：

1. D0 等价性、mask、token sampling、DDP denominator 和 teacher detach 全部通过；
2. D1/D2/D3 在 seed=42 上 finite，且没有触发梯度或判别器停止门；
3. 代码、数据、教师、初始化和候选定义仍未查看 `test_local`；
4. 扩展时一次只改变一个变量。

允许的候选范围：

```text
lambda_coral ∈ {0.03, 0.1, 0.3}
```

不得同时搜索 CORAL/SWD 距离、token 数、随机方向数和权重；不得只给 D3 搜索而不给 D1/D2 同等预算。D2 的 MMD 只有在 SWD reference 和结果均完成后，且 SWD 明确无效时，才允许另行登记 D2-MMD，不得在 D2 目录中混合两种定义。

---

## 18. 结果目录与产物约定

建议使用独立目录：

```text
result/D_MobileNetV2_RASPP_server/
  D0/seed_42/
  D1/seed_42_lambda_0.1/
  D2/seed_42_lambda_0.1_slices_64/
  D3/seed_42_lambda_0.01/
  D4/seed_42_lambda_0.01/
```

扩展 seed 时使用：

```text
D1/seed_3407_lambda_0.1/
D1/seed_260805_lambda_0.1/
```

每次运行至少保存：

- `config.json`：D 编号、CE/feature/distribution/adversarial 开关、类型、lambda、抽样和 projection hash、warm-up、seed、batch、80k 预算、优化器、AMP、deterministic；
- `feature_taps.json` 与 shape audit；
- `first_batch_audit.json`；
- shared-init、teacher、PCA/scaler、manifest hash；
- 固定 SWD direction hash，或 CORAL reduction/schema hash；
- `training_history.json`：CE、feature、distribution/adversarial、total loss、warm-up、LR；
- `gradient_norms.jsonl`：分布梯度、比例、cosine、token counts 和判别器状态；
- `last_checkpoint.pth` 及恢复状态；
- `best_checkpoint.pth` 及 `.sha256`；
- `dev_metrics.json`、19×19 confusion matrix、逐图 confusion JSONL；
- `metrics.json`、`efficiency.json`、`software.json`；
- `test_local_evaluated=false`。

`metrics.json` 必须记录：

- D 编号和 distribution spec hash；
- shared-init、teacher、PCA、manifest hash；
- global batch、global token count、token cap、mask policy；
- CORAL/SWD/adversarial 的 reduction、方向、判别器和 lambda；
- warm-up、gradient-gate、discriminator-gate 结果；
- D0 与 K1 的等价性验收结果；
- R5/K4 只能作为 optional context metadata，不能作为初始化来源。

---

## 19. 必须禁止的实现和解释

不得：

- 使用 R5、K4、A0-FT 或任何已训练学生 checkpoint 启动 D0-D4；
- 重新生成 D-specific shared initialization；
- 让 ignore 像素进入 distribution token、均值、协方差、SWD 或 discriminator；
- 教师和学生分别使用各自 batch statistics 后再比较；
- 在 D1/D2/D3 中偷偷加入 pointwise MSE、R2 cosine 或 logits KD；
- 跨 optimizer step 缓存 token或统计量；
- 未登记地改变 token cap、SWD slices、方向 seed、判别器容量或更新次数；
- 正式训练中动态修改 lambda；
- 把 D4 当作主候选；
- 把 R5 或 K4 的非配对差异解释为 D 组因果增益；
- 查看 `test_local` 后继续调参；
- 因单个 seed 的最高 mIoU 直接扩展全部 D 实验。

---

## 20. 推荐实现与执行顺序

### 20.1 实现顺序

```text
1. 固化 D0-D4 ExperimentSpec、distribution token schema 和 shared initialization
2. 复用 K1 FixedPCAProjection 与 shared student/teacher forward
3. 实现 masked distribution token extraction 和 deterministic sampling
4. 实现 D0 并完成 K1 等价性验收
5. 实现 CORAL mean/covariance reference loss 与 DDP test
6. 实现 64-slice SWD reference loss、排序和 DDP global-token test
7. 预留小型 discriminator，完成 D3/D4 alternating-update smoke test
8. 运行 D0 seed=42，随后运行 D1/D2 seed=42
9. 只有 D1/D2 均无有效增量时，运行 D3 seed=42；再运行 D4 诊断
10. 只扩展通过筛选的 D1/D2/D3 到 seed=3407/260805
11. 汇总 paired 差值、三 seed 统计、分布曲线、梯度和判别器状态
12. 冻结 D 组候选名单后再决定最终 test_local 评估
```

### 20.2 D 组完成标准

D 组完成不是“D3 一定最高”，而是：

1. D0 与 K1 的初始化、首 batch、feature loss 和协议可审计一致；
2. D1/D2 的 mask、token sampling、统计 reduction、SWD direction 和 DDP reference test 通过；
3. D3/D4 的 discriminator 更新顺序、detach、梯度符号和停止门通过；
4. teacher/PCA/shared-init/data hash 链通过；
5. distribution/CE、distribution/feature 和 discriminator stability 均已记录；
6. 通过筛选的候选完成三 seed，或明确记录未扩展及原因；
7. best dev checkpoint 可重载复现；
8. 分布项、边界、小目标和 per-class 结果均已报告；
9. `test_local` 在候选冻结前未查看；
10. R5/K4 的协议差异已明确标注，未被用作不匹配的因果基线。
