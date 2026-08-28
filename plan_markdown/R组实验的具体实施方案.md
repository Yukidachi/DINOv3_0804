# R 组实验的具体实施方案：Cityscapes 关系知识蒸馏

版本：2026-08-19
适用范围：R0-R4，首轮只在 `MobileNetV2 + R-ASPP`、Cityscapes 本地划分上执行  
依据：`plan_markdown/K组实验的具体实施方案.md`、`plan_markdown/K组训练部分总结.md`、`知识蒸馏实验分析与后续实验方向.md`、`plan_markdown/Cityscapes知识蒸馏实验详单.md`

本文把 R0-R4 从关系知识的概念定义细化为可直接实现的初始化、数据配对、关系矩阵、损失、训练、验收、统计和产物约定。R 组只研究关系约束，不改变 K 组已经锁定的教师、学生、数据、优化器、评估和留出集协议。

---

## 1. R 组要回答的问题

### 1.1 当前已锁定的前置事实

K 组已经完成 K0-K4、K3-G 和 A0-FT 三 seed。当前重要结果为：

| 项目 | 结果 | 对 R 组的意义 |
|---|---:|---|
| K1 | `0.52203 ± 0.00219` mIoU | R0 的受控 feature-KD 锚点 |
| K2 | `0.55193 ± 0.00425` mIoU | 响应蒸馏在 scratch 训练中的主要增益 |
| K3 | `0.54975 ± 0.00403` mIoU | feature 与 logits 在 mIoU 上未显示可加性 |
| K4 | `0.57049 ± 0.00582` mIoU | 当前最佳 V2 候选，但属于 A0 分阶段初始化协议 |

K4 从 A0 probe checkpoint 初始化并加入 logits KD，不是 R 组的受控基线。R0-R4 不得加载 K4、A0-FT、K2 或 K3 checkpoint，也不得把 R 组结果归因于 K4 的初始化收益。

### 1.2 R 组共同科学问题

R 组回答“教师特征之间的关系信息，除逐点 feature KD 外是否还能为学生提供增量”：

1. `R0`：新的 R 训练入口能否严格复现 K1；
2. `R1`：同一 batch 内不同图像之间的关系是否提供额外信息；
3. `R2`：同一图像不同空间位置之间的关系是否提供额外信息；
4. `R3`：跨图像关系和图内空间关系是否互补；
5. `R4`：去除逐点 feature KD 后，关系约束是否仍能单独提供可用监督。

R 组不回答：

- K4 的 A0 初始化是否适合关系蒸馏；
- logits KD 与关系 KD 的联合最优权重；
- PCA、层位、学生结构、增强或教师 checkpoint 的优劣；
- GAN、CORAL、MMD、SWD、KPCA 或激活函数替换问题。

---

## 2. R0-R4 实验矩阵

| 编号 | 硬标签 CE | A0 feature KD | R1 跨图像关系 | R2 图内空间关系 | 目的 |
|---|---|---|---|---|---|
| R0 | 是 | 是 | 否 | 否 | K1 受控复现 |
| R1 | 是 | 是 | 是 | 否 | 测量 batch 内图像关系增量 |
| R2 | 是 | 是 | 否 | 是 | 测量图内空间关系增量 |
| R3 | 是 | 是 | 是 | 是 | 测量两类关系是否互补 |
| R4 | 是 | 否 | 选定一项 | 选定一项 | relation-only 欠约束诊断 |

R4 只选择 R1 或 R2 中筛选通过的一项，不能同时启用 R1 和 R2。R4 只有一个 seed，仅作机制诊断，不得作为主候选。

首轮执行顺序固定为：

```text
R0(seed=42) -> R1(seed=42) -> R2(seed=42)
-> R3(seed=42, 仅当 R1/R2 均通过) 
-> R4(seed=42, 仅当至少一个单关系候选通过)
-> 只扩展通过筛选的候选到三 seed
```

不得在 seed=42 筛选前直接启动 R0-R4 全部三 seed。

---

## 3. 所有 R 运行必须一致的协议

### 3.1 数据、标签和增强

- `train_local=2530`、`dev_local=445`、`test_local=500`；组合清单 SHA-256 必须为：

  ```text
  033161572be28a6de295e0c5dfb62d83cd4d0a18b6039321347c58ab28b9d3c2
  ```

- 训练视图：随机缩放 `[0.5,2.0]`、随机裁剪 `512×1024`、水平翻转。
- 教师和学生必须接收同一增强后的图像 tensor。
- 标签使用 `labelIds -> trainIds 0..18`，其余值为 `ignore_index=255`。
- dev 使用原分辨率 `1024×2048`、单尺度、无水平翻转。
- 不加入 class-mix、copy-paste、伪标签、类别权重、Dice、Focal、辅助分割头或多尺度推理。

关系损失在训练 crop 上计算。R1 的关系 batch 使用当前物理 micro-batch 中的四张图；不得跨不同 optimizer step 缓存图像或特征。

### 3.2 留出集规则

所有 R 运行的 `metrics.json` 必须写入：

```json
{"test_local_evaluated": false}
```

`test_local` 不得用于训练、checkpoint 选择、关系权重选择、seed 扩展、候选筛选或解释关系损失。只有代码、数据、候选名单和协议全部冻结后，才允许进行一次最终留出评估。

### 3.3 教师

所有 R 组固定使用：

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
6. 教师 feature 关系 target 必须 `detach()`，不得出现教师梯度。

### 3.4 学生、输出步长和 feature tap

学生固定为 `MobileNetV2 + R-ASPP`、`output_stride=16`、`weights=None`：

| 层 | 学生 tap | `512×1024` 训练 crop 形状 |
|---|---|---|
| OS=4 | `backbone.3` | `[B,24,128,256]` |
| OS=8 | `backbone.6` | `[B,32,64,128]` |
| OS=16 | `backbone.17` | `[B,320,32,64]` |
| head 输入 | `backbone.18` | `[B,1280,32,64]` |

学生和教师 logits 若用于 R0-R4 的 CE 或审计，均遵守 K 组的原分辨率上采样与 `align_corners=False` 约定。

### 3.5 共同 scratch 初始化

R0-R4 复用 K 组已经生成并审计过的共同初始化：

```text
result/K_MobileNetV2_RASPP_server/shared_init/seed_<seed>/student_init.pth
```

规则：

1. R0 seed=42 必须加载现有 K seed=42 shared init；
2. R1-R4 seed=42 必须加载同一个 state dict；
3. 未来扩展到 seed=3407/260805 时，使用对应 seed 的 K shared init；
4. 不重新生成 R-specific init；
5. 不加载 A0/A5 probe、A0-FT、K2、K3、K4、S2-0 或其他学生 checkpoint；
6. 同一 seed 的 R0-R4 step=0 `student_state_sha256` 必须一致；
7. 不同 seed 的初始化 hash 必须不同。

R 组沿用 K 组的 scratch 随机策略：模型构建、DataLoader、DistributedSampler 和增强的 seed 处理必须与 K 公共入口一致。不得继承 K4 的 A0 初始化或 `seed+rank` staged-init 规则。

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

R1 关系矩阵使用同步后的物理 micro-batch：

```text
B_relation = per_gpu_batch × world_size = 4
```

不得把梯度累积后的 batch=8 用于 R1 关系构造。R1 的实现必须与单进程四张图参考实现一致。

### 3.7 warm-up

复用 K 组辅助项 warm-up：

```text
warmup_steps = 4000
warmup(s) = min(1, s / 4000)
```

CE 从第一步起使用完整权重；feature KD 和关系项只使用 warm-up 权重。`s` 必须按 optimizer step 计算，不得按 micro-batch 计算。

---

## 4. R0 的 A0 feature anchor

R0-R3 保留 K1 的固定 A0 feature KD，R4 才移除该项。复用：

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
- 不使用标签 mask、类别权重或稀有类采样；
- projection 是固定 buffer，不在 optimizer；
- 教师 features 用同一增强图像并 detach；
- PCA 参数记录中的 teacher hash、data manifest hash 必须匹配；
- 记录 `feat_loss_os4/os8/os16`、总 feature loss 和 projection hash；
- 首轮固定 `lambda_feat=1.0`。

关系矩阵不使用 projected teacher feature。A0 projection 只服务于点对点 feature anchor，避免把 R 组退化为另一种 PCA 空间上的点对点约束。

---

## 5. 关系 feature 的共同约定

### 5.1 教师和学生使用 native feature

R1/R2 的关系构造分别使用教师和学生的原始 tap feature：

```text
teacher relation source = f_t^l
student relation source = f_s^l
```

教师 feature 不经过 A0 PCA，学生 feature 也不经过 A0 projection。余弦关系对通道维度不敏感，native feature 可以保留教师原始几何信息，同时避免关系项与 `L_feat` 使用完全相同的 target 表示。

### 5.2 三层聚合

关系项在 OS=4/8/16 分别计算并等权平均：

\[
L_{rel}=\frac{1}{3}\sum_{l\in\{4,8,16\}}L_{rel}^l.
\]

关系项的矩阵计算使用 FP32；AMP 训练时应在归一化、矩阵乘法和 reduction 前转换为 FP32，避免半精度下 cosine 矩阵出现非 finite。

### 5.3 数值约定

- 使用 `eps` 防止零范数除法，具体值由共享实现固定并写入 `config.json`；首轮建议 `1e-6`；
- 行向量归一化后再进行矩阵乘法；
- 不对矩阵再次做 Frobenius 范数归一化；
- 不在已经按有效元素归一化后再次全元素取 mean；
- 教师 relation target detach；
- 全部有效位置为空时显式失败或按数据管线重采样，不返回 NaN 或静默零损失。

---

## 6. R1：跨图像 batch 关系

### 6.1 masked GAP

对每个图像、每个 feature tap，将原始 valid mask 用 nearest-neighbor resize 到 feature 分辨率，得到：

\[
m_b^l(h,w)\in\{0,1\}.
\]

对教师和学生分别计算 masked global average：

\[
g_{s,b}^l=\frac{\sum_{h,w}m_b^l(h,w)f_{s,b}^l(:,h,w)}
{\sum_{h,w}m_b^l(h,w)},
\]

\[
g_{t,b}^l=\frac{\sum_{h,w}m_b^l(h,w)f_{t,b}^l(:,h,w)}
{\sum_{h,w}m_b^l(h,w)}.
\]

每个图像必须至少有一个 valid feature location。ignore 像素不能进入 GAP 分子或分母。

### 6.2 batch cosine matrix

沿通道维做 FP32 行归一化：

\[
q_{s,b}^l=\frac{g_{s,b}^l}{\|g_{s,b}^l\|_2+\epsilon},
\qquad
q_{t,b}^l=\frac{g_{t,b}^l}{\|g_{t,b}^l\|_2+\epsilon}.
\]

将四张物理 batch 图像堆叠为 `[4,C]` 后计算：

\[
M_s^l=q_s^l(q_s^l)^T,
\qquad
M_t^l=q_t^l(q_t^l)^T.
\]

矩阵形状固定为 `[B_relation,B_relation]=[4,4]`。

### 6.3 R1 正式损失

保留对角线，按全部 `B_relation^2` 项计算 MSE：

\[
L_{R1}^l=\frac{1}{B^2}\sum_{i=1}^{B}\sum_{j=1}^{B}
\left(M_{s,ij}^l-M_{t,ij}^l\right)^2.
\]

\[
L_{R1}=\frac{1}{3}\sum_{l\in\{4,8,16\}}L_{R1}^l.
\]

对角线通常接近 1，但仍保留在定义和 denominator 中，不能在不同实现之间改变 reduction。

### 6.4 DDP 验收

R1 必须从两个 rank 收集物理 batch 的 feature 或 GAP 向量后再构造 `[4,4]` 矩阵。该 gather 必须保持 student feature 的 autograd 路径；普通不带反向传播的 `all_gather` 不能直接用于正式 loss，否则跨 rank 样本对学生梯度会被截断。可使用可微 gather 或等价的梯度保持实现。必须验证：

1. 分布式矩阵与单进程四样本参考矩阵逐元素一致；
2. 分布式 loss 与参考 loss 在容差内一致；
3. 分布式 student gradient 与参考 gradient 在容差内一致；
4. teacher relation target 不产生梯度；
5. rank 间收集顺序和样本 ID 可审计；
6. 不缓存上一 step 的 feature，不跨 step 拼 batch。

---

## 7. R2：图内空间关系

### 7.1 masked pooling

对每个图像和每个 tap：

1. 将原始 valid mask resize 到 feature 分辨率；
2. 将 mask 用 adaptive average pooling 得到 `8×16` 的 valid fraction；
3. 对 `mask * feature` 做同样的 `8×16` pooling；
4. 用 pooled feature sum 除以 pooled valid fraction；
5. valid fraction 大于 0 的 bin 标记为 valid token；
6. 没有任何 valid token 时显式失败或重采样。

得到：

```text
student tokens: [B,128,Cs]
teacher tokens: [B,128,Ct]
```

128 个 token 按 `8×16` 的 row-major 顺序排列。部分有效 bin 按有效像素比例进行 masked average，不能把 ignore 像素当作零特征参与平均。

### 7.2 token cosine matrix

对每个 token 沿通道维 FP32 归一化：

\[
\tilde X_{s,b,p}^l=\frac{X_{s,b,p}^l}{\|X_{s,b,p}^l\|_2+\epsilon},
\qquad
\tilde X_{t,b,p}^l=\frac{X_{t,b,p}^l}{\|X_{t,b,p}^l\|_2+\epsilon}.
\]

计算：

\[
A_{s,b}^l=\tilde X_{s,b}^l(\tilde X_{s,b}^l)^T,
\qquad
A_{t,b}^l=\tilde X_{t,b}^l(\tilde X_{t,b}^l)^T.
\]

每张图像得到 `[128,128]` token cosine matrix。

### 7.3 valid pair mask 和 R2 损失

令 `v_{b,p}^l` 表示 token 是否 valid，定义：

\[
V_{b,p,q}^l=v_{b,p}^lv_{b,q}^l.
\]

只对 `V=1` 的 token pair 计算：

\[
L_{R2}^l=
\frac{\sum_{b,p,q}V_{b,p,q}^l
\left(A_{s,b,pq}^l-A_{t,b,pq}^l\right)^2}
{\sum_{b,p,q}V_{b,p,q}^l}.
\]

\[
L_{R2}=\frac{1}{3}\sum_{l\in\{4,8,16\}}L_{R2}^l.
\]

对 valid token 的对角线保留在分子和 denominator 中。不得在有效 pair 已经归一化后再次对全部 `128×128` 元素取 mean。

禁止：

- 使用未经过 mask 的 ignore token；
- 对高分辨率矩阵做 Frobenius normalization 后再次全元素平均；
- 使用 A0 projection 改变 teacher relation target；
- 为 teacher/student 增加额外空间 adapter；
- 用一个全局有效比例替代逐 token pair mask。

---

## 8. 总损失和 R0-R4 定义

硬标签 CE 沿用 K 组定义：

\[
L_{seg}=\frac{1}{|V|}\sum_{(b,h,w)\in V}
-\log softmax(z_s)_{b,y_{b,h,w},h,w}.
\]

总损失如下：

### R0

```text
L = L_seg + warmup(s) × 1.0 × L_feat
```

### R1

```text
L = L_seg + warmup(s) × (1.0 × L_feat + lambda_R1 × L_R1)
lambda_R1 = 0.03  # 首轮起始值
```

### R2

```text
L = L_seg + warmup(s) × (1.0 × L_feat + lambda_R2 × L_R2)
lambda_R2 = 0.03  # 首轮起始值
```

### R3

```text
L = L_seg + warmup(s) × (
      1.0 × L_feat
    + lambda_R1_selected × L_R1
    + lambda_R2_selected × L_R2
  )
```

R3 使用 R1、R2 各自筛选出的权重，不在第一次 R3 运行前重新联合搜索两个权重。

### R4

R4 绑定 R1 或 R2 中按预注册规则选出的单个关系项：

```text
L = L_seg + warmup(s) × lambda_selected × L_selected_relation
```

R4 不包含 `L_feat`、logits KD 或另一种关系项，只作欠约束诊断。

---

## 9. 关系权重和梯度门

### 9.1 起始权重

R1 首轮和 R2 已完成的 seed=42 筛选运行均从以下权重开始：

```text
lambda_rel = 0.03
```

R2 seed=42（`result/R_MobileNetV2_RASPP_server/R2/seed_42/`）的 reduction 审计已通过：每个 OS 层均为
`sum(valid pair squared error) / global valid-pair count`，保留 valid token 对角线，三层最后等权平均；DDP
使用全局分子/分母归约，并对本 rank 梯度乘 `world_size` 后交给 DDP 平均，前向数值与全局定义一致。
R2 的 local/distributed reference tests、valid-pair denominator、teacher detach 和 first-batch base equivalence
均通过，因此没有发现额外 mean、Frobenius normalization、重复除以 token/pair 数或 DDP 梯度缩放错误。

该运行的 161 条梯度记录中，`rho_rel/feat` 范围为 `0.0033698–0.0117547`，固定审计点 `1/4000/20000/40000/60000/80000`
均未进入目标下限 `0.05`；`relation/CE` 最大值为 `0.0307698`，没有触发“连续三次超过 CE 的 2 倍”停止条件。
因此当前证据支持“`lambda_R2=0.03` 过小”，而不是 reduction 实现缩放错误。R2 的 mIoU 为 `0.5289363`，相对 R0
`0.5221200` 增加 `0.0068163`，与 R1 `0.5289685` 基本相当；这次权重调整只为满足预注册梯度门，不能把它解释为性能增益。

只有实现验收通过但梯度比例不在目标范围时，才允许在正式 80k 训练前按预注册候选重做短程校准。对于 R2，
由于 `lambda=0.03` 的整个观测范围低于下限，下一次校准先尝试更大的：

```text
R1: lambda_rel ∈ {0.015, 0.03, 0.06}
R2: lambda_rel ∈ {0.015, 0.03, 0.06, 0.3}
```

R2 的下一轮默认值固定为 `lambda_R2=0.3`。在其他条件、初始化、batch、teacher、PCA、增强和 80k
预算不变的前提下，若梯度轨迹近似线性，0.3 相对 0.03 会将有效比例放大约 10 倍，预期量级约为
`0.0337–0.1175`；实际比例必须以新运行日志为准，不能用该线性外推替代审计。若 `0.3` 仍不能进入
`0.05–0.20`，或触发 relation/CE 停止门，先检查 reduction/mask/梯度记录，再决定是否注册更大权重；
不得在正式训练中动态修改 lambda，也不得事后挑选有利权重。

### 9.2 梯度比例

在共同学生 OS=16 tap 记录：

\[
g_{feat}=\|\nabla_{\theta_{OS16}}L_{feat}\|_2,
\qquad
g_{rel}=\|\nabla_{\theta_{OS16}}L_{rel}\|_2.
\]

比较 warm-up 和 lambda 生效后的有效比例：

\[
\rho_{rel/feat}=\frac{warmup(s)\lambda_{rel}g_{rel}}
{warmup(s)\lambda_{feat}g_{feat}+\epsilon}.
\]

目标范围：

```text
0.05 <= rho_rel/feat <= 0.20
```

同时记录 relation/CE 比例和 CE、feature、relation 两两 gradient cosine。关系项的有效梯度连续三次记录超过 CE 的 2 倍时，停止并检查实现。

### 9.3 预检无效条件

正式训练前停止该候选，如果：

1. relation loss、CE、feature loss 或 total loss 非 finite；
2. relation gradient 长期为零；
3. teacher 参数出现 gradient；
4. R1 分布式参考测试失败；
5. R2 valid pair denominator 为零而未显式处理；
6. relation/feature 梯度比例明显偏离目标且不是 reduction 配置问题；
7. 关系项使用了错误 mask、错误 batch 或错误 teacher/PCA target。

---

## 10. 正式训练前的单元测试和 smoke test

当前仓库没有已提交的 R 组关系损失测试文件。实现必须先补齐确定性参考测试，再启动正式训练。

### 10.1 通用关系矩阵测试

- 行归一化后的向量范数在容差内为 1；
- teacher/student relation matrix 完全相同时 loss 约为 0；
- 对 feature 向量做正比例缩放不改变 cosine matrix；
- teacher/student 通道数不同仍可计算关系矩阵；
- signed cosine 值保留，不得取绝对值；
- 对角线按正式定义保留；
- 零范数和非 finite 输入按约定显式处理；
- teacher target detach，teacher 无 grad。

### 10.2 R1 测试

- masked GAP 不读取 ignore 像素；
- 只修改 ignore 像素不改变 R1；
- 修改 valid 像素会改变 R1；
- R1 矩阵为 `[4,4]`；
- denominator 恰为 `B_relation²`；
- 两 rank 结果与单进程四样本参考值一致；
- 分布式 student gradient 与单进程参考值一致；
- 同时置换 teacher 和 student 样本顺序不改变 loss；
- 只置换 teacher 样本会改变 loss；
- 不跨 optimizer step 缓存关系 feature。

### 10.3 R2 测试

- pooled feature 形状为 `[B,128,C]`；
- relation matrix 形状为 `[B,128,128]`；
- 部分有效 bin 按有效比例做 masked average；
- invalid token pair 不参与 loss；
- 只修改 ignore 像素不改变 R2；
- 修改 valid 像素会改变 R2；
- valid token 对角线保留；
- denominator 等于 valid pair 数量；
- 关系梯度能传到学生 backbone/head，teacher 无 grad。

### 10.4 R0 等价性测试

R0 正式训练前必须核对 K1 与 R0：

- seed=42 shared-init hash 相同；
- teacher checkpoint hash 相同；
- A0 PCA/scaler hash 相同；
- 首 batch 图像路径、label hash、增强状态相同；
- `L_feat` 首 batch 数值在容差内一致；
- warm-up 曲线一致；
- R0 seed=42 的结果应与 K1 seed=42 `mIoU=0.52212` 处于预先定义的实现误差和随机评估误差范围内。

若 R0 无法复现 K1，必须先排查公共入口，不得继续解释 R1/R2 增益。

### 10.5 DDP 和恢复测试

- 两卡 DDP 单步 forward/backward 返回码为 0；
- accumulation 的前 N-1 个 micro-batch 使用 `no_sync()`；
- scheduler 每 optimizer step 只前进一步；
- resume 恢复 step、LR、AMP scaler、DataLoader generator、best checkpoint 和 relation 配置；
- rank 0 独占写文件，collective 顺序一致；
- 正常退出按 K 组的 worker/CUDA/barrier/destroy 顺序执行。

---

## 11. 训练流程和评估

### 11.1 每个 seed 的启动流程

1. 验证数据锁和组合 SHA-256；
2. 加载并冻结 T1，验证 checkpoint hash；
3. 加载 A0 PCA bundle，验证 teacher/data hash；
4. 加载 K shared init，验证 student init hash；
5. 构造 R spec，确认关系开关和 lambda；
6. 进行 `512×1024` shape audit；
7. 执行 loss、mask、relation matrix 和 gradient smoke test；
8. 进入 80k optimizer-step 训练。

### 11.2 每个训练 step

1. student 提取 OS=4/8/16 feature 和 R-ASPP 输入；
2. student 计算 logits 和硬标签 CE；
3. teacher 在同一增强图像上提取 native features；
4. R0-R3 经固定 A0 projection 计算 `L_feat`；
5. R1 计算同步物理 batch 的 masked-GAP relation；
6. R2 计算每张图像的 8×16 token relation；
7. 按 optimizer step 应用 warm-up 和 relation lambda；
8. 反向、梯度累积、optimizer/scheduler step；
9. 记录 loss、有效 token/pair 数和梯度审计字段。

R1/R2 不应分别调用教师两次。共享的 teacher native feature 前向应同时服务 A0 projection 和 relation target；关系 target 必须 detach，不能改变教师状态。

### 11.3 dev 和 checkpoint

- dev 只使用学生模型，不加载 teacher 或 projection 参与推理；
- 输出 mIoU、mAcc、pixel accuracy、19 类 IoU、small-object mIoU、boundary F1 和 confusion matrix；
- checkpoint candidate key 沿用 `(mIoU, mAcc, pixel_accuracy, -loss)`；
- 固定跑满 80k，仅按 dev mIoU 选择 best checkpoint；
- 训练结束重载 best checkpoint，复现 dev 指标并生成逐图 confusion JSONL；
- efficiency 只测学生 MobileNetV2+R-ASPP，不计入 teacher、projection 或训练期 relation 计算。

---

## 12. 梯度和审计产物

沿用 K 组 `gradient_norms.jsonl`，增加：

```text
grad_l2_relation_r1_os4
grad_l2_relation_r1_os8
grad_l2_relation_r1_os16
grad_l2_relation_r2_os4
grad_l2_relation_r2_os8
grad_l2_relation_r2_os16
grad_l2_relation_effective_os16
relation_valid_token_count
relation_valid_pair_count
relation_physical_batch_size
```

固定审计点沿用 K3-G：

```text
optimizer steps = 1, 4000, 20000, 40000, 60000, 80000
```

普通梯度日志建议每 500 optimizer steps 一条。每条记录还应包含 raw relation loss、weighted relation loss、warm-up 权重、lambda、LR、CE/feature/relation gradient cosine 和 finite 状态。

---

## 13. 三 seed 扩展和候选筛选

### 13.1 seed=42 阶段

固定执行：

1. R0 seed=42；
2. R0 通过等价性验收后运行 R1 seed=42；
3. R1 通过实现、梯度和效果门后运行 R2 seed=42；
4. R3 仅在 R1、R2 都通过时运行；
5. R4 仅在至少一个单关系候选通过时运行。

### 13.2 候选有效性门

R1/R2/R3 可扩展到三 seed，至少满足以下条件之一：

**主指标路径：**

- seed=42 dev mIoU 高于 R0；
- 增益大于 K1 三 seed mIoU 样本标准差 `0.00219`；
- 相同 dev 图像的 paired bootstrap 支持正向差值；
- 训练 finite 且梯度比例通过。

**机制路径：**

- mIoU 不劣于 R0 的 `-0.00219`；
- boundary F1 提升至少达到 K1 的样本标准差 `0.00613`，或 small-object mIoU 提升至少达到 `0.00851`；
- relation loss 持续下降；
- relation/feature 有效梯度比例处于 5%-20%；
- 结果回答预注册的结构或空间问题。

仅凭单次最高 mIoU、无 paired 对照、改动 batch/init/teacher/PCA 或查看 test_local，均不得扩展。

### 13.3 扩展规则

- seed=42 是三 seed 结果的第一个成员，不重复运行同一配置；
- 通过筛选的 R1/R2/R3 才扩展 `3407/260805`；
- R4 永远只运行 seed=42；
- R3 使用 R1、R2 各自筛出的 lambda，不重新进行联合网格；
- 若 R1、R2 都未通过，停止 R3，不用 R3 掩盖单项失败。

### 13.4 R2 独立诊断模式

当研究目标是单独诊断 R2 图内空间关系，而不是立即执行上述预注册的
R0→R1→R2 因果筛选链时，`dino_r2_server.py` 支持独立运行
`seed=42/3407/260805`。该模式的边界如下：

- 不加载 R0 或 R1 checkpoint，不使用 R0/R1 loss；R0、R1 metrics 只作为可选的 paired 对照；
- 仍必须通过 K1 的代码、训练协议、teacher/PCA 资源校验，并加载对应 seed 的 K 组 shared initialization；
- 非 `seed=42` 时，shared initialization 使用该 seed 自己的 sidecar、嵌入 seed、模型规格和 state hash 校验，不与 K1 `seed=42` 的文件 hash 强行相等；
- 输出目录使用 `R2/seed_<seed>_lambda_<lambda>/`，缺少 R0/R1 对照时对应的比较值记录为 `null`；
- 独立 R2 结果不能替代预注册 R1→R2 流程中的效果门，也不能单独作为 R3 启动依据；R3 仍要求配对的 R1 和 R2 结果均完成并通过筛选。

同样，`dino_r3_server.py` 提供独立 R3 诊断模式，可直接使用
`seed=42/3407/260805` 研究联合 R1+R2 损失。该模式不要求 R0、R1 或 R2
metrics/checkpoint 作为启动 gate；它只加载当前 seed 的 K shared initialization，
并保留 K1 的代码、协议、teacher/PCA 和模型规格校验。R3 独立模式关闭关系梯度
比例审查和超 CE 停止 gate，不生成梯度审查记录，但仍检查 CE、feature、R1、R2
和 total loss 的有限性，以及 teacher/projection 不接收梯度。独立 R3 结果中的
R0/R1/R2 对比值在缺失对应结果时记录为 `null`，不能替代原预注册流程中的
R1/R2 筛选结论。

---

## 14. 统计和结果解释

每个完成运行必须报告：

- 最佳 dev step；
- mIoU、mAcc、pixel accuracy；
- small-object mIoU、boundary F1、19 类 IoU；
- confusion matrix 和逐图 confusion JSONL；
- CE、feature、R1、R2、total loss；
- relation valid token/pair 数；
- raw/effective gradient 和 cosine；
- checkpoint reload 复现误差；
- `test_local_evaluated=false`。

三 seed 候选报告：

- `mean ± sample std`，ddof=1；
- 每个 seed 相对 R0 的 mIoU 差值；
- 445 张相同 dev 图像的 paired bootstrap 95% CI；
- boundary、小目标和每类 IoU 的差值；
- 最佳 step 分布和训练稳定性；
- 不与 K4 做因果性 unpaired 宣称，K4 只作不同初始化协议的上下文参考。

允许的表述：

| 结果 | 允许结论 |
|---|---|
| R1/R0 或 R2/R0 三 seed 稳定为正 | 对应关系项提供独立增量 |
| R1/R0 或 R2/R0 CI 跨 0 | 性能证据不足，不能宣称有效 |
| R3 不高于单项最佳 | 两种关系在主指标上未显示互补 |
| R3 高于两项且梯度稳定 | 支持关系项存在互补，但仍需报告协议和指标取舍 |
| R4 有效 | 仅说明关系项有诊断性监督价值，不说明可替代 feature KD |
| R4 失败 | 说明 relation-only 可能欠约束，不否定 R1/R2 |

差值小于 seed 波动或 CI 跨 0 时，统一描述为“性能相近/证据不足”。

---

## 15. R4 的选定规则

R4 只绑定一个单关系候选。若 R1 和 R2 均通过，按以下固定字典序选择：

1. seed=42 dev mIoU 较高者；
2. 若相同，small-object mIoU 较高者；
3. 若仍相同，boundary F1 较高者；
4. 仍相同则选 R1。

R4 复用选定候选的 relation 定义、mask、lambda 和 warm-up，只移除 `L_feat`。R4 结果不得成为主模型选择依据，也不得因为 R4 单 seed 结果改变 R1/R2 的筛选规则。

---

## 16. 结果目录与产物约定

建议使用独立目录：

```text
result/R_MobileNetV2_RASPP_server/
  R0/seed_42/
  R1/seed_42/
  R2/seed_42_lambda_0.3/
  R3/seed_42/
  R4/seed_42/
```

R2 的新运行目录固定使用 `seed_<seed>_lambda_<lambda>`，例如重新复现 `0.03` 时保存为
`seed_42_lambda_0.03/`，当前 `0.3` 结果保存为 `seed_42_lambda_0.3/`；已有的 `R2/seed_42/`
是修改前的 legacy 目录，不自动移动或覆盖。config、checkpoint、history、metrics
和 resume 状态必须全部位于同一目录，禁止不同权重共用一个 `seed_<seed>` 目录。其他候选扩展时增加
`seed_3407/` 和 `seed_260805/`。不重新生成 shared init，直接在运行配置中记录所复用的 K shared-init 路径和 hash。

每个运行至少保存：

- `config.json`：R 编号、CE/feature/relation 开关、relation 类型、lambda、warm-up、seed、batch、80k 预算、优化器、AMP、deterministic；
- `feature_taps.json` 与 shape audit；
- `first_batch_audit.json`；
- `student_init_sha256`、teacher SHA-256、PCA/scaler SHA-256、manifest SHA-256；
- `training_history.json`：CE、feature、R1、R2、total loss、warm-up、LR；
- `gradient_norms.jsonl`：分层 relation 梯度、比例、cosine、valid counts；
- `last_checkpoint.pth` 及恢复状态；
- `best_checkpoint.pth` 及 `.sha256`；
- `dev_metrics.json`、19×19 confusion matrix、逐图 confusion JSONL；
- `metrics.json`、`efficiency.json`、`software.json`；
- `test_local_evaluated=false`。

`metrics.json` 必须记录：

- R 编号和 relation spec hash；
- shared-init、teacher、PCA、manifest hash；
- `physical_relation_batch_size=4` 和 `effective_optimizer_batch_size=8`；
- relation feature source 为 native teacher/student feature；
- mask、pooling、matrix reduction、diagonal policy、epsilon；
- lambda、warm-up、gradient-gate 结果；
- R0 与 K1 的等价性验收结果。

---

## 17. 必须禁止的实现和解释

不得：

- 使用 K4 或 A0-FT checkpoint 启动 R0-R4；
- 重新生成 R-specific shared initialization；
- 继承 K4 的 A0 staged-init 或 `seed+rank` 模型构建策略；
- 把 A0 projected teacher feature 用作 relation target；
- 把 accumulated batch=8 当作 R1 的关系 batch；
- 让 ignore 像素进入 GAP、token pooling 或 relation reduction；
- 在已经归一化的矩阵上再次做未登记的 Frobenius normalization；
- 使用 `batchmean` 或额外的类维 mean 改变矩阵尺度；
- 正式训练中动态改变 lambda；
- 在 seed=42 筛选前启动所有 R 组的三 seed；
- 把 R4 当作主候选；
- 加入 logits KD、GAN、CORAL、MMD、SWD、KPCA 或激活替换；
- 查看 `test_local` 后继续调参；
- 把 R 结果与 K4 的差异解释为关系项因果增益。

---

## 18. 推荐实现和执行顺序

### 18.1 实现顺序

```text
1. 固化 R0-R4 ExperimentSpec 和 relation reduction policy
2. 添加 R1/R2 的确定性 reference loss 与 mask 单测
3. 复用 K shared init、T1 teacher、A0 projection 和 K 评估入口
4. 实现 R0 并完成 K1 等价性验收
5. 实现 R1 的同步 physical batch relation 和 DDP reference test
6. 实现 R2 的 masked 8×16 token relation 和 valid-pair test
7. 建立 lambda=0.03 的单 batch/短程梯度校准
8. 运行 R0 seed=42，随后按门控顺序运行 R1/R2
9. 仅在 R1/R2 均通过时运行 R3；仅在单项有效时运行 R4
10. 扩展通过筛选的 R1/R2/R3 到三 seed
11. 汇总 paired 差值、bootstrap CI、梯度和结构指标
12. 冻结候选名单后再决定最终 test_local 评估
```

### 18.2 R 组完成标准

R 组完成不是“R3 一定最高”，而是：

1. R0 与 K1 的初始化、首 batch、feature loss 和协议可审计一致；
2. R1/R2 的 mask、pooling、cosine、reduction 和 DDP reference test 通过；
3. teacher/PCA/shared-init/data hash 链通过；
4. relation 梯度比例和 finite gate 通过；
5. seed=42 筛选依据和 R4 选择规则已预先固定；
6. 通过筛选的候选完成三 seed，或明确记录未扩展及原因；
7. best dev checkpoint 可重载复现；
8. 关系项、边界、小目标和 per-class 结果均已报告；
9. `test_local` 在候选冻结前未查看；
10. 任何 R 结果与 K4 的协议差异均已明确标注。
