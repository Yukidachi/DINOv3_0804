# K 组实验的具体实施方案：Cityscapes 特征知识与响应知识的 2×2 消融

版本：2026-08-12  
适用范围：K0-K3，首轮只在 `MobileNetV2 + R-ASPP` 上执行  
依据：`知识蒸馏实验分析与后续实验方向.md`、`plan_markdown/Cityscapes知识蒸馏实验详单.md`、`plan_markdown/A实验的具体实施方案.md`、`plan_markdown/A组训练部分总结.md`

本文把 K0-K3 从实验矩阵细化为可直接实现的初始化、教师前向、特征投影、logits KL、训练、验收、统计和产物约定。K 组不再进行“无标签预训练 + 冻结 probe”，而是在同一 scratch 初始化上始终使用硬标签 CE，并通过 2×2 实验分离 feature KD 与 pixel-logit KD 的贡献。

---

## 1. K 组要回答的问题

### 1.1 已锁定的前置事实

当前基线与 A 组已经满足启动 K0-K3 的条件：

| 项目 | 已锁定结果 | 对 K 组的意义 |
|---|---:|---|
| T1 教师 | `0.776024 ± 0.002086` mIoU | 稳定且明显强于学生，可提供 features 与有效 logits |
| S2-0 scratch | `0.495653 ± 0.005220` mIoU | K0 的历史参照；K0 仍需在 K 入口受控重跑 |
| A0 probe | `0.39268 ± 0.00664` mIoU | 固定 `StandardScaler+PCA` 是主 feature KD 机制 |
| A5 probe | `0.39888 ± 0.00117` mIoU | 冻结 probe 最优、最稳定，但不进入主 K0-K3 的机制定义 |
| A0-FT | `0.56395` mIoU（seed=42） | 证明 A0 表征可监督适配；不是 K 组初始化 |
| A5-FT | `0.54890` mIoU（seed=42） | 作为 A5 补充证据；不是 K 组初始化 |

主 K0-K3 将 A0 固定 `StandardScaler+PCA` 选为唯一 feature KD 机制。理由是：

1. A0 是机制最简、投影固定、哈希可审计的主基线；
2. A0-FT 是当前最高的端到端结果；
3. A5 的训练期学生 adapter 在 K 组持续端到端训练中会引入额外变量，不适合放入主 2×2；
4. 若 K1 有效，后续可另设补充对照验证 A5，但不得改写 K1/K3 的预注册定义。

### 1.2 K0-K3 的共同科学问题

K 组回答“在真实监督分割训练中，教师的中间表征和像素响应分别提供多少增量，二者是否互补”：

1. `K1-K0`：没有教师 logits 时，中间 feature KD 是否优于纯 CE；
2. `K2-K0`：没有 feature KD 时，pixel-logit KD 是否优于纯 CE；
3. `K3-K1`：已有 feature KD 后，教师 logits 是否仍有增量；
4. `K3-K2`：已有 logits KD 后，feature KD 是否仍提供额外信息。

K 组不回答 PCA 机制优劣，不调整层位、PCA 维度、head、增强或初始化来源。上述变量已由 P0/A 组锁定。

---

## 2. 四组实验必须完全相同的协议

### 2.1 数据、标签和增强

- `train_local=2530`、`dev_local=445`、`test_local=500`；组合清单 SHA-256 必须为：

  ```text
  033161572be28a6de295e0c5dfb62d83cd4d0a18b6039321347c58ab28b9d3c2
  ```

- 训练视图：随机缩放 `[0.5,2.0]`、随机裁剪 `512×1024`、水平翻转；教师和学生必须接收同一增强后的 tensor。
- 标签：`labelIds -> trainIds 0..18`，其余为 `ignore_index=255`。
- dev：原分辨率 `1024×2048`、单尺度、无水平翻转。
- `test_local` 不参与 K0-K3 训练、checkpoint 选择、损失调参或实验筛选；K 阶段所有运行的 `test_local_evaluated` 必须为 `false`。
- 不加入 class-mix、copy-paste、伪标签、类别权重、Dice、Focal、辅助分割头或多尺度推理。

### 2.2 教师

所有 K 组固定使用：

```text
result/T1_DINOv3_RASPP/seed_3407/t1_dinov3_raspp_teacher.pth
```

checkpoint SHA-256：

```text
73cb1d3161c746d1b4ea30918ec6a1f0de5e3a4952c000cf85ddf95f3ccaddeb
```

要求：

1. 使用 `dino_t1.load_teacher_for_distillation()` 加载，不得误用 `dino.py` 中的 T0 loader；
2. 调用 `freeze_for_distillation()`，教师始终 `eval()` 且全部参数 `requires_grad=False`；
3. 教师不进入 optimizer、不包 DDP；每个 rank 各自持有同一冻结教师副本；
4. K1/K3 使用教师 `extract_features()` 的 `os4/os8/os16`；K2/K3 使用教师 `forward()` 的已训练 19 类 R-ASPP logits；
5. 禁止使用随机 head、T0 head、未训练 head 或重新选择教师 seed。

### 2.3 学生、输出步长和特征 tap

学生固定为 `MobileNetV2 + R-ASPP`、`output_stride=16`、`weights=None`：

| 层 | 学生 tap | 学生形状（512×1024） | 教师 tap | 教师形状 |
|---|---|---|---|---|
| OS=4 | `backbone.3` | `[B,24,128,256]` | ConvNeXt stage 0 | `[B,96,128,256]` |
| OS=8 | `backbone.6` | `[B,32,64,128]` | ConvNeXt stage 1 | `[B,192,64,128]` |
| OS=16 | `backbone.17` | `[B,320,32,64]` | ConvNeXt stage 3 | `[B,768,32,64]` |
| head 输入 | `backbone.18` | `[B,1280,32,64]` | — | — |

学生和教师最终 logits 均双线性上采样到输入尺寸 `[B,19,512,1024]`，固定 `align_corners=False`。

### 2.4 共同 scratch 初始化与 batch 顺序

这是 K 组公平性的核心约束：

1. K0/K1/K2/K3 **全部从 scratch 开始**，不得加载 A0/A5 probe、A0-FT/A5-FT、S2-0 或其他学生 checkpoint；
2. `A_best` 仅表示 feature target 的投影机制，不表示学生初始化；
3. 对每个 seed，在模型构建后、optimizer 创建前保存共同的 `student_init_seed_<seed>.pth` 和 SHA-256；四组必须加载这一个 state dict；
4. 正式 seed 为 `42/3407/260805`；同一 seed 下四组使用相同 student init、R-ASPP head init、DistributedSampler seed、DataLoader generator state 和 batch 顺序；
5. 不同 seed 使用各自初始化，不能把 seed=42 的权重复制给另外两个 seed；
6. K0 是在 K 组公共入口内受控重跑的 CE-only 基线，不直接复用已有 S2-0 数值或 checkpoint。

验收：同一 seed 的四组在 step=0 的 `student_state_sha256` 必须完全一致；不同 seed 的哈希必须不同。

### 2.5 预算与优化

| 项目 | 固定值 |
|---|---|
| optimizer steps | 80,000 |
| global batch | 8 |
| optimizer | `SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)` |
| scheduler | poly，power=0.9，min LR ratio=0.01 |
| AMP | 开启 |
| deterministic | 开启 |
| checkpoint 选择 | 只按 dev mIoU |
| eval 间隔 | 每 5,000 optimizer steps 附近的首个 epoch 边界 |

服务器 DDP 沿用 `dino_s2_0_server.py`：`spawn`、默认 `pin_memory=False`、梯度累积时非最后 micro-batch 使用 `no_sync()`、正常退出按“停止 worker → CUDA synchronize → 释放 DDP/optimizer → barrier → destroy process group”。

---

## 3. K0-K3 实验矩阵

| 编号 | 硬标签 CE | A0 feature KD | T1 logits KD | 目的 |
|---|---|---|---|---|
| K0 | 是 | 否 | 否 | 受控 CE-only 基线 |
| K1 | 是 | 是 | 否 | 测量中间 feature KD 的独立增量 |
| K2 | 是 | 否 | 是 | 测量 pixel-logit KD 的独立增量 |
| K3 | 是 | 是 | 是 | 测量两种知识的互补性 |

四组唯一允许不同的是损失项开关。教师、student init、增强、batch 顺序、优化器、训练预算、评估代码和 checkpoint 选择必须一致。

不得从 K0/K1/K2 的 checkpoint 启动 K3；四组均从共同 step=0 state 独立训练。

---

## 4. 损失的正式定义

### 4.1 硬标签分割损失（K0-K3 全部启用）

对学生全分辨率 logits `z_s∈R^{B×19×H×W}`：

\[
L_{seg}=\frac{1}{|V|}\sum_{(b,h,w)\in V}
-\log softmax(z_s)_{b,y_{b,h,w},h,w},
\]

其中 `V={(b,h,w) | y[b,h,w] != 255}`。实现应使用 `cross_entropy(..., ignore_index=255, reduction="sum") / valid_pixel_count`，与 S2-0 保持一致。若 batch 无有效像素，按数据管线约定重采样；不得把该 batch 静默记为零损失。

### 4.2 A0 固定特征损失（K1/K3）

复用已锁定产物：

```text
result/A_MobileNetV2_RASPP_server/pca_shared/
```

加载 `scaler_os4/os8/os16.npz`、`pca_os4/os8/os16.npz`，使用 A0 的显式 `FixedPCAProjection`：

\[
\hat f_t^l=P_l(f_t^l),
\]

\[
L_{feat}=\frac{1}{3}\sum_{l\in\{4,8,16\}}
\operatorname{mean}_{B,C,H,W}\left(f_s^l-\hat f_t^l\right)^2.
\]

要求：

- 三层等权；每层先对 BCHW 全元素取 mean，再除以 3；
- 不使用标签 mask、类别权重或稀有类采样；
- projection 是固定 buffer，`requires_grad=False`，不在 optimizer；
- 教师 features 使用同一增强图像并 `detach()`；
- PCA 参数记录中的 teacher checkpoint hash、数据 manifest hash 必须与本次运行一致；
- 记录 `feat_loss_os4/os8/os16`、总 feature loss、投影参数哈希。

首轮固定：

```text
lambda_feat = 1.0
```

### 4.3 全分辨率 masked logits KL（K2/K3）

教师和学生都通过各自完整 R-ASPP 前向得到 `[B,19,H,W]` logits。首轮不在 OS=16 logits 上计算 KL，原因是全分辨率路径可直接使用原始 ignore mask，避免定义 mask 下采样规则这一额外变量。

温度固定：

```text
T = 4
lambda_logit = 0.5
```

定义：

\[
p_t=softmax(z_t/T),\qquad \log p_s=log\_softmax(z_s/T),
\]

\[
L_{logit}=T^2\cdot\frac{1}{|V|}
\sum_{(b,h,w)\in V}\sum_{c=1}^{19}
p_{t,c}\left(\log p_{t,c}-\log p_{s,c}\right).
\]

推荐实现顺序：

```python
valid = targets != 255                             # [B,H,W]
teacher_prob = softmax(teacher_logits / T, dim=1)
student_log_prob = log_softmax(student_logits / T, dim=1)
kl_per_class = kl_div(student_log_prob, teacher_prob,
                      reduction="none")           # [B,19,H,W]
kl_per_pixel = kl_per_class.sum(dim=1)             # [B,H,W]
loss_logit = (T * T) * kl_per_pixel[valid].mean()
```

禁止事项：

- 不得把 ignore 像素作为教师软标签参与均值；
- 不得对 19 类再次取 mean，否则会额外缩小 19 倍；
- 不得使用 `batchmean` 后再手工除像素；
- 不得遗漏 `T²`；
- 不得 detach 学生 logits；教师 logits 必须 detach；
- 不得让 K2/K3 使用不同的教师 head 或不同 logits 分辨率。

### 4.4 总损失与 warm-up

设全局 optimizer step 为 `s`，总步数 80,000，辅助项 warm-up 步数：

```text
warmup_steps = 0.05 × 80000 = 4000
warmup(s) = min(1, s / 4000)
```

仅辅助项 warm-up，CE 从第一步起全权重启用：

\[
L_{K0}=L_{seg},
\]

\[
L_{K1}=L_{seg}+warmup(s)\cdot 1.0\cdot L_{feat},
\]

\[
L_{K2}=L_{seg}+warmup(s)\cdot 0.5\cdot L_{logit},
\]

\[
L_{K3}=L_{seg}+warmup(s)\cdot
\left(1.0\cdot L_{feat}+0.5\cdot L_{logit}\right).
\]

梯度累积时，warm-up 的 `s` 必须按 optimizer step 计算，而不是 micro-batch 数。

---

## 5. 梯度尺度与正式训练前验收门

正式 80k 运行前，先对每个 K 编号执行相同 batch 的单步/短程验收。K1/K2/K3 至少记录辅助项对一个共同学生高层 tap（建议 OS=16 student feature）的梯度范数：

```text
grad_l2_seg_os16
grad_l2_feat_os16       # K1/K3
grad_l2_logit_os16      # K2/K3
grad_l2_total_student
```

也可额外记录 OS=4/8，但四组必须使用相同计算点。

验收规则：

1. CE、各层 feature loss、KL、总损失均 finite；
2. 教师所有参数 grad 必须为 `None`；projection 无 grad；学生 backbone 与 head 都应有 grad；
3. step=1 的 warm-up 权重为 `1/4000`，step=4000 达到 1.0；
4. 辅助项达到满权重后，若任一辅助项对共同学生层的梯度长期大于 CE 梯度 2 倍，停止正式运行并检查 reduction、`T²`、类维求和、像素 mask 和梯度累积；
5. 首轮不得因为梯度偏大直接单独调 K1/K2/K3 权重。只有确认实现无误、并为所有相关组重新预注册权重后，才允许另建扩展编号；
6. K0-K3 相同 seed 的第一个 batch 路径、label hash 和 student step=0 hash 必须一致。

“长期”以连续的预注册日志点为准；入口建议每 500 optimizer steps记录一次分量和梯度。不得根据单个异常 batch 判定失败。

---

## 6. 训练流程

### 6.1 共同启动流程

每个 seed：

1. 验证数据锁与组合 SHA-256；
2. 加载并冻结 T1 `seed=3407`，验证 checkpoint sidecar；
3. 对 K1/K3 加载 A0 PCA bundle，验证 PCA teacher/data hash；K0/K2 不加载或实例化 projection；
4. 用 `weights=None` 构建一次学生，保存共同 init state；
5. K0-K3 分别重建相同结构并严格加载共同 state，验证 SHA-256；
6. 做 `512×1024` shape audit；
7. 做 loss/mask/梯度 smoke test；
8. 进入 80k step 训练。

### 6.2 每个训练 step 的前向

推荐共享前向骨架：

1. student `extract_features(images)`，得到 OS=4/8/16 和 R-ASPP 输入；
2. student head 计算低分辨率 logits，并双线性上采样到输入尺寸；
3. 计算 `L_seg`；
4. 若启用 feature KD：教师提取三层 feature，经固定 A0 projection 后计算 `L_feat`；
5. 若启用 logits KD：教师完整 forward 得到全分辨率 logits，计算 masked `L_logit`；
6. 按当前 optimizer step 应用 warm-up，反向总损失；
7. 按 DDP 梯度累积规则 step optimizer/scheduler。

K3 不应分别调用教师两次。一次教师 feature 提取后，可用教师 `teacher_head(features["os16"])` 得到低分辨率 logits，再按教师 forward 的同一 `bilinear/align_corners=False` 规则上采样。必须加单测证明该共享路径与 `teacher(images)` 输出在 FP32 容差内一致。

### 6.3 dev 评估与 checkpoint 选择

- dev 只使用学生模型；评估时不加载教师或 projection 参与推理；
- 每次评估输出 mIoU、mAcc、pixel accuracy、19 类 IoU、small-object mIoU、boundary F1、confusion matrix；
- candidate key 固定为 `(mIoU, mAcc, pixel_accuracy, -loss)`；
- 最佳 checkpoint 按 dev mIoU 选出，训练仍运行到 80k 固定预算；
- 训练结束后重载 best checkpoint，必须复现保存的 dev 指标并生成逐图 confusion JSONL；
- 效率评估只测学生 `MobileNetV2+R-ASPP`，不得把教师/PCA/训练期 KD 计算计入部署参数量和延迟。

---

## 7. K0-K3 的逐项定义

### K0：CE-only 受控基线

```text
L = L_seg
teacher = 不构建
projection = 不构建
```

K0 不是简单引用 S2-0。它必须在 K 公共入口中运行三 seed，确保初始化 state、batch 顺序、评估间隔和产物格式与 K1-K3 一致。K0 与已有 S2-0 三 seed 若差异超过既有波动，应先排查公共入口，不得继续解释 KD 增益。

### K1：硬标签 CE + A0 feature KD

```text
L = L_seg + warmup × 1.0 × L_feat
teacher features = os4/os8/os16
teacher logits = 不使用
```

K1 回答固定低秩中间表征在监督训练中是否提供增量。它不加载 A0 预训练学生，不使用 A5 adapter，不使用 logits KD。

### K2：硬标签 CE + pixel-logit KD

```text
L = L_seg + warmup × 0.5 × L_logit
T = 4
teacher features = 仅用于生成 logits 的内部前向，不计算 feature loss
```

K2 回答经过训练的教师 R-ASPP 响应是否提供类别相似度、边界和稀有类信息。必须确认 KL 只在非 ignore 像素计算。

### K3：硬标签 CE + A0 feature KD + pixel-logit KD

```text
L = L_seg + warmup × (1.0 × L_feat + 0.5 × L_logit)
T = 4
```

K3 必须复用 K1/K2 的同一损失实现，不得复制后产生公式分叉。K3 回答两种知识是否互补，不是“把所有损失堆在一起后是否最高”。

---

## 8. 必须完成的单元测试和 smoke test

### 8.1 初始化与数据配对

- 同一 seed 下 K0-K3 step=0 model state SHA-256 相同；
- seed 42/3407/260805 的 init hash 两两不同；
- 相同 seed 的前 N 个 batch 图像路径、label tensor hash 和几何增强参数一致；
- global batch=`local batch × accumulation × world size=8`。

### 8.2 feature KD

- A0 projection 三层输出 shape 与学生 tap 完全一致；
- 固定 projection 不在 optimizer，训练前后参数 hash 不变；
- feature loss 等于三层 BCHW mean 的算术平均；
- 交换层顺序不改变总 feature loss；
- feature loss 不读取 target/mask；
- PCA 参数的 teacher/data hash 不匹配时入口必须失败。

### 8.3 logits KD

- `T=1` 时与直接 softmax/log-softmax 手算值一致；
- 教师和学生 logits 完全相同时 KL≈0；
- 修改 ignore 像素处的 teacher/student logits 不改变 KL；
- 修改一个 valid 像素 logits 会改变 KL；
- 类维使用 sum、有效像素维使用 mean；
- `T²` 存在；T=4 的实现与参考公式一致；
- 全部像素 ignore 时必须触发数据重采样/显式错误，不能返回 NaN 或 0；
- K3 的教师共享 feature→head 路径与 `teacher(images)` 输出等价。

### 8.4 训练与 DDP

- K0 只产生 CE；K1 只额外产生 feature；K2 只额外产生 logit；K3 两者都有；
- 教师无 grad，学生 backbone/head 有 grad；
- accumulation 前 N-1 个 micro-batch 使用 `no_sync()`；
- scheduler 每 optimizer step 只前进一步；
- resume 后 step、LR、AMP scaler、generator state、best checkpoint 和 loss 配置完全恢复；
- rank 0 独占写文件；所有 rank collective 顺序一致；
- smoke test 结束后所有 rank 返回码为 0。

---

## 9. 三 seed 统计与结果解释

K0-K3 全部运行 `seed=42/3407/260805`，报告：

- 每个 seed 的最佳 dev step 和全部主指标；
- `mean ± sample std`（ddof=1）；
- 相同 seed 下的配对差值：`K1-K0`、`K2-K0`、`K3-K1`、`K3-K2`；
- dev 逐图 paired bootstrap 95% CI；
- 19 类 IoU 差值、small-object mIoU、boundary F1；
- 训练稳定性、最佳 step 是否集中在预算末端；
- CE/feature/KL 分量和梯度比曲线。

解释规则：

| 结果 | 允许的结论 |
|---|---|
| K1 > K0 且三 seed 稳定 | A0 中间特征在监督训练中有独立增量 |
| K2 > K0 且三 seed 稳定 | 有效教师 logits 提供独立响应知识 |
| K3 > K1 | logits 在 feature KD 之外仍有增量 |
| K3 > K2 | feature KD 在 logits KD 之外仍有增量 |
| K3≈max(K1,K2) | 两类知识存在较大冗余，不应宣称互补 |
| K3 < K1 或 K2 | 辅助项可能竞争；先查梯度尺度和 reduction，不直接调参 |

差值小于 seed 波动或 CI 跨 0 时描述为“性能相近/证据不足”，不能按单次最高值宣布胜出。

---

## 10. 超参数解锁与停止条件

首轮 K0-K3 完成前，禁止搜索温度和 KD 权重。

只有满足以下条件，才允许单变量扩展：

1. K2 或 K3 相对对应基线在至少三 seed 上稳定受益；
2. KL 实现、mask、`T²` 和梯度尺度验收全部通过；
3. 代码、数据、教师、初始化和候选定义仍未查看 `test_local`。

解锁后每次只改一个变量：

```text
T ∈ {2, 8}                # lambda_logit 固定 0.5
或
lambda_logit ∈ {0.25,1.0} # T 固定 4
```

不得同时搜索 T 与 lambda，不得只给 K3 搜索而不给 K2 同等预算。`lambda_feat=1.0` 在 K 主矩阵中保持不变。

停止条件：

- K2/K3 三 seed 均无改善：停止 logits 温度/权重搜索；
- K1 无改善：进入关系实验前先确认 A0 feature KD 的监督训练实现，而不是直接用更复杂关系损失掩盖失败；
- 教师梯度非空、hash 不匹配、KL mask 错误、loss 非 finite 或辅助梯度长期超过门槛：该运行无效，修复后从 step=0 重跑；
- 查看 `test_local` 后不得再调整 K 组设置。

---

## 11. 结果目录与产物约定

建议使用独立目录：

```text
result/K_MobileNetV2_RASPP_server/
  shared_init/
    seed_42/student_init.pth(.sha256)
    seed_3407/student_init.pth(.sha256)
    seed_260805/student_init.pth(.sha256)
  K0/seed_*/
  K1/seed_*/
  K2/seed_*/
  K3/seed_*/
```

每次运行至少保存：

- `config.json`：K 编号、loss 开关、T、lambda、warm-up、seed、global batch、80k 预算、优化器、AMP、deterministic；
- `feature_taps.json` 与 shape audit；
- `student_init_sha256`、训练脚本 SHA-256、数据锁 SHA-256；
- K1/K3 的 PCA/scaler/manifest hash；
- K1/K2/K3 的教师 checkpoint hash；
- `training_history.json`：CE、feature 总/分层、KL、warm-up 权重、LR、梯度范数；
- `gradient_norms.jsonl`：每 500 step 的 CE/feature/logit/total 梯度；
- `last_checkpoint.pth`：model、optimizer、scheduler、scaler、generator state、当前 step；
- `best_checkpoint.pth` 及 `.sha256`；
- `dev_metrics.json`、19×19 confusion matrix、逐图 confusion JSONL；
- `efficiency.json`：仅学生部署模型；
- `software.json` 或 metrics 内嵌 Python/PyTorch/torchvision/CUDA/platform 版本；
- `test_local_evaluated=false`。

K0 目录也必须保存与 K1-K3 相同 schema 的字段；未启用的损失写 `null`/`false`，不能省略关键 schema 导致汇总脚本分叉。

---

## 12. 推荐实现与执行顺序

### 12.1 实现顺序

```text
1. 固化 ExperimentSpec(K0/K1/K2/K3) 和共享初始化产物
2. 实现并单测 masked full-resolution KL
3. 复用 A0 FixedPCAProjection，并单测 hash/shape/reduction
4. 实现共享 student+teacher 前向，只由 spec 开关损失项
5. 建立单 batch loss/gradient smoke test
6. 建立 DDP 两卡 smoke 和有序退出验收
7. 运行 K0-K3 seed=42 短程预检，核对首 batch/init/loss
8. 正式运行 K0-K3 × seeds 42/3407/260805，均为 80k steps
9. 汇总配对差值、三 seed 统计和 bootstrap CI
10. 冻结 K 组结论，再决定是否解锁 T/lambda 搜索或进入 R 组
```

### 12.2 服务器命令示意

入口名称可以不同，但四组应由同一共享实现和 `--experiment` 开关驱动：

```bash
torchrun --standalone --nproc_per_node=2 dino_k_server.py \
  --experiment K0 --seed 42 \
  --batch-size 2 --global-batch-size 8 \
  --num-workers 8 --multiprocessing-context spawn \
  --no-pin-memory --persistent-workers
```

依次把 `--experiment` 替换为 K1/K2/K3，并对 seed 42/3407/260805 全部执行。不要为四组复制四份训练循环；实验差异应由只读 spec 控制，避免损失实现和 DDP 生命周期分叉。

### 12.3 K 组完成标准

K 组完成不是“某个 K 编号最高”，而是：

1. 四组同 seed 的 init 和 batch 顺序可证明一致；
2. 教师/PCA/hash 链通过；
3. KL mask/reduction/`T²` 单测通过；
4. K0-K3 三 seed 均跑满 80k 固定预算；
5. best dev checkpoint 可重载复现；
6. 四个预注册配对差值和 CI 均已报告；
7. `test_local` 未查看；
8. 是否进入温度/权重搜索或 R 组有明确、基于三 seed 的解锁结论。
