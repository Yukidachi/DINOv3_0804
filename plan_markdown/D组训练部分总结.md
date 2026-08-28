# Cityscapes 知识蒸馏 D 组训练部分总结

更新日期：2026-08-24

## 1. 总结结论

D0 已完成 `seed=42` 锚点，D1 和 D2 已完成 `seed=42/3407/260805` 三 seed 扩充。所有运行均为 80k optimizer steps，`test_local` 均未查看。三 seed 统计使用样本标准差（ddof=1），D1/D2 的差值均相对同 seed K1。

1. **D0 与 K1 的等价性成立。** D0 `seed=42` 与 K1 的最佳 mIoU 均为 `0.522120`，差值为 0，最佳模型 state hash 相同，首批两 rank 审计通过 `1e-6` 容差。因此 D1/D2 的 matched 对照链有效。
2. **D1 CORAL 在训练 seed 层面保持正向，但 paired-bootstrap 只在 seed=42 上给出明确的图像层支持。** D1 三个 seed 相对 K1 的 mIoU 差值为 `+0.010911/+0.001138/+0.001216`，三次均为正，均值 `+0.004422 ± 0.005620`；D1 mIoU 为 `0.526454 ± 0.006117`。D1-PB 的 95% CI 分别为 `seed=42: [+0.000616,+0.020829]`、`seed=3407: [-0.007420,+0.009566]`、`seed=260805: [-0.005336,+0.008016]`。因此当前最严谨的表述是“D1 具有小幅、跨 seed 同方向的增量，但图像层证据有限”，不能写成三个 seed 都显著提升，也不能把 `seed=42` 的 `+0.010911` 当作典型收益。
3. **D2 SWD 当前配置不值得继续作为主线。** D2 相对 K1 的 mIoU 差值为 `+0.007394/-0.004076/-0.008406`，均值 `-0.001696 ± 0.008164`；两个新增 seed 均低于同 seed K1。D2 的 pixel accuracy 三次均下降，boundary F1 三次均下降，说明 seed=42 的 small-object 增益不能抵消整体稳定性问题。
4. **D2 的问题很可能与 SWD 的有效梯度过强有关，但目前不能只归因于 lambda。** D2 的 SWD/feature 梯度比例在三个 seed 的固定审计点约为 `0.30-0.93`，持续高于预注册的 `0.05-0.20` 目标带；这与 D2 的高方差和后续 seed 退化相符。但这仍是机制证据，不是仅凭结果证明“唯一原因就是 lambda=0.1”。
5. **D1 不建议立刻把 lambda 从 0.1 提高。** D1 三个 seed 的 CORAL/feature 比例约为 `0.047-0.117`，大部分位于目标带；现有证据不表明 CORAL 梯度普遍偏小。D1 的平均 mIoU 增益已超过 K1 的单次样本标准差阈值，但仍有较大 seed 波动，当前 `lambda_coral=0.1` 应作为稳定基线保留。
6. **D2-MMD 暂不建议直接开展。** MMD 可能改变梯度尺度和分布几何，但没有现有结果证明它一定优于 SWD。应先停止当前 `SWD, lambda=0.1` 主线；若确实要研究分布距离，再单独登记 `D2-MMD`，固定除距离定义外的全部配置，并先做短程实现/梯度校准，不能把 MMD 与 SWD 混在同一 D2 结论中。
7. **D5 可以作为新的探索方向，但不能立即作为正式 D 组实验开展。** D5 可定义为在 R5 的 `CE+feature+R2+logits` 基础上加入 `CORAL`，即 `CE+(feature+CORAL)+R2+logits`。但现有 D 组只注册到 D4，且 R5 目前只有 `seed=42`，因此 D5 与 R5 的单 seed 对比不能作为稳定因果结论。若确实研究该问题，应先完成 R5 的 matched 三 seed和 paired统计，再单独登记 D5 的因果问题、权重、梯度停止门和三 seed预算；D5不能回写成原D1的验证。
8. **D3/D4 当前不需要进行。** D3 的原解锁条件是 D1、D2 都通过实现验收但都没有形成有意义增量。当前 D1 三 seed 方向一致且平均为正，D2 虽然失败但不满足“D1、D2 均无效”，因此 D3 不解锁。D4 始终只是 adversarial-only 诊断，不是主候选，当前也不值得投入。

---

## 2. 固定训练协议

D0-D2 只在 MobileNetV2+R-ASPP 路线上进行，未列出的配置完全相同：

| 项目 | 固定值 |
|---|---|
| 数据划分 | `train_local/dev_local/test_local=2530/445/500`；`test_local` 未查看 |
| 数据清单组合哈希 | `033161572be28a6de295e0c5dfb62d83cd4d0a18b6039321347c58ab28b9d3c2` |
| 教师 | T1 DINOv3 ConvNeXt-T+R-ASPP，冻结；checkpoint SHA-256 为 `73cb1d3161c746d1b4ea30918ec6a1f0de5e3a4952c000cf85ddf95f3ccaddeb` |
| 学生 | MobileNetV2+R-ASPP，`weights=None`，`output_stride=16` |
| 初始化 | K 组同 seed shared scratch initialization；不加载已训练学生 checkpoint |
| 训练预算 | 80,000 optimizer steps，global batch=8；物理 distribution batch=4 |
| 优化器 | SGD，lr=`0.01`，momentum=`0.9`，weight decay=`1e-4`，poly power=`0.9` |
| feature KD | A0 固定 StandardScaler+PCA，OS=4/8/16 等权 MSE，`lambda_feat=1.0` |
| 辅助项 warm-up | 前 4,000 optimizer steps 线性 warm-up |
| 分布 token | 每层每个 physical global micro-batch 最多 256 个有效 token；ignore token 排除 |
| checkpoint 选择 | 固定跑满 80k，只按 `dev_local mIoU` 选择最佳 checkpoint |

各组唯一变量为：

| 实验 | 损失组成 | 权重/设置 | 目的 |
|---|---|---|---|
| D0 | `CE+feature` | 无分布项 | K1 受控复现锚点 |
| D1 | `CE+feature+CORAL` | `lambda_coral=0.1` | 均值/协方差分布增量 |
| D2 | `CE+feature+SWD` | `lambda_swd=0.1`，64 fixed slices | 随机投影分布距离增量 |

Excel 中 D2 行的“`MMD / SWD`”是原实验矩阵描述；实际运行目录和 `metrics.json` 均明确为 SWD。

---

## 3. seed=42 结果

三组 seed=42 的最佳 step 均为 `75129`。这些数值是单个 seed 的最佳 dev checkpoint，不代表三 seed 均值。

| 实验 | mIoU | mAcc | pixel Acc | small-object mIoU | boundary F1 |
|---|---:|---:|---:|---:|---:|
| D0 | `0.522120` | `0.589288` | `0.918958` | `0.408331` | `0.441898` |
| D1 CORAL | `0.533031` | `0.598906` | `0.915707` | `0.412481` | `0.441126` |
| D2 SWD | `0.529514` | `0.594780` | `0.908164` | `0.428653` | `0.433619` |

### 3.1 三 seed 汇总

| 实验 | mIoU | mAcc | pixel Acc | small-object mIoU | boundary F1 |
|---|---:|---:|---:|---:|---:|
| K1 anchor | `0.522032 ± 0.002194` | `0.595458 ± 0.007600` | `0.911730 ± 0.006515` | `0.409779 ± 0.008514` | `0.434824 ± 0.006129` |
| D1 CORAL | `0.526454 ± 0.006117` | `0.598474 ± 0.004390` | `0.912531 ± 0.007435` | `0.414889 ± 0.011564` | `0.436389 ± 0.010864` |
| D2 SWD | `0.520336 ± 0.007948` | `0.596087 ± 0.004896` | `0.901373 ± 0.007520` | `0.411833 ± 0.014585` | `0.426451 ± 0.008397` |

### 3.2 相对同 seed K1 的差值

| 实验 | seed=42 | seed=3407 | seed=260805 | 均值 ± 样本标准差 |
|---|---:|---:|---:|---:|
| D1 mIoU-K1 | `+0.010911` | `+0.001138` | `+0.001216` | **`+0.004422 ± 0.005620`** |
| D2 mIoU-K1 | `+0.007394` | `-0.004076` | `-0.008406` | **`-0.001696 ± 0.008164`** |

D1 三个 seed 均为正，但增益主要由 seed=42 拉高；D2 的两个新增 seed 均为负，说明 seed=42 的结果不具备代表性。D2 small-object mIoU 的三 seed 差值为 `+0.020322/+0.000612/-0.014773`，同样不稳定。

### 3.3 D1-PB 的结果

D1-PB 是对 D1 结果进行图像层统计评估的实验，不重新训练模型，也不产生新的 checkpoint。它的比较对象不是 D0 的单次运行，而是每个 seed 的 matched K1：

| D1 结果 | 配对对照 |
|---|---|
| D1 `seed=42` | K1 `seed=42` |
| D1 `seed=3407` | K1 `seed=3407` |
| D1 `seed=260805` | K1 `seed=260805` |

每个 D1/K1 运行目录中的 `dev_per_image_confusion.jsonl` 应包含相同 445 张 `dev_local` 图像的逐图 `19×19` confusion matrix。D1-PB 首先按 image name 对齐两个文件，并检查图像集合、顺序、有效像素统计和类别维度一致；任何配对失败都应终止统计，不能静默丢弃图像。

对一个 seed 的一次 bootstrap 重采样，具体步骤为：

1. 从 445 张 dev 图像中有放回抽取 445 个图像索引；同一次抽样索引同时用于 D1 和 K1，保持配对关系。
2. 分别累加抽中图像的 confusion matrix，得到 D1 和 K1 各自的总 `19×19` confusion matrix。
3. 从总 confusion matrix 计算 19 类 IoU，再取 19 类 IoU 的平均得到 mIoU；不能先平均逐图 mIoU。
4. 记录该次重采样的 `delta = mIoU(D1)-mIoU(K1)`。
5. 重复 `100000` 次，使用固定 bootstrap random seed `260820`，得到该 seed 的 delta 分布。

对 `seed=42/3407/260805` 分别完成上述过程，并至少保存以下统计量：原始 checkpoint 的 mIoU 差值、bootstrap delta 均值、delta 标准差、2.5% 分位数、97.5% 分位数和 95% CI。三 seed 的 CI 必须分别报告，不能把三个 seed 的逐图 confusion 直接混合成一个 bootstrap 数据集；训练 seed 不确定性和图像重采样不确定性是两种不同来源。

D1-PB 实际结果如下。`checkpoint delta` 是原始最佳 checkpoint 的 mIoU 差值；`bootstrap delta mean` 是 100,000 次图像重采样差值的均值。两者不同是正常的，因为 bootstrap 每次会重新抽取 445 张图像。

| seed | checkpoint delta | bootstrap delta mean | bootstrap delta std | 95% CI | CI 是否排除 0 |
|---:|---:|---:|---:|---|---|
| 42 | `+0.010911` | `+0.010436` | `0.005168` | `[+0.000616,+0.020829]` | 是 |
| 3407 | `+0.001138` | `+0.001118` | `0.004327` | `[-0.007420,+0.009566]` | 否 |
| 260805 | `+0.001216` | `+0.001325` | `0.003397` | `[-0.005336,+0.008016]` | 否 |

三个 seed 均完成了图像对齐检查：类别数为 19，图像数为 445，image name 集合和顺序一致，valid pixel 数一致，paired 检查通过。每个 seed 均使用 bootstrap random seed `260820` 和 `100000` 次重复；三 seed 没有混合成一个 combined bootstrap，因为训练 seed 不确定性与图像重采样不确定性必须分开报告。

D1-PB 的判断口径如下：

| 结果 | 允许表述 |
|---|---|
| 某 seed 的 95% CI 完全高于 0 | 该 seed 在 445 张 dev 图像上具有正向的图像层证据 |
| 某 seed 的 95% CI 跨 0 | 该 seed 的图像层增益证据不足，不能说该 seed 显著优于 K1 |
| 三个 seed 的训练差值均为正，但部分 CI 跨 0 | D1 具有跨 seed 的同方向正向结果，但图像层证据有限，不能写成全面显著提升 |
| 三 seed 的均值为正且大多数 CI 支持正向 | D1 获得较强的稳定增量证据，但仍需结合训练 seed 方差解释 |

D1-PB 不能替代三 seed 的 `mean ± sample std`，也不能替代最终 `test_local` 评估。它只量化固定 checkpoint 在 445 张 dev 图像上的抽样不确定性，不证明 D1 在其他城市、官方 val 或官方 test 上必然提升。

---

## 4. D0 与 K1 等价性验收

D0 通过 D 组入口验收：

- D0 与 K1 `seed=42` 的最佳 mIoU 都是 `0.522120045088882`，差值为 0；
- 两者最佳模型 state SHA-256 都是 `b263af6d629bd61e1f60b8037c50b0df3df80849de9005c394675b8346ddf48e`；
- 两个 rank 的首批 audit 均通过，绝对和相对容差为 `1e-6`；
- D1/D2 三个 seed 均使用对应 K shared initialization，`test_local_evaluated=false`。

因此 D1/D2 的主比较应使用同 seed K1，而不是 R5 或 K4。D0 结果文件中遗留的 “R1/R2 may proceed” interpretation 字段属于旧文本，不改变等价性结论。

---

## 5. 梯度尺度与结果解释

### 5.1 D1 CORAL

D1 三个 seed 在 OS=16 固定审计点的 CORAL/feature 有效梯度比例约为：

```text
seed=42:    0.089/0.053/0.086/0.066/0.046/0.091
seed=3407:  0.087/0.064/0.117/0.048/0.117/0.100
seed=260805:0.057/0.089/0.047/0.060/0.058/0.071
```

审计点顺序为 `step=1/4000/20000/40000/60000/80000`。比例大部分位于预注册的 `0.05-0.20` 目标带，少数点略低于 `0.05`。因此不能据此断言 D1 的 `lambda_coral=0.1` 普遍偏小。若要调参，建议只做一个短程 `lambda_coral=0.3` 校准，不建议直接投入完整三 seed；D1 `0.1` 应保留为 matched 基线。

### 5.2 D2 SWD

D2 三个 seed 的 SWD/feature 比例均明显偏高：

```text
seed=42:    0.347/0.353/0.546/0.421/0.542/0.925
seed=3407:  0.316/0.430/0.758/0.523/0.691/0.559
seed=260805:0.302/0.396/0.476/0.565/0.599/0.624
```

这支持“`lambda_swd=0.1` 使 SWD 相对 feature MSE 过强，可能扰乱逐点表征锚点”的解释，但不能证明唯一原因就是 lambda。D2 应停止当前主线；若必须继续研究 SWD，应先在 `lambda_swd=0.03` 做单变量短程校准，验证梯度比例能否进入 `0.05-0.20`，再决定是否完整训练。不能同时改变 slices、token cap 和 lambda。

### 5.3 MMD 是否可能更好

MMD 可能比 SWD 更平滑，也可能因 kernel bandwidth、样本数和高维特征尺度而失效；当前材料没有依据判断它一定更好。MMD 不能作为 D2 失败后的自动替代方案。若登记 `D2-MMD`，应固定 teacher statistics、token sampling、层间 reduction、feature loss 和 optimizer，只改变距离定义，并先通过 reference test 与梯度校准。优先级低于 D5 和 D1 的确认性校准。

---

## 6. 对四个问题的回答

### 6.1 D2 效果不佳是否来自 lambda？MMD 是否会更好？

**部分支持，但不能下定论。** 三 seed 结果证明 D2 当前配置不稳定；梯度审计证明 SWD 项长期高于 feature 项目标比例，因此 `lambda_swd=0.1` 是首要嫌疑。下一步若要验证，应先做 `lambda_swd=0.03` 的短程校准或单 seed筛选。MMD 暂不直接替换 SWD，除非完成独立登记、reference test 和同等梯度校准；没有证据表明 MMD 会自动改善结果。

### 6.2 D1 是否应该调大 lambda？

**暂不建议直接调大。** D1 的 CORAL/feature 比例总体已接近目标带，且三 seed mIoU 差值均为正。当前更需要确认小幅增益是否可重复，而不是把 `lambda_coral=0.1` 改成正式主配置。可以做 `lambda_coral=0.3` 的短程校准；只有当比例仍低于目标带、训练稳定且 dev 方向明确改善时，才考虑完整运行。

### 6.3 是否可以基于 R5 做 D5？

**可以作为新探索实验，但当前不应直接启动。** 建议定义：

```text
R5 = CE + feature + R2 + logits
D5 = CE + feature + CORAL + R2 + logits
```

其中保留 R5 的 `lambda_feat=1.0`、`lambda_r2=0.3`、`lambda_logit=0.5`、`T=4`，新增 `lambda_coral=0.1`；“替换 feature”为“将 feature 项替换成 feature+CORAL 混合项”，不能移除 feature MSE。D5 需要先等待 R5 完成 `seed=3407/260805` 和 paired统计，再按同一 matched seed设计启动；不能只用现有 R5 `seed=42` 做稳定结论。D5 是新实验，必须单独登记，不能把 D1 的三 seed结果直接外推到 R5。

### 6.4 D3、D4 是否需要？

**当前不需要。** D1 三 seed仍显示一致的正向 mIoU 差值，D2 虽然失败，但原计划要求 D1、D2 均无有效增量后才解锁 D3。D4 是 adversarial-only 诊断，不是主候选。除非后续研究目标明确转向“对抗分布项是否能在显式 CORAL/SWD 无效时提供监督”，否则不建议运行 D3/D4。

---

## 7. 后续实验优先级

| 优先级 | 实验/工作 | 建议 | 原因 |
|---|---|---|---|
| P0 | D1-PB | 已完成 | seed=42 的 CI 排除 0，其余两个 seed 的 CI 跨 0；不重新训练 |
| P1 | R5 `seed=3407/260805` | 建议优先于 D5 | R5 目前只有单 seed，D5缺少稳定的 matched 基线 |
| P1 | D1 `lambda_coral=0.3` 短程校准 | 可选 | 验证 D1 是否因权重偏小；不改变 `0.1` 主基线 |
| P2 | D2 `lambda_swd=0.03` 短程校准 | 仅在需要解释 SWD 时执行 | 直接验证过强梯度是否是主要原因 |
| P2 | D5 | 暂不直接启动 | 需单独登记，并先解决 R5 三 seed基线问题 |
| P3 | D2-MMD | 暂缓 | 没有证据优于经过校准的 SWD，且需重新登记和校准 |
| P3 | D3/D4 | 暂不执行 | D1 并未失效，未满足 D3 原解锁条件 |

若完成 R5 三 seed后，单独登记的 D5 在 matched seed设计下相对 R5 的 mIoU 增量高于预设波动阈值、paired-bootstrap支持正向且 CORAL梯度稳定，再扩展或保留 D5 为候选。若 D5 无增量，则停止“在 R5 上继续堆分布损失”的方向。

---

## 8. 当前状态与数据注意事项

| 项目 | 状态 | 说明 |
|---|---|---|
| D0 `seed=42` | 已完成 | 与 K1 完全等价的 anchor |
| D1 CORAL | 已完成三 seed | `0.526454 ± 0.006117`；三次相对 K1 均为正 |
| D1-PB | 已完成 | 三 seed、每 seed 100,000 次配对重采样；仅 seed=42 的 95% CI 排除 0 |
| D2 SWD | 已完成三 seed | `0.520336 ± 0.007948`；均值低于 K1，当前配置停止 |
| D2-MMD | 未运行 | 暂缓，不能假定优于 SWD |
| D5 | 未运行 | 可作为新探索方向，但需先完成 R5 三 seed并单独登记 |
| D3/D4 | 未运行 | 当前不满足 D3 解锁条件，D4仅诊断 |
| `test_local` | 未查看 | 所有 D1/D2 metrics 均为 `test_local_evaluated=false` |

- D1/D2 三 seed的主指标、逐 seed差值和梯度审计已更新；D0 仍只运行 `seed=42`，其余锚点引用对应 K1。
- 本次结论依据 `result/D_MobileNetV2_RASPP_server/` 下各 seed 的 `metrics.json`、`gradient_norms.jsonl` 和训练产物；Excel 仅作人工汇总，JSON 为权威来源。
- 三 seed mIoU 差值已足以判断 D2 当前配置停止；D1-PB 进一步表明 D1 只有 seed=42 的图像层 95% CI 排除 0，不能把训练 seed 层面的三次正差值写成三次图像层显著提升。
- D1-PB 结果位于 `result/D_MobileNetV2_RASPP_server/D1-PB/`，包括每个 seed 的摘要 JSON、bootstrap delta `.npy` 分布和 `summary.json`；`summary.json` 记录 `combined_bootstrap.computed=false`，这是有意保持训练 seed 与图像重采样不确定性分离。
- `efficiency.json` 仍未完成有效部署测量，不能据此比较 D1/D2 的移动端开销。
- R5 只有 `seed=42` 时，D5 只能先做 matched 单 seed增量；不能将 D5 单 seed结果写成稳定结论。
- `test_local` 继续作为最终留出集，在候选名单和权重冻结前不得查看。
