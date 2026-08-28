# Cityscapes 知识蒸馏 K 组训练部分总结

更新日期：2026-08-17

## 1. 总结结论

K0-K3 已按预注册协议完成 `seed=42/3407/260805` 三次训练，12 个运行均达到 80k optimizer steps；作为外部协议参考的 A0-FT 已补齐相同三 seed，K4 也已完成三 seed，K3-G `seed=42` 梯度夹角审计已完成 80k。K0-K4 与 K3-G 共 16 个 K 组运行全部完成，最佳 checkpoint、dev 指标、逐图 confusion matrix、训练历史和相关梯度日志完整。当前结论如下：

1. **响应蒸馏是 scratch 端到端训练中的主要增益来源。** K2 三 seed mIoU 为 `0.55193 ± 0.00425`，相对 K0 的配对增益为 `+0.04024/+0.05185/+0.04288`；三个 seed 的逐图 paired-bootstrap 95% CI 均完全高于 0。
2. **特征蒸馏单独使用也有效，但增益明显小于响应蒸馏。** K1 mIoU 为 `0.52203 ± 0.00219`，相对 K0 的平均增益为 `+0.01509`，三个 seed 的 paired-bootstrap 95% CI 同样均高于 0。因此不能把 K1 解释为无效，只能说其任务指标增益较弱。
3. **K3 与 K2 的 mIoU 性能相近，未证明两种知识在主指标上可加。** K3 mIoU 为 `0.54975 ± 0.00403`，相对 K2 的配对差值为 `+0.00328/-0.00736/-0.00246`，三次 95% CI 均跨 0。主指标排序应写为 `K2≈K3>K1>K0`，不应按单个 seed 宣称 K3 胜出。
4. **“不增加 mIoU”不等于“特征知识完全冗余”。** K3 相比 K2 的平均 boundary F1 提高 `+0.01736`，pixel accuracy 提高 `+0.00356`，但 mAcc 降低 `-0.01414`、small-object mIoU 降低 `-0.00919`。特征项更像改变了边界与类别均衡之间的取舍，而不是提供统一的 mIoU 增益。
5. **A0-FT 三 seed 后不再稳定高于 K2/K3。** A0-FT mIoU 为 `0.55215 ± 0.01022`，与 K2 的同 seed 差值为 `+0.01325/-0.01028/-0.00228`，均值仅 `+0.00023`；与 K3 的差值为 `+0.00996/-0.00293/+0.00018`，均值 `+0.00241`。seed=42 是对 A0-FT 偏有利的单次结果，主指标应写为 `A0-FT≈K2≈K3`，而不是 A0-FT 稳定领先。
6. **“20-40 epoch 后辅助损失都停在 1 附近”与日志不完全一致。** 20-40 epoch 只对应约 step 6340-12680；此后 feature loss 和 logits KL 仍持续下降。feature loss 的三层均值到训练末期约为 K1 `1.01-1.03`、K3 `1.08-1.11`，而带 `T²` 的原始 logits KL 最终约为 `0.75-0.79`，并非停在 1。
7. **K3-G 在固定审计样本上未观察到负夹角，更支持 feature 信号弱而非已测样本中的方向冲突。** 6 个时点、2 个 rank 的全部已记录 cosine 均非负；`cos(CE,feature)` 在 OS=4/8 的均值为 `0.473/0.480`，OS=16 仅 `0.028`。warm-up 后 feature 有效梯度约为 CE 的 `4%-13%`，logits 则约为 CE 的 `0.85-1.76×`。但审计只覆盖每个 rank 的一个固定 micro-batch，共 4 张图，不能据此宣称全数据集或参数梯度不存在冲突。
8. **K4 已补齐最后一个必须新增的因果单元，并成为当前最佳 V2 候选。** K4 三 seed mIoU 为 `0.56818/0.56618/0.57710`，汇总为 `0.57049 ± 0.00582`；相对 A0-FT 的增量 `Delta_s` 为 `+0.00423/+0.01981/+0.03095`，三个 seed 均为正。
9. **A0 初始化与在线 logits KD 存在可测的负向、次加性交互。** 预注册交互量 `I_s=(K4-A0-FT)-(K2-K0)` 为 `-0.03600/-0.03204/-0.01192`，三者同号，均值 `-0.02665`，且 `|mean(I)|>0.00425`。这表示 logits KD 在 A0 初始化上仍有稳定增益，但增量小于其在 scratch 初始化上的增量，二者部分重叠而非完全独立可加。
10. **K 组当前没有必须补充的实验，可以进入 R 组。** K3-noOS8 仅在层位归因成为论文核心问题时解锁，logits 温度/权重搜索仅在冻结最终 KD 配置时解锁；二者均不影响当前 K 组机制结论。`test_local` 继续保持未查看。

---

## 2. 固定训练协议

K0-K3 使用同一 Cityscapes 本地划分、教师、学生结构、初始化生成规则、batch 顺序、优化器和评估实现：

| 项目 | 固定值 |
|---|---|
| `train_local/dev_local/test_local` | `2530/445/500`；`test_local` 当前未查看 |
| 数据清单组合哈希 | `033161572be28a6de295e0c5dfb62d83cd4d0a18b6039321347c58ab28b9d3c2` |
| 教师 | T1 `seed=3407`，DINOv3 ConvNeXt-T + R-ASPP，冻结 |
| 教师 checkpoint SHA-256 | `73cb1d3161c746d1b4ea30918ec6a1f0de5e3a4952c000cf85ddf95f3ccaddeb` |
| 学生 | MobileNetV2 + R-ASPP，`weights=None`，`output_stride=16` |
| 正式 seed | `42/3407/260805`；同 seed 下 K0-K3 共用 step=0 state 和 batch 顺序 |
| 预算 | 80,000 optimizer steps，global batch=8 |
| 优化器 | SGD，lr=0.01，momentum=0.9，weight decay=`1e-4`，poly power=0.9 |
| 辅助项 | 前 4000 optimizer steps 线性 warm-up |
| feature KD | A0 固定 StandardScaler+PCA，OS=4/8/16 等权 MSE，`lambda_feat=1.0` |
| logits KD | 全分辨率 masked pixel KL，`T=4`、`lambda_logit=0.5`，包含 `T²` |
| checkpoint 选择 | 训练固定跑满 80k，仅按 `dev_local` mIoU 选择最佳 checkpoint |

四组唯一变量是损失开关：K0=`CE`，K1=`CE+feature`，K2=`CE+logits`，K3=`CE+feature+logits`。K4 不属于该 scratch 2×2，而是从同 seed 的 A0 最优 probe checkpoint 初始化，完全解冻后按相同 80k SGD+poly 预算训练，相对 A0-FT 唯一增加 K2 的 logits KD；不使用在线 feature KD。所有运行的 `test_local_evaluated=false`。

---

## 3. 逐 seed 结果

| 实验 | seed | 最优 step | mIoU | mAcc | pixel Acc | small-object mIoU | boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K0 | 42 | 75129 | `0.51047` | `0.58437` | `0.89673` | `0.40202` | `0.39397` |
| K0 | 3407 | 75129 | `0.50480` | `0.58288` | `0.88727` | `0.40177` | `0.38877` |
| K0 | 260805 | 70057 | `0.50555` | `0.59609` | `0.88401` | `0.40358` | `0.37775` |
| K1 | 42 | 75129 | `0.52212` | `0.58929` | `0.91896` | `0.40833` | `0.44190` |
| K1 | 3407 | 80000 | `0.51980` | `0.59314` | `0.90631` | `0.40208` | `0.43145` |
| K1 | 260805 | 70057 | `0.52418` | `0.60395` | `0.90992` | `0.41892` | `0.43112` |
| K2 | 42 | 70057 | `0.55070` | `0.62164` | `0.92618` | `0.43913` | `0.43361` |
| K2 | 3407 | 70057 | `0.55665` | `0.65719` | `0.92103` | `0.46905` | `0.41979` |
| K2 | 260805 | 80000 | `0.54843` | `0.61981` | `0.91514` | `0.45019` | `0.42061` |
| K3 | 42 | 75129 | `0.55399` | `0.61967` | `0.92775` | `0.44928` | `0.44393` |
| K3 | 3407 | 80000 | `0.54929` | `0.61820` | `0.92330` | `0.43758` | `0.44111` |
| K3 | 260805 | 80000 | `0.54597` | `0.61836` | `0.92196` | `0.44396` | `0.44104` |
| K4 | 42 | 75129 | `0.56818` | `0.63233` | `0.93268` | `0.45496` | `0.46347` |
| K4 | 3407 | 80000 | `0.56618` | `0.63564` | `0.92757` | `0.45769` | `0.45825` |
| K4 | 260805 | 70057 | `0.57710` | `0.65825` | `0.92980` | `0.49994` | `0.45908` |

所有运行都实际完成 80k，表中的最优 step 只是 dev 选择结果。K2/K3/K4 各有部分 seed 在最终 step 最优，也有部分在 70057/75129 最优；当前证据不支持统一缩短或延长训练预算。

K3-G `seed=42` 同样完成 80k，最佳 step 为 `75129`，mIoU 为 `0.55065`。它与原 K3 `seed=42` 的 mIoU 相差 `-0.00334`，因此只作为机制诊断运行，不替代上表中的正式 K3 指标。

---

## 4. 三 seed 汇总与配对统计

### 4.1 主指标汇总

下表标准差为样本标准差（ddof=1，N=3）。

| 实验 | mIoU | mAcc | pixel Acc | small-object mIoU | boundary F1 |
|---|---:|---:|---:|---:|---:|
| K0 | `0.50694 ± 0.00308` | `0.58778 ± 0.00724` | `0.88934 ± 0.00661` | `0.40246 ± 0.00098` | `0.38683 ± 0.00828` |
| K1 | `0.52203 ± 0.00219` | `0.59546 ± 0.00760` | `0.91173 ± 0.00651` | `0.40978 ± 0.00851` | `0.43482 ± 0.00613` |
| **K2** | **`0.55193 ± 0.00425`** | **`0.63288 ± 0.02107`** | `0.92078 ± 0.00552` | **`0.45279 ± 0.01513`** | `0.42467 ± 0.00775` |
| K3 | `0.54975 ± 0.00403` | `0.61874 ± 0.00080` | **`0.92434 ± 0.00303`** | `0.44361 ± 0.00586` | **`0.44203 ± 0.00165`** |
| A0-FT（外部协议参考） | `0.55215 ± 0.01022` | `0.63121 ± 0.01170` | `0.91991 ± 0.00395` | `0.45261 ± 0.01245` | `0.45307 ± 0.01477` |
| **K4（A0 初始化+logits KD）** | **`0.57049 ± 0.00582`** | **`0.64207 ± 0.01410`** | **`0.93001 ± 0.00256`** | **`0.47086 ± 0.02522`** | **`0.46027 ± 0.00281`** |

在 scratch 2×2 内，K2 的平均 mAcc 和 small-object mIoU 最高，K3 的 pixel accuracy 最高且 boundary F1 的 seed 波动最小；A0-FT 与 K2/K3 的 mIoU 均值接近，但 seed 波动更大。K4 使用不同的分阶段初始化协议，不能作为 scratch 2×2 的第五个单元，但其 mIoU、mAcc、pixel accuracy 和 small-object mIoU 均值均为当前表中最高，boundary F1 也与 A0-FT 接近，因此是当前最佳 V2 候选。协议差异和指标取舍仍需同时报告，不能把 K4 的领先归因于单一 loss。

### 4.2 预注册配对差值

paired bootstrap 对每个 seed 的 445 张相同 dev 图像进行成对重采样，每次先聚合 19×19 confusion matrix 再计算 mIoU；使用 100,000 次重采样。所有逐图 confusion 聚合均以小于 `1e-15` 的误差复现对应 `metrics.json` mIoU。

| 对比 | 三个 seed 的 mIoU 差值 | 平均差值 | 每 seed 95% CI 结论 |
|---|---|---:|---|
| K1-K0 | `+0.01165/+0.01499/+0.01863` | `+0.01509` | 3/3 完全高于 0；区间端点总范围 `[+0.00089,+0.02750]` |
| K2-K0 | `+0.04024/+0.05185/+0.04288` | `+0.04499` | 3/3 完全高于 0；区间端点总范围 `[+0.03020,+0.06231]` |
| K3-K1 | `+0.03187/+0.02949/+0.02179` | `+0.02772` | 3/3 完全高于 0；区间端点总范围 `[+0.01306,+0.04172]` |
| K3-K2 | `+0.00328/-0.00736/-0.00246` | `-0.00218` | 0/3 排除 0；区间端点总范围 `[-0.01951,+0.01095]` |

因此，K1 和 K2 相对 K0 的独立收益均成立，且 logits 在 feature KD 之外仍有明确增量；反方向不成立，即当前 feature KD 没有在 logits KD 之上提供可测的 mIoU 增量。

### 4.3 A0-FT 与 K2/K3 的同 seed 描述性差值

下表是三个训练 seed 的最佳 dev mIoU 直接相减，不是逐图 paired-bootstrap 结果。A0-FT 与 K2/K3 的训练协议不同，因此这些差值用于判断稳定性，不用于把差异归因给单一 loss 或初始化因素。

| 对比 | seed=42 | seed=3407 | seed=260805 | 平均差值 |
|---|---:|---:|---:|---:|
| A0-FT - K2 | `+0.01325` | `-0.01028` | `-0.00228` | `+0.00023` |
| A0-FT - K3 | `+0.00996` | `-0.00293` | `+0.00018` | `+0.00241` |

两组差值均混合正负号。现有证据不支持“A0-FT 稳定高于在线 KD”，也不支持反向宣称 K2/K3 稳定高于 A0-FT；更准确的主指标结论是三者处于同一水平区间。

---

## 5. 为什么 K2 高于 K1，K3 又没有高于 K2

### 5.1 响应目标与分割任务更直接对齐

K2 的教师 logits 来自已训练并冻结的 19 类 R-ASPP，直接编码每个像素的类别分布、类间相似度和边界不确定性。其监督空间与最终 mIoU 目标一致。A0 feature MSE 则要求 MobileNetV2 在三个中间层逐点拟合经 PCA 压缩的 DINOv3 表征，其中仍包含纹理、几何和上下文等不一定直接服务于 19 类判别的信息。

这能解释 K2 相对 K0 的 mIoU 增益约为 K1 的三倍，但目前只能称为与数据一致的机制解释，不能仅由结果表证明所有中间特征都是任务无关噪声。

### 5.2 特征对齐存在层位和空间约束不均衡

A0 的 PCA 累计解释方差为 OS=4 `61.9%`、OS=8 `43.4%`、OS=16 `96.5%`。K 组末期 feature MSE 也始终呈现 `OS=8 > OS=16 > OS=4`：

| 实验 | OS=4 末期范围 | OS=8 末期范围 | OS=16 末期范围 | 三层均值末期范围 |
|---|---:|---:|---:|---:|
| K1 | `0.776-0.795` | `1.207-1.217` | `1.046-1.067` | `1.012-1.026` |
| K3 | `0.852-0.883` | `1.299-1.329` | `1.091-1.115` | `1.080-1.109` |

OS=8 同时具有最低 PCA 解释方差和最高残差，说明它最难拟合；但 K3-G 中 OS=8 的 feature 梯度与 CE/logits 保持正向对齐，且是后期较强的 feature 信号之一。因此高残差不能直接解释为冲突或有害约束，三层等权平均仍会隐藏层间不均衡，但删除 OS=8 的优先级已降低。

### 5.3 两种辅助项的优化作用不对称

K3-G 在 step `1/4000/20000/40000/60000/80000` 对固定审计 batch 记录了三层 tap 上的分项梯度范数和两两 cosine。每层每个 pair 共 12 个 per-rank 观测，全部有限且没有负值：

| tap | `cos(CE,feature)` 均值 | `cos(CE,logit)` 均值 | `cos(feature,logit)` 均值 | 主要解释 |
|---|---:|---:|---:|---|
| OS=4 | `0.473` | `0.772` | `0.566` | feature 与任务方向中等正对齐 |
| OS=8 | `0.480` | `0.775` | `0.575` | 与 OS=4 相近，不支持 OS=8 方向冲突 |
| OS=16 | `0.028` | `0.708` | `0.031` | feature 与 CE/logits 近正交，logits 仍与 CE 对齐 |

warm-up 后，feature 有效梯度约为同层 CE 的 `4%-13%`，logits 有效梯度约为 CE 的 `0.85-1.76×`；按全部 per-rank 观测，logits/feature 有效梯度比约为 `3.8-32.8×`，中位约 `13.5×`。对应地：

- K2 与 K3 的 logits KL 曲线在所有 seed 上几乎重合，说明加入 feature KD 没有明显改变响应拟合轨迹；
- K3 的 feature loss 比 K1 高约 `0.07-0.09`，说明加入 logits KD 后学生保留了更大的中间表示残差；
- K3≈K2 的 mIoU 与“logits 项主导、feature 项在主指标上边际作用较弱”一致。

K3-G 因此把机制解释收窄为：在被审计的固定 4 张图和 6 个模型状态上，没有观察到方向冲突；OS=4/8 feature 信号正向但较弱，OS=16 feature 信号较弱且近正交。它不能排除其他 batch、其他 seed 或参数梯度空间中的冲突，也不能单独证明 K3≈K2 的因果原因。

### 5.4 mIoU 不可加，但边界信息可能仍有补充

K3 相比 K2 没有 mIoU 增益，却稳定获得更高 boundary F1；K1 相比 K0 的 boundary F1 平均增益也达到 `+0.04800`。这表明固定多层特征约束可能更偏向局部结构或边界平滑，而 logits KD 更直接改善类别 IoU 和小目标均值。

因此最准确的表述是：**当前 feature KD 与 logits KD 在 mIoU 上不互补，但 feature KD 可能提供边界侧信息，同时牺牲部分 mAcc 和 small-object mIoU。** 是否属于真实结构互补，需要额外的边界/层位实验验证。

---

## 6. A0-FT 三 seed 后的协议比较

### 6.1 这是不同训练协议，不是同一 2×2 内的失败

| 项目 | A0-FT | K2/K3 |
|---|---|---|
| 初始化 | A0 最优 probe checkpoint | 同 seed 共同 scratch state |
| 监督前经历 | 40k 无标签 feature pretrain + 40k 冻结 head probe | 无 |
| 当前阶段 | 80k 纯 CE 全模型微调 | 80k CE+KD 从 scratch 联合训练 |
| backbone 累计更新 | 40k pretrain + 80k FT = 120k | 80k |
| KD 时序 | 先 feature KD，再移除 KD 做 CE | KD 与 CE 同时优化 |
| seed 数 | 3（`42/3407/260805`） | 3 |

A0-FT 三 seed mIoU 为 `0.56395/0.54637/0.54615`，均值 `0.55215 ± 0.01022`。其均值与 K2 的 `0.55193 ± 0.00425`、K3 的 `0.54975 ± 0.00403` 接近，但 A0-FT 的 seed 波动更大。原先只看 seed=42 得到的领先没有在另外两个 seed 上复现。

### 6.2 指标取舍仍然不同

A0-FT 与 K2 的平均 mAcc 和 small-object mIoU 几乎相同，但 A0-FT 的 boundary F1 比 K2 高 `+0.02840`；相对 K3，A0-FT 的平均 mAcc、small-object mIoU 和 boundary F1 分别高 `+0.01247/+0.00900/+0.01104`，pixel accuracy 则低 `-0.00444`。因此 A0-FT 的主要稳定优势不在 mIoU，而在边界及部分类别均衡指标。

### 6.3 K4 已完成：直接增量与交互量

K4 从同 seed 的 A0 最优 probe checkpoint 初始化，相对 A0-FT 只增加 K2 的 logits KD，因此可以估计 logits KD 在分阶段 feature 初始化之上的增量。三 seed 结果如下：

| seed | K4 mIoU | A0-FT mIoU | `Delta_s=K4-A0-FT` | `K2-K0` | `I_s=Delta_s-(K2-K0)` |
|---:|---:|---:|---:|---:|---:|
| 42 | `0.56818` | `0.56395` | `+0.00423` | `+0.04024` | `-0.03600` |
| 3407 | `0.56618` | `0.54637` | `+0.01981` | `+0.05185` | `-0.03204` |
| 260805 | `0.57710` | `0.54615` | `+0.03095` | `+0.04288` | `-0.01192` |
| **均值** | **`0.57049`** | **`0.55215`** | **`+0.01833`** | **`+0.04499`** | **`-0.02665`** |

三个 `Delta_s` 均为正，说明 logits KD 在 A0 初始化之上仍提供稳定的 mIoU 增量。三个 `I_s` 均为负，且 `|mean(I)|=0.02665>0.00425`，满足预注册的可测交互判据。交互方向为负，表示 A0 分阶段 feature 初始化与在线 logits KD 的收益部分重叠：logits KD 在 A0 初始化上的平均增量 `+0.01833`，小于其在 scratch 初始化上的平均增量 `+0.04499`。这不表示 logits KD 在 K4 中失效，也不能把差值解释为简单线性可加；准确结论是二者存在次加性交互。

K4 的平均 mAcc、pixel accuracy、small-object mIoU 和 boundary F1 分别为 `0.64207/0.93001/0.47086/0.46027`。相对 A0-FT，其 mIoU、mAcc、pixel accuracy、small-object mIoU 和 boundary F1 的平均差值分别为 `+0.01833/+0.01087/+0.01011/+0.01825/+0.00720`，但部分辅助指标的逐 seed 差值仍有混合正负号，因此不宣称所有指标均逐 seed 稳定提升。

---

## 7. 辅助损失维持非零说明什么

### 7.1 先修正时间与数值口径

每个 epoch 约 317 optimizer steps，20-40 epoch 仅对应约 step 6340-12680，占 80k 总预算的 8%-16%。warm-up 在 step 4000、约 epoch 13 结束。日志显示：

- feature loss：epoch 40 约为 K1 `1.34-1.36`、K3 `1.47-1.50`，到 epoch 253 继续降到约 `1.01-1.03/1.08-1.11`；
- logits KL：epoch 40 约为 `1.73-1.91`，到 epoch 253 继续降到约 `0.75-0.79`；
- CE：epoch 40 约为 `0.42-0.51`，最终约 `0.23-0.25`。

所以“epoch 20-40 后辅助项完全不再收敛、总损失下降只靠 CE”不成立。更准确的描述是：辅助项在中后期进入收益递减区，feature loss 在最终约 15 个 epoch 才基本变平，KL 到训练末期仍缓慢下降。

### 7.2 不同损失的绝对数值不能直接比较

CE、feature MSE 和 logits KL 使用不同的目标空间、reduction 和缩放：feature 是三层 BCHW MSE 的平均；logits 是有效像素上的类维求和 KL，并乘以 `T²=16`；总损失中 KL 又乘 `lambda_logit=0.5`。上述曲线范围都是未乘 lambda/warm-up 的 raw component loss；`training_history.json` 的 `train.loss` 实际记录 CE，真正参与优化的聚合值记录在 `total_loss_micro_batch_mean`。梯度日志中的分项范数同样是乘系数前的原始值，有效贡献需乘 `warmup×lambda`。因此“都在 1 附近”没有统一的收敛含义，也不能由绝对数值判断谁更重要。应联合观察：

1. 相对初值下降比例；
2. 各项对同一学生层的梯度范数和夹角；
3. dev mIoU、每类 IoU、boundary F1 的变化；
4. 去掉该项后的受控差值。

### 7.3 非零平台通常表示折中解，不等于蒸馏失败

学生容量小于教师、体系结构不同，并且 CE 与 KD 的最优点未必一致，因此 feature MSE 和 KL 没有理由收敛到 0。非零残差可能同时包含：

- 学生容量和有效秩限制下无法拟合的教师成分；
- PCA 后仍保留但与 Cityscapes 类别目标不完全一致的变化；
- hard labels、teacher logits 和中间表示之间的 Pareto 折中；
- poly LR 后期减小后剩余但难以继续消除的误差。

K2 在 KL 保持明显非零时仍比 K0 提升约 `0.045` mIoU，已经直接证明“KD loss 不趋近 0”与“KD 没有提供收益”不是同一件事。

相关机制与 dense prediction 中逐点约束过强的问题可参考 Hinton 等的经典 KD、Cho 与 Hariharan 的 teacher-student capacity gap、Liu 等的结构化分割蒸馏，以及 Shu 等的 Channel-wise Distillation：

- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*, 2015: <https://arxiv.org/abs/1503.02531>
- Cho, Hariharan, *On the Efficacy of Knowledge Distillation*, ICCV 2019: <https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html>
- Liu et al., *Structured Knowledge Distillation for Semantic Segmentation*, CVPR 2019: <https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.html>
- Shu et al., *Channel-Wise Knowledge Distillation for Dense Prediction*, ICCV 2021: <https://openaccess.thecvf.com/content/ICCV2021/html/Shu_Channel-Wise_Knowledge_Distillation_for_Dense_Prediction_ICCV_2021_paper.html>

---

## 8. K 组停止决定与后续实验

两个 P0 诊断项和唯一必须新增的 K4 因果单元均已完成。当前没有为了支撑 K 组结论而必须继续补充的实验：

| 优先级 | 实验 | 当前决定 | 唯一变量/范围 | 回答的问题 | 停止或报告条件 |
|---|---|---|---|---|---|
| P0 | A0-FT 扩展 `seed=3407/260805` | **已完成** | 只补 seed，其余保持现有协议 | A0-FT 是否稳定高于 K2/K3 | 三 seed 结果显示 mIoU 与 K2/K3 持平且 seed 差值混合正负号 |
| P0 | K3 梯度夹角审计 | **已完成** | K3-G `seed=42`，固定 4 张图、6 个 step、OS=4/8/16 | 已测状态下是否出现方向冲突，以及分项梯度尺度 | 未观察到负 cosine；feature 弱、OS=16 近正交；结论不得外推到全数据集或参数梯度 |
| P1 | K4：A0 初始化 + CE+logits KD | **已完成，停止** | 相对 A0-FT 只增加 K2 logits KD；已完成 `42/3407/260805` 三 seed | logits KD 在 A0 初始化上是否仍有增量，二者是否存在可测交互 | `Delta_s` 三 seed 均为正；`I_s` 三 seed 同为负且 `abs(mean(I))=0.02665>0.00425`，满足停止条件 |
| P1 | K3-noOS8 | **延期** | 若启动，相对 K3 只移除 OS=8 项，完成三 matched seeds | OS=8 是否造成 small-object/类别取舍 | 仅在层位归因成为明确目标时启动；以 small-object 为主指标，mIoU 均值非劣界为 `-0.00425`，三 seed 后停止 |
| P2 | K2/K3 单变量 logits 权重复验 | **延期** | 固定 `T=4`，比较 `lambda_logit=0.25/1.0`，K2/K3 等预算 | `0.5` 是否为候选集合中的局部最佳 | 仅在冻结最终 KD 配置前启动；先做 4 个 seed=42 运行，候选在 K2、K3 中均比 `0.5` 高 `>0.00425` 才扩另外两 seed |

当前推荐决定为：**停止继续扩展 K 组，进入 R 组实验。** K3-noOS8 和 logits 权重搜索保留为条件触发项，不纳入进入 R 组之前的前置条件。若后续需要冻结最终 KD 超参数，再单变量解锁 logits 权重；若层位归因成为论文核心问题，再解锁 K3-noOS8。

R 组按已登记顺序推进：先用 `R0 seed=42` 复现 K1 分支以排除代码分支差异，再运行 `R1 seed=42`，随后评估 `R2`；只有固定 seed 下出现 dev mIoU 增益、稳定训练曲线或明确机制证据时才扩展三 seed，`R3` 只组合已证明有效的关系项，`R4` 仅作欠约束诊断。GAN、MMD、KPCA 仍不进入当前主线，`test_local` 继续保持最终留出。

---

## 9. 当前状态与后续解锁

| 项目 | 状态 | 说明 |
|---|---|---|
| K0-K3 三 seed | 已完成 | 12 个运行均跑满 80k |
| 共享初始化/首 batch 审计 | 已通过 | 同 seed 的 K0-K3 可受控比较 |
| 教师/PCA/hash 链 | 已通过 | K1/K3 使用锁定 A0 投影，K1/K2/K3 使用锁定 T1 |
| best checkpoint 重载与逐图 confusion | 已完成 | 逐图聚合精确复现主指标 |
| paired bootstrap | 已完成 | K1-K0、K2-K0、K3-K1 显著；K3-K2 证据不足 |
| A0-FT 三 seed | 已完成 | mIoU `0.55215 ± 0.01022`；与 K2/K3 的同 seed 差值混合正负号 |
| K3-G 梯度夹角审计 | 已完成 | 6 个固定时点均已记录；固定 4 张图上无负 cosine，OS=16 feature 近正交 |
| K4 三 seed | 已完成 | 3 个运行均跑满 80k；mIoU `0.57049 ± 0.00582`，三个 `Delta_s` 均为正，三个 `I_s` 同为负并满足可测交互判据 |
| K 组阶段决定 | 已收敛 | 无必须补充实验；当前最佳 V2 候选为 K4，可以进入 R 组 |
| K3-noOS8 | 延期 | OS=8 未显示方向冲突；仅在需要层位因果归因时启动 |
| 温度/权重搜索 | 延期 | `lambda_logit=0.5` 已证明有效但未证明最优；在冻结最终配置前再决定 |
| Excel 人工汇总 | 已核对 | A0-FT、K0-K4 的已录入指标及 boundary/small-object 列均与 JSON 一致 |
| `test_local` | 未查看 | 继续保持最终留出 |

---

## 10. 结果文件与数据注意事项

- K0：`result/K_MobileNetV2_RASPP_server/K0/seed_*/metrics.json`
- K1：`result/K_MobileNetV2_RASPP_server/K1/seed_*/metrics.json`
- K2：`result/K_MobileNetV2_RASPP_server/K2/seed_*/metrics.json`
- K3：`result/K_MobileNetV2_RASPP_server/K3/seed_*/metrics.json`
- K4：`result/K_MobileNetV2_RASPP_server/K4/seed_*/metrics.json`
- K3-G 梯度夹角审计：`result/K_MobileNetV2_RASPP_server/K3-G/seed_42/gradient_norms.jsonl`
- K3-G 运行指标：`result/K_MobileNetV2_RASPP_server/K3-G/seed_42/metrics.json`
- 训练曲线：各目录下 `training_history.json`
- K0-K4 梯度范数日志：各目录下 `gradient_norms.jsonl`
- paired bootstrap 输入：各目录下 `dev_per_image_confusion.jsonl`
- 共享初始化：`result/K_MobileNetV2_RASPP_server/shared_init/`
- A0-FT 外部协议参考：`result/A_MobileNetV2_RASPP_server/A0-FT/seed_*/a0_ft_dev_metrics.json`
- 人工汇总：[实验数据.xlsx](../实验数据.xlsx)

人工 Excel 已完成本轮更新：K4 `seed=42/3407/260805` 已录入，A0-FT、K0-K4 的 `mIoU/mAcc/pixel accuracy/boundary F1@2px/small-object mIoU` 与对应 JSON 一致。后续仍以 result JSON 为权威来源，Excel 只作人工汇总。
