# Cityscapes 知识蒸馏 K 组训练部分总结

更新日期：2026-08-15

## 1. 总结结论

K0-K3 已按预注册协议完成 `seed=42/3407/260805` 三次训练，12 个运行均达到 80k optimizer steps，最佳 checkpoint、dev 指标、逐图 confusion matrix、训练历史和梯度日志完整。当前结论如下：

1. **响应蒸馏是 scratch 端到端训练中的主要增益来源。** K2 三 seed mIoU 为 `0.55193 ± 0.00425`，相对 K0 的配对增益为 `+0.04024/+0.05185/+0.04288`；三个 seed 的逐图 paired-bootstrap 95% CI 均完全高于 0。
2. **特征蒸馏单独使用也有效，但增益明显小于响应蒸馏。** K1 mIoU 为 `0.52203 ± 0.00219`，相对 K0 的平均增益为 `+0.01509`，三个 seed 的 paired-bootstrap 95% CI 同样均高于 0。因此不能把 K1 解释为无效，只能说其任务指标增益较弱。
3. **K3 与 K2 的 mIoU 性能相近，未证明两种知识在主指标上可加。** K3 mIoU 为 `0.54975 ± 0.00403`，相对 K2 的配对差值为 `+0.00328/-0.00736/-0.00246`，三次 95% CI 均跨 0。主指标排序应写为 `K2≈K3>K1>K0`，不应按单个 seed 宣称 K3 胜出。
4. **“不增加 mIoU”不等于“特征知识完全冗余”。** K3 相比 K2 的平均 boundary F1 提高 `+0.01736`，pixel accuracy 提高 `+0.00356`，但 mAcc 降低 `-0.01414`、small-object mIoU 降低 `-0.00919`。特征项更像改变了边界与类别均衡之间的取舍，而不是提供统一的 mIoU 增益。
5. **K2/K3 未超过 A0-FT 的原因目前无法由现有结果判断，因为训练协议和初始化不同。** A0-FT 从完成 40k 特征预训练和 40k 冻结 probe 的 checkpoint 出发，再做 80k CE 微调；K0-K3 均从共同 scratch state 直接训练 80k。A0-FT 的部署 backbone 在评估前累计获得 120k 次更新，且进入监督阶段前已经处于教师特征对齐的初始化。该差值与“初始化、训练时序和更新预算”的整个协议组合混杂，不能直接归因为“在线 KD 弱于 A0”。
6. **“20-40 epoch 后辅助损失都停在 1 附近”与日志不完全一致。** 20-40 epoch 只对应约 step 6340-12680；此后 feature loss 和 logits KL 仍持续下降。feature loss 的三层均值到训练末期约为 K1 `1.01-1.03`、K3 `1.08-1.11`，而带 `T²` 的原始 logits KL 最终约为 `0.75-0.79`，并非停在 1。
7. **现有证据更支持“表示约束较弱、任务约束占主导”，尚不能直接证明梯度冲突。** OS=16 tap 上 feature 梯度约比 CE 小 10-100 倍；logits 梯度则明显更强。K3 的 feature residual 高于 K1，但只有记录梯度夹角后，才能区分目标冲突、容量上限和单纯的梯度尺度不足。

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

四组唯一变量是损失开关：K0=`CE`，K1=`CE+feature`，K2=`CE+logits`，K3=`CE+feature+logits`。所有运行的 `test_local_evaluated=false`。

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

所有运行都实际完成 80k，表中的最优 step 只是 dev 选择结果。K2/K3 各有部分 seed 在最终 step 最优，也有部分在 70057/75129 最优；当前证据不支持统一缩短或延长训练预算。

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

K2 的平均 mIoU、mAcc 和 small-object mIoU 最高；K3 的 pixel accuracy 和 boundary F1 最高且 boundary F1 的 seed 波动最小。K2/K3 不应只保留一个总分排序，而应报告这种指标取舍。

### 4.2 预注册配对差值

paired bootstrap 对每个 seed 的 445 张相同 dev 图像进行成对重采样，每次先聚合 19×19 confusion matrix 再计算 mIoU；使用 100,000 次重采样。所有逐图 confusion 聚合均以小于 `1e-15` 的误差复现对应 `metrics.json` mIoU。

| 对比 | 三个 seed 的 mIoU 差值 | 平均差值 | 每 seed 95% CI 结论 |
|---|---|---:|---|
| K1-K0 | `+0.01165/+0.01499/+0.01863` | `+0.01509` | 3/3 完全高于 0；区间端点总范围 `[+0.00089,+0.02750]` |
| K2-K0 | `+0.04024/+0.05185/+0.04288` | `+0.04499` | 3/3 完全高于 0；区间端点总范围 `[+0.03020,+0.06231]` |
| K3-K1 | `+0.03187/+0.02949/+0.02179` | `+0.02772` | 3/3 完全高于 0；区间端点总范围 `[+0.01306,+0.04172]` |
| K3-K2 | `+0.00328/-0.00736/-0.00246` | `-0.00218` | 0/3 排除 0；区间端点总范围 `[-0.01951,+0.01095]` |

因此，K1 和 K2 相对 K0 的独立收益均成立，且 logits 在 feature KD 之外仍有明确增量；反方向不成立，即当前 feature KD 没有在 logits KD 之上提供可测的 mIoU 增量。

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

OS=8 同时具有最低 PCA 解释方差和最高残差，是当前 feature KD 最可疑的约束点。三层等权平均会隐藏这种不均衡。

### 5.3 两种辅助项的优化作用不对称

在共同的 student OS=16 tap 上，日志中的原始 feature 梯度约为 `4e-4-1.1e-3`，通常比 CE 小 10-100 倍；原始 logits 梯度在中期可达到 `6e-2-1.1e-1`，明显强于 feature 项。实际反向贡献还需分别乘以当时的 `warmup×lambda`；warm-up 结束后 feature 系数为 1.0，logits 系数为 0.5。对应地：

- K2 与 K3 的 logits KL 曲线在所有 seed 上几乎重合，说明加入 feature KD 没有明显改变响应拟合轨迹；
- K3 的 feature loss 比 K1 高约 `0.07-0.09`，说明加入 logits KD 后学生保留了更大的中间表示残差；
- K3≈K2 的 mIoU 与“logits 项主导、feature 项在主指标上边际作用较弱”一致。

这里仍不能直接下结论为“梯度冲突”，因为当前日志只有梯度范数，没有 `CE/feature/logits` 梯度的两两 cosine。K3 的较高 feature residual 也可能来自容量上限、梯度尺度不足或两者共同作用。

### 5.4 mIoU 不可加，但边界信息可能仍有补充

K3 相比 K2 没有 mIoU 增益，却稳定获得更高 boundary F1；K1 相比 K0 的 boundary F1 平均增益也达到 `+0.04800`。这表明固定多层特征约束可能更偏向局部结构或边界平滑，而 logits KD 更直接改善类别 IoU 和小目标均值。

因此最准确的表述是：**当前 feature KD 与 logits KD 在 mIoU 上不互补，但 feature KD 可能提供边界侧信息，同时牺牲部分 mAcc 和 small-object mIoU。** 是否属于真实结构互补，需要额外的边界/层位实验验证。

---

## 6. 为什么 K2/K3 没有超过 A0-FT

### 6.1 这是不同训练协议，不是同一 2×2 内的失败

| 项目 | A0-FT | K2/K3 |
|---|---|---|
| 初始化 | A0 最优 probe checkpoint | 同 seed 共同 scratch state |
| 监督前经历 | 40k 无标签 feature pretrain + 40k 冻结 head probe | 无 |
| 当前阶段 | 80k 纯 CE 全模型微调 | 80k CE+KD 从 scratch 联合训练 |
| backbone 累计更新 | 40k pretrain + 80k FT = 120k | 80k |
| KD 时序 | 先 feature KD，再移除 KD 做 CE | KD 与 CE 同时优化 |
| seed 数 | 1（seed=42） | 3 |

A0-FT `seed=42` mIoU 为 `0.56395`，比同 seed K2 高 `0.01325`、比 K3 高 `0.00996`。这个差距可能确实存在，但当前只能归因于整个“预训练初始化 + 分阶段优化 + 更长参数更新暴露”的组合，不能归因于某一个 KD loss。

### 6.2 分阶段训练可能避免持续表示约束

A0-FT 在进入监督微调后完全关闭 feature KD，允许 backbone 围绕 hard-label CE 自由适配；K3 则在 80k 全程保留 feature MSE。若 feature 目标与任务最优表示并不完全一致，A0-FT 可以先获得较好的教师初始化，再摆脱中间层逐点约束；K3 必须持续折中。这是合理假设，但需要同初始化的新增对照才能确认。

### 6.3 A0-FT 仍缺少三 seed

A0-FT 当前只有 seed=42，不能与 K2/K3 的三 seed 均值做正式显著性比较。虽然 seed=42 的差距大于 K3 的组内标准差，但这不足以替代 A0-FT 三 seed。K 组结论应以 K0-K3 的配对结果为主，A0-FT 只列为不同协议的外部参考。

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

## 8. 建议增加的最小实验集

不建议立即开展大网格。现阶段优先做能区分原因的少量实验：

| 优先级 | 新实验 | 唯一变量 | 回答的问题 | 停止条件 |
|---|---|---|---|---|
| P0 | A0-FT 扩展 `seed=3407/260805` | 只补 seed，其余保持现有协议 | A0-FT 是否稳定高于 K2/K3，而非 seed=42 偶然值 | 得到三 seed mean±std 和与 K2/K3 的同 seed 配对差值后停止 |
| P0 | K3 梯度夹角审计 | 不改训练目标，重跑一个 `seed=42` K3 并记录 CE/feature/logits 在 OS=4/8/16 的两两 cosine | K3 是目标冲突，还是 feature 梯度单纯过小 | 在预注册固定 batch 和 step 覆盖 warm-up 结束、中期、后期；跑满 80k 后停止 |
| P1 | K4：A0 初始化 + CE+logits KD | 相对 A0-FT 只增加 K2 的 logits KD | logits KD 能否在 A0 分阶段预训练初始化上继续增益并超过 A0-FT | 先跑 seed=42；若相对 A0-FT 增益小于当前 K seed 波动则不扩 seed |
| P1 | K3-noOS8 | 相对 K3 只把 OS=8 系数从 `1/3` 置 0，OS=4/16 仍各保持 `1/3` | OS=8 的低解释方差和高残差是否造成 mAcc/small-object 退步 | 先跑 seed=42；只有同时不降 mIoU 且改善类别/边界取舍才扩 seed |
| P2 | K2/K3 单变量 logits 权重复验 | 固定 `T=4`，只比较 `lambda_logit=0.25/1.0` | 当前 0.5 是否已接近合适尺度 | K2/K3 必须使用同一候选权重；先单 seed 筛选，不做 T×lambda 网格 |

推荐执行顺序为：**先补 A0-FT 三 seed 和梯度夹角审计，再做 K4；只有需要解释 K3 的边界/小目标取舍时才做 K3-noOS8，最后再考虑 logits 权重。**

不建议此时直接进入 GAN、MMD、KPCA 或更复杂关系损失。当前最重要的未知量是训练时序和多目标作用方式，不是缺少更多损失函数。

---

## 9. 当前状态与后续解锁

| 项目 | 状态 | 说明 |
|---|---|---|
| K0-K3 三 seed | 已完成 | 12 个运行均跑满 80k |
| 共享初始化/首 batch 审计 | 已通过 | 同 seed 的 K0-K3 可受控比较 |
| 教师/PCA/hash 链 | 已通过 | K1/K3 使用锁定 A0 投影，K1/K2/K3 使用锁定 T1 |
| best checkpoint 重载与逐图 confusion | 已完成 | 逐图聚合精确复现主指标 |
| paired bootstrap | 已完成 | K1-K0、K2-K0、K3-K1 显著；K3-K2 证据不足 |
| `test_local` | 未查看 | 继续保持最终留出 |
| 温度/权重搜索 | 已解锁但非最高优先级 | K2/K3 稳定受益；应先完成机制诊断 |
| A0-FT 三 seed | 待补 | 当前阻塞 A0-FT 与 K2/K3 的正式统计比较 |

---

## 10. 结果文件与数据注意事项

- K0：`result/K_MobileNetV2_RASPP_server/K0/seed_*/metrics.json`
- K1：`result/K_MobileNetV2_RASPP_server/K1/seed_*/metrics.json`
- K2：`result/K_MobileNetV2_RASPP_server/K2/seed_*/metrics.json`
- K3：`result/K_MobileNetV2_RASPP_server/K3/seed_*/metrics.json`
- 训练曲线：各目录下 `training_history.json`
- 梯度日志：各目录下 `gradient_norms.jsonl`
- paired bootstrap 输入：各目录下 `dev_per_image_confusion.jsonl`
- 共享初始化：`result/K_MobileNetV2_RASPP_server/shared_init/`
- A0-FT 外部参考：`result/A_MobileNetV2_RASPP_server/A0-FT/seed_42/a0_ft_dev_metrics.json`
- 人工汇总：[实验数据.xlsx](../实验数据.xlsx)

人工 Excel 的 `K组-有标签蒸馏` 工作表存在两处需修正：最后一列虽然写作 `boundary F1@2px`，当前 K0-K3 已录入数值实际对应 JSON 的 `small_object_mIoU`；K3 `seed=260805` 行尚未录入。本文的 boundary F1 均直接读取各运行 JSON 的 `boundary_f1` 字段，不使用该 Excel 列。
