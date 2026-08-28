# Cityscapes 知识蒸馏 H 组训练部分总结

更新日期：2026-08-26

## 1. 总结结论

H0 已完成 `seed=42/3407/260805` 三 seed，H1/H2/H3 仅完成 `seed=42` 筛选运行，`test_local` 均未查看。所有已完成运行均为 80,000 optimizer steps，结果 JSON 是本总结的权威来源，Excel 仅作交叉核对。

1. **H0 建立了 H 组内部的 ReLU6 matched anchor。** H0 三 seed mIoU 为 `0.560413 ± 0.002410`（样本标准差，ddof=1）。H0 是 H 组内部对照，不应直接等同于 R5 的 `seed=42` 单次结果 `0.573653`。
2. **H1/H2/H3 在 seed=42 均低于 H0。** H1 mIoU 为 `0.516091`，相对同 seed H0 下降 `0.047002`；H2 为 `0.523697`，下降 `0.039397`；H3 为 `0.522735`，下降 `0.040358`。三种 Hardswish 位置均未通过注册的 seed=42 筛选门，因此不扩展到 `seed=3407/260805`。
3. **退步不是单一 mIoU 数字波动。** H1 的 mAcc、pixel accuracy、small-object mIoU 和 boundary F1 均下降；H2 的 small-object mIoU 下降 `0.088311`，且 `bus` IoU 为 `0.055652`、`train` IoU 约为 `5.5e-5`，出现明显稀有类崩溃。该现象是机制观察，不足以单独证明原因。
4. **“教师模型的 ReLU6 导致退步”这一推测不成立。** 教师是 DINOv3 ConvNeXt-T，ConvNeXt block 使用 `nn.GELU()`；H0-H3 使用完全相同且冻结的教师 checkpoint。实验变量是学生 MobileNetV2 的 ReLU6 到 Hardswish 替换，不是教师激活替换。
5. **学生激活与蒸馏目标的几何失配仍是可检验的机制假设，但尚未被证明。** 首批 feature/logit loss、收敛时 loss 和梯度审计不能证明存在严重的特征分布失配；当前没有 activation statistics、feature norm、CKA 或独立分布距离证据。
6. **没有部署效率结论。** 四个 H 运行的 `efficiency.json` 均为 `enabled=false`、`result=null`，因此不能宣称 Hardswish 更快、更省内存或更易融合。
7. **H3 没有修复 H2 的精度问题。** H3 只替换后段 depthwise 激活，mIoU `0.522735`，与 H2 的差值仅 `-0.000962`，仍比 H0 低 `0.040358`。H3 未复现 H2 的 `bus/train` 近乎崩溃，但总体精度没有恢复。
8. **当前不继续 h-GELU/SiLU/GELU 激活网格。** H2 未通过 seed=42 门，H3 也未产生正向筛选信号；按 H 方案，H4-H6 不解锁。paired-bootstrap 已进一步确认 H1/H2/H3 相对 H0 的图像层差异均为负，因此没有足够理由继续堆叠未登记激活；仍可补部署效率和激活/特征分布诊断。

## 2. 固定训练协议

H0-H3 未列出的配置完全相同：

| 项目 | 固定值 |
|---|---|
| 数据划分 | `train_local/dev_local/test_local=2530/445/500`；`test_local` 未查看 |
| 数据清单组合哈希 | `033161572be28a6de295e0c5dfb62d83cd4d0a18b6039321347c58ab28b9d3c2` |
| 教师 | 冻结 T1 DINOv3 ConvNeXt-T + R-ASPP |
| 教师 checkpoint SHA-256 | `73cb1d3161c746d1b4ea30918ec6a1f0de5e3a4952c000cf85ddf95f3ccaddeb` |
| 学生 | MobileNetV2 + R-ASPP，`weights=None`，`output_stride=16` |
| 初始化 | K 组同 seed shared scratch initialization |
| seed=42 初始化 state SHA-256 | `262c283ee987c402d4e34e3fada682e8addd914c112e6923b25df7fe0bd58c0d` |
| PCA 参数记录 SHA-256 | `990a3e0645a522055f024d4fd7d22cad2624f057bff0ab88dd377d7d49fa345c` |
| 损失 | `CE + warmup * (1.0*feature + 0.3*R2 + 0.5*logit_KL_T2)` |
| logits 温度 | `T=4`，包含 `T^2` 因子 |
| 辅助项 warm-up | 前 4,000 optimizer steps 线性 warm-up |
| 训练预算 | 80,000 optimizer steps，effective global batch=8 |
| 优化器 | SGD，lr=`0.01`，momentum=`0.9`，weight decay=`1e-4`，poly power=`0.9` |
| checkpoint 选择 | 只按 `dev_local` mIoU 选择最佳 checkpoint |

H0-H3 的 seed=42 使用相同 shared initialization state；首批图像、标签和输入 hash 在两个 rank 上一致。H0 的严格 loss equivalence audit 通过；H1/H2/H3 按激活消融设计只比较 loader identity，不比较学生依赖的首批 loss 标量。

## 3. 激活替换实现审计

| 实验 | 学生激活 | 替换范围 | 替换数 | 已完成运行 |
|---|---|---|---:|---|
| H0 | 保持 ReLU6 | 原始 inverted-residual 站点 | 0 | `42/3407/260805` |
| H1 | ReLU6 -> Hardswish | blocks `1..17` 的 expansion/depthwise | 33 | `42` |
| H2 | ReLU6 -> Hardswish | blocks `14..17` 的 expansion/depthwise，OS=16 后段 | 8 | `42` |
| H3 | ReLU6 -> Hardswish | blocks `14..17` 的 depthwise-only | 4 | `42` |

Hardswish 实现为 `torch.nn.Hardswish(inplace=True)`，公式为 `x*ReLU6(x+3)/6`。所有 H 运行均保持：

- `backbone.0.2` stem、`backbone.18.2` final 1x1 和 `head.project.2` 的 ReLU6 不变；
- inverted residual 的 linear bottleneck `conv.2` 无激活，不被替换；
- block 数、stride/dilation、OS=4/8/16 feature taps、R-ASPP 和输出尺寸不变；
- 激活 reference test 通过，记录的 H1/H2/H3 replacement count 分别为 33、8 和 4。

因此 H1-H3 的实现边界与 H 方案登记一致，不能把当前退步归因于误替换 stem、projection 或 linear bottleneck。

## 4. 教师激活与特征空间问题

### 4.1 教师并不是 ReLU6 模型

`dino_t1.py` 将教师定义为 DINOv3 ConvNeXt-T + R-ASPP；`dinov3-main/dinov3/models/convnext.py` 的 ConvNeXt block 激活为 `nn.GELU()`。教师在 H0-H3 中均加载同一 checkpoint，执行 `eval()`、冻结参数，并在 `torch.no_grad()` 下生成 OS=4/8/16 特征和 pixel logits。教师 R-ASPP head 的 project 分支虽然保留 ReLU6，但它在四个运行中完全相同，也不是 H 组变量。

### 4.2 对“教师激活不一致”的结论

不能说“教师仍为 ReLU6，导致 H1/H2 退步”。现有代码和产物支持的严格结论是：

1. 教师 backbone 是 GELU ConvNeXt，而不是 ReLU6 MobileNetV2；
2. H0-H3 的 teacher checkpoint、teacher 输出接口、PCA、R2 target、logit target 和 loss 权重完全相同；
3. seed=42 的 matched 比较中，唯一模型变量是学生 ReLU6 -> Hardswish 的替换位置；
4. Hardswish 改变了学生激活值域和反向导数，可能使学生同时拟合固定 PCA feature target、native R2 relation target 和 pixel-logit target 的优化几何发生变化；
5. 这只是与结果相容的机制假设，不是已经证明的因果机制。

首批审计中 H0/H1/H2/H3 的 feature loss 分别约为 `4.4053/4.4200/4.4057/4.4054`，logit loss 约为 `21.4714/21.4193/21.4507/21.4614`；收敛时各项 loss 也处于相近量级。这不支持“蒸馏目标完全无法匹配”的说法。要验证特征空间假设，后续应单独登记并记录学生/教师 projected target 的逐层均值、标准差、能量、激活分位数和 CKA/MMD/SWD 等统计，不能仅凭 mIoU 退步反推原因。

不得据此修改教师激活、教师架构或教师 checkpoint；当前证据不支持这些干预。

## 5. seed=42 结果

以下均为最佳 `dev_local` checkpoint，不能视为三 seed 均值：

| 实验 | best step | mIoU | mAcc | pixel Acc | small-object mIoU | boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 75129 | `0.563093` | `0.637902` | `0.928136` | `0.465712` | `0.448784` |
| H1 | 55158 | `0.516091` | `0.592205` | `0.909467` | `0.407219` | `0.416976` |
| H2 | 75129 | `0.523697` | `0.587776` | `0.924252` | `0.377401` | `0.446227` |
| H3 | 55158 | `0.522735` | `0.585036` | `0.923480` | `0.395691` | `0.423684` |

相对同 seed H0 的描述性差值：

| 比较 | mIoU | mAcc | pixel Acc | small-object mIoU | boundary F1 |
|---|---:|---:|---:|---:|---:|
| H1-H0 | `-0.047002` | `-0.045697` | `-0.018669` | `-0.058493` | `-0.031807` |
| H2-H0 | `-0.039397` | `-0.050126` | `-0.003884` | `-0.088311` | `-0.002556` |
| H3-H0 | `-0.040358` | `-0.052867` | `-0.004656` | `-0.070021` | `-0.025099` |

H1/H3 的 best checkpoint 均在 step `55158`，最终 step=80000 的 dev mIoU 分别约为 `0.470511/0.503081`；H2 最终约为 `0.513372`。这显示 H1/H3 存在后期 dev 退化，但不能仅凭训练曲线区分优化不稳定、泛化退化或 checkpoint 选择偶然性。H2 的主要异常集中在稀有类别：`bus` IoU=`0.055652`、`train` IoU=`0.0000549`；H3 对应为 `0.183852/0.289575`，没有复现该类别崩溃。

## 6. H0 三 seed 锚点

标准差为样本标准差（ddof=1）：

| 实验 | mIoU | mAcc | pixel Acc | small-object mIoU | boundary F1 |
|---|---:|---:|---:|---:|---:|
| H0 | `0.560413 ± 0.002410` | `0.634399 ± 0.003766` | `0.926978 ± 0.001146` | `0.466036 ± 0.001349` | `0.446009 ± 0.002474` |

H0 的三 seed 方差只能描述 ReLU6 anchor，不能替 H1-H3 构造不存在的三 seed 方差。H1-H3 仍只能写成 seed=42 screening result，不能写成“稳定劣于 H0”。

## 7. 梯度审计

`gradient_norms.jsonl` 在 OS=16 student tap 记录了 total auxiliary effective gradient 与 CE gradient 的比例。固定审计点为 `1/4000/20000/40000/60000/80000`：

| 实验 | 1 | 4000 | 20000 | 40000 | 60000 | 80000 |
|---|---:|---:|---:|---:|---:|---:|
| H0 seed=42 | `0.000391` | `1.017334` | `0.976287` | `0.950615` | `1.355680` | `1.038985` |
| H0 seed=3407 | `0.000409` | `1.622048` | `5.073445` | `1.280359` | `0.979669` | `2.091224` |
| H0 seed=260805 | `0.000380` | `1.235942` | `1.037008` | `0.975355` | `0.978899` | `1.827540` |
| H1 seed=42 | `0.000403` | `1.091285` | `0.988531` | `1.078988` | `1.500201` | `1.076862` |
| H2 seed=42 | `0.000396` | `1.057271` | `0.903790` | `1.061518` | `1.481661` | `1.098459` |
| H3 seed=42 | `0.000393` | `1.167762` | `1.033977` | `1.244597` | `1.306317` | `1.192669` |

所有运行均 finite。停止门是 total auxiliary/CE 连续 3 次超过 `2.0`；虽然 H0 seed=3407 和个别运行存在单点超过 2 的记录，但没有运行触发连续三次停止门。约 step=58000 时，四个 H 运行都出现 CE 接近 0、导致 total auxiliary/CE 瞬时尖峰；H3 最高约 `135.7`，但 H0/H1/H2 也出现同类尖峰，且任一运行的连续超阈值记录均未达到 3 次。因此这更像公共训练/指标比值 artifact，而不是 H3 特有的激活爆炸。H1-H3 的梯度尺度不能证明或排除教师/学生激活机制。

## 8. 配对统计与部署证据边界

- H0/H1/H2/H3 seed=42 均保存了 445 张 `dev_local` 图像的 `dev_per_image_confusion.jsonl`，已用于本次 H-PB。
- H-PB 已完成。每个比较使用相同 445 张 `dev_local` 图像、聚合 19x19 confusion matrix、100,000 次配对重采样和 bootstrap random seed `260820`；训练 seed 没有混合。

| 比较 | checkpoint delta | bootstrap delta mean | bootstrap std | 95% CI | CI 是否排除 0 |
|---|---:|---:|---:|---|---|
| H1-H0 | `-0.047002` | `-0.046488` | `0.005552` | `[-0.057515,-0.035691]` | 是，支持负向 |
| H2-H0 | `-0.039397` | `-0.039290` | `0.005284` | `[-0.049826,-0.029048]` | 是，支持负向 |
| H3-H0 | `-0.040358` | `-0.040424` | `0.006333` | `[-0.053516,-0.028708]` | 是，支持负向 |
| H2-H1 | `+0.007606` | `+0.007198` | `0.005315` | `[-0.003525,+0.017354]` | 否，跨 0 |
| H3-H2 | `-0.000962` | `-0.001134` | `0.004837` | `[-0.010746,+0.008299]` | 否，跨 0 |

H1/H2/H3 相对 H0 的 paired-bootstrap CI 均完全低于 0，支持它们在这 445 张 dev 图像上低于 H0；这是固定 checkpoint 的图像层证据，不能替代训练 seed 稳定性。H2-H1 和 H3-H2 的 CI 均跨 0，不能据此宣称后段 expansion+depthwise 与 depthwise-only 存在可靠差异。

paired bootstrap 只量化固定 checkpoint 在 445 张 dev 图像上的重采样不确定性，不能替代 H1-H3 的训练 seed 扩展。H-PB 的 combined bootstrap 保持未计算，这是为了不混合训练 seed 不确定性与图像重采样不确定性。
- 四个 H 运行的 `efficiency.json` 均为 `enabled=false`、`result=null`，没有 latency、MACs、peak memory、ONNX/TensorRT/TFLite export 或 operator fusion 结果。因此不能作任何精度-效率 Pareto 结论。

## 9. MobileNetV3 论文对照

MobileNetV3 论文中的证据不能直接解释为“在任意 MobileNetV2 分割蒸馏实验中把 ReLU6 换成 Hardswish 就会提升精度”：

1. 论文 Table 5 的 h-swish 消融是 **MobileNetV3-Large 的 ImageNet 分类**，并非本项目的 MobileNetV2+R-ASPP Cityscapes 蒸馏。表中 ReLU 为 `74.5`，h-swish@16 为 `75.4`，h-swish@112 为 `75.0`；这是特定 V3-Large 架构和分类训练协议下的结果。
2. MobileNetV3 不是只替换激活函数。论文同时使用了 platform-aware NAS、NetAdapt、SE、重新设计的 block/channel、训练和部署优化；表 1/2 还明确显示 h-swish 只放在较深的部分，浅层仍使用 ReLU。论文第 5.2 节解释，深层激活的空间开销更小，且收益主要在深层实现。
3. 论文 Table 7 的 Cityscapes 分割结果反而说明，V3 的优势不能简单归因于 h-swish：在相同表格设置下，V2+R-ASPP 为 `72.84`，V3+R-ASPP 为 `72.64`；使用 LR-ASPP 后 V2 为 `72.97`，V3 为 `72.37`。论文报告的 V3 分割优势来自完整 backbone、通道调整、OS/atrous 设置和 LR-ASPP 等组合，而不是 h-swish 的独立因果效果。
4. 本项目还额外加入了冻结 DINOv3 教师、固定 StandardScaler+PCA feature KD、native R2 relation KD 和 pixel-logit KD。学生激活必须同时适应这些固定目标；这与论文的 scratch supervised training 不是同一个优化问题。

因此，H 组与 MobileNetV3 结果不矛盾。更准确的解释是：h-swish 的收益依赖于架构、激活位置、任务、训练协议、特征接口和部署实现；本项目的 matched seed=42 结果只支持“当前 V2+R-ASPP+R5 蒸馏协议下，H1-H3 三种 h-swish 位置均未显示精度收益”。论文来源为 Howard 等，*Searching for MobileNetV3*，arXiv:1905.02244，Section 5.2、Table 5、Section 6.4、Table 7。

## 10. 当前结论能回答什么

**可以回答：**

- 在锁定的 R5-compatible protocol、相同 seed=42 初始化和相同数据流下，H1 全量、H2 后段 expansion+depthwise、H3 后段 depthwise-only 的最佳 dev mIoU 均低于 H0 ReLU6 anchor；
- H1/H2/H3 的 seed=42 运行均未达到继续扩展的筛选门；
- H2 的稀有类和 small-object 指标出现明显异常，值得作为后续诊断目标；
- H3 与 H2 的 mIoU 基本相同，且 H3 未复现 H2 的 bus/train 崩溃；这说明 H2 的稀有类崩溃不能简单归因于“加入 expansion Hardswish”这一单一因素；
- H-PB 在 445 张 dev 图像上的 CI 支持 H1/H2/H3 均低于 H0；H2-H1 与 H3-H2 的 CI 跨 0，因此没有图像层证据证明两种后段放置存在可靠总体差异；
- 教师不是“ReLU6 教师”，而是冻结的 DINOv3 ConvNeXt-T，教师激活不是当前 H1-H3 的实验变量；
- MobileNetV3 论文的 h-swish 结果属于完整 V3 架构和特定任务/训练协议，不能作为本项目 h-swish 必然有效的依据。

**不能回答：**

- H1/H2/H3 是否跨随机 seed 稳定劣于 H0；
- H1/H2/H3 哪一个具有普遍更好的位置效果；
- 退步是否由 teacher/student activation mismatch 的某个具体统计量造成；
- 修改教师激活或架构是否能修复退步；
- Hardswish 在移动设备上是否更快或更容易融合；
- H3 与 H2 的小幅差异是否能推广到训练 seed；当前只有一个 seed，paired-bootstrap CI 虽已跨 0，但不能替代多 seed 证据。

## 11. 后续优先级与停止条件

| 优先级 | 工作 | 当前决定 |
|---|---|---|
| P0 | 冻结 H0-H3 seed=42 筛选结论 | H1-H3 不扩展三 seed；不把单 seed 写成稳定结论 |
| P1 | H3 三 seed扩展 | 不扩展；H2 未通过门，H3 与 H2 基本相同且均低于 H0 |
| P1 | H paired bootstrap | 已完成；CI 支持 H1/H2/H3 低于 H0，但不替代训练 seed 证据 |
| P2 | H0-H3 部署测量 | 统一设备、精度、输入和 warm-up 后测 latency、memory、MACs、export/fusion；测量前不作效率结论 |
| P2 | 激活/特征分布诊断 | 另行登记统计实验，记录 projected teacher 与 student 各层分布；不修改当前教师以“修复”结果 |
| P3 | H4-H6：SiLU/GELU/h-GELU | 暂不解锁；H2 未通过精度与效率筛选，H3 也未形成正向位置证据 |

若后续研究目标是正式比较 H1-H3 的位置因果效应，则必须先重新登记并让候选同时完成 matched `seed=3407/260805`，再报告三 seed 均值、逐 seed 差值和 paired bootstrap。按当前 H 方案的解锁条件，H 组应停止激活函数网格，结论停留在“seed=42 负向筛选信号，未发现 Hardswish 增益”；h-GELU 不进入下一轮。

## 12. 当前状态与数据注意事项

| 项目 | 状态 | 说明 |
|---|---|---|
| H0 ReLU6 anchor | 已完成三 seed | `0.560413 ± 0.002410` mIoU |
| H1 all-in-block Hardswish | 已完成 seed=42 | `0.516091` mIoU；未扩展 |
| H2 blocks 14..17 Hardswish | 已完成 seed=42 | `0.523697` mIoU；未扩展 |
| H3 depthwise-only | 已完成 seed=42 | `0.522735` mIoU；与 H2 相近且低于 H0；未扩展 |
| H paired bootstrap | 已完成 | 5 个 seed=42 比较；候选-H0 CI 均为负且排除 0，H2-H1/H3-H2 CI 跨 0 |
| efficiency/export | 未测量 | 四个 H 运行均为 `enabled=false`、`result=null` |
| `test_local` | 未查看 | 所有 H metrics 为 `test_local_evaluated=false` |
| Excel | 已交叉核对 | H 页数值与 JSON 一致；JSON 仍为权威来源 |

主要证据位于：

- `result/H_MobileNetV2_RASPP_server/H0/seed_{42,3407,260805}/`；
- `result/H_MobileNetV2_RASPP_server/H1/seed_42/`；
- `result/H_MobileNetV2_RASPP_server/H2/seed_42/`；
- `result/H_MobileNetV2_RASPP_server/H3/seed_42/`；
- `dino_h0_server.py`、`dino_h1_server.py`、`dino_h2_server.py`；
- `dino_h3_server.py`；
- `dinov3-main/dinov3/models/convnext.py`；
- `实验数据.xlsx` 的 H 组激活函数页。
