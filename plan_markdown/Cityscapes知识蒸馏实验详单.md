# Cityscapes 知识蒸馏实验详单

版本：2026-08-15

本表是执行清单；实验依据与结果解释见根目录 `知识蒸馏实验分析与后续实验方向.md`。

## 1. 全实验固定协议

| 项目 | 固定要求 |
|---|---|
| 数据 | `train_local=2530`：官方 train 排除 darmstadt/jena/krefeld/weimar；`dev_local=445`：上述 4 城市；`test_local=500`：官方 val；官方 test 1525 张仅作网站提交 |
| 标签 | 官方 `labelIds -> trainIds 0..18`，其余为 `ignore_index=255`；不用 color mask，不用 test 占位 mask 评分 |
| 输入 | train：scale `[0.5,2.0]`、crop `512×1024`、水平翻转；dev/test：`1024×2048` 单尺度、无翻转 |
| 模型 | 教师 DINOv3 ConvNeXt-T+R-ASPP；学生 V2+R-ASPP；学生 V3-Small+LR-ASPP |
| 特征 | 全模型 `output_stride=16`；固定对齐 OS=4/8/16；teacher/student 共用同一几何增强 |
| 监督 | 19 类 pixel CE；主表不加入 Dice、Focal、辅助头、多尺度或 flip inference |
| KD 默认 | `T=4`、`lambda_logit=0.5`、`lambda_feat=1.0`；辅助项前 5% step 线性 warm-up |
| 预算 | scratch 监督训练 80k steps；A 组无标签预训练 40k+冻结 head probe 40k；global batch 建议 8，显存不足用梯度累积 |
| 优化 | 学生首选 SGD+poly LR；教师首选 AdamW 分组 LR；只允许在 P0 预跑调整一次，锁定后不得因实验而变 |
| 随机性 | 筛选 seed `42`；确认 seed `42/3407/260805`；同组固定 batch 顺序和初始化 |
| 指标 | 19 类 mIoU 主指标；另报每类 IoU、mAcc、pixel Acc、boundary F1、参数量、MACs、显存和 batch=1 延迟 |
| 选择 | 只按 dev mIoU 选 checkpoint；test_local 在方案冻结前不可查看；训练 adapter 在效率测试前移除 |

## 2. 实验总表

表中只写每组相对固定协议的变量；未写出的配置必须完全相同。

| 阶段 | 编号 | 模型/目标 | 本次实验的唯一要求 | 完成条件与产物 |
|---|---|---|---|---|
| 准备 | P0-1 | 数据清单 | 按完整城市生成 2530/445/500 清单并查重 | 3 个 txt、规则、类别统计和 SHA-256 |
| 准备 | P0-2 | 标签 | 实现 `labelIds -> 0..18/255` | 可视化 20 张；mask 只含 0..18/255 |
| 准备 | P0-3 | 指标 | 单测 ignore、resize 和 confusion matrix | 完美预测 mIoU=1；输出 19×19 矩阵 |
| 准备 | P0-4 | 特征 taps | 记录三模型 OS=4/8/16 模块名、通道和 H×W | `feature_taps.json`；同 crop 可空间对齐 |
| 准备 | P0-5 | 分割头 | DINO/V2 用 R-ASPP，V3-Small 用 LR-ASPP；均输出 19 类 | 参数量及前向/反向 smoke test |
| 准备 | P0-6 | 训练管线 | 测试确定性、保存和 resume | 同 seed 首批 loss 一致；resume 后 step/LR 正确 |
| 准备 | P0-7 | PCA 管线 | 只读 train；每图每层≤128 token、总量≤200k/层 | 抽样清单、Scaler/PCA、解释方差和哈希 |
| 基线 | T0 | DINOv3+R-ASPP | 冻结 backbone，只训练 head | 3 seed；冻结 DINO 分割基线 |
| 基线 | T1 | DINOv3+R-ASPP | head warm-up 后只解冻最后 stage | 3 seed；从 T0/T1 仅按 dev 锁定一个教师及哈希 |
| 基线 | S2-F | 随机 V2+R-ASPP | `weights=None`，冻结 backbone，只训练 head | 1 seed；A 组 probe 下界 |
| 基线 | S2-0 | V2+R-ASPP | `weights=None`，端到端 CE，无 KD | 3 seed；V2 主基线和效率指标 |
| 基线 | S3-0 | V3-Small+LR-ASPP | `weights=None`，端到端 CE，无 KD | 3 seed；V3 主基线和效率指标 |
| 基线 | S2-P | V2+R-ASPP | 唯一变化为 ImageNette2-320 预训练初始化 | 3 seed；与 scratch 分表报告；标记为资源受限预训练基线 |
| 基线 | S3-P | V3-Small+LR-ASPP | 唯一变化为 ImageNette2-320 预训练初始化 | 3 seed；与 scratch 分表报告；标记为资源受限预训练基线 |
| 投影 | A0 | 固定 PCA，T→S | 每层 StandardScaler+PCA 到学生原生通道；无标签预训练 | seed 42；probe mIoU、稀有类 IoU、CKA |
| 投影 | A0-FT | A0 端到端微调 | 从 A0 最优 probe checkpoint（backbone+head）出发，解冻整个 backbone，按 S2-0 的 80k step、SGD+poly、端到端 pixel CE 微调；不加任何 KD 项 | seed 42；fine-tune mIoU；与 S2-0 scratch 基线和 A0 probe 对比；记录源 probe checkpoint 与初始/最终 backbone 哈希 |
| 投影 | A1 | 固定 1×1 Conv，T→S | 将 A0 的 Scaler+PCA 精确写入 Conv | 训练前各层 `max_abs_error<=1e-5`；指标应等价 A0 |
| 投影 | A1-FT | A1 端到端微调 | 从 A1 最优 probe checkpoint 出发，其余与 A0-FT 完全相同 | seed 42；fine-tune mIoU；数值上应与 A0-FT 接近（A1 骨干等价于 A0） |
| 投影 | A2 | 可学习 PCA-Conv，T→S | A1 初始化，adapter LR=学生 LR×0.1 | seed 42；判断固定锚点是否关键 |
| 投影 | A3 | 固定随机正交，T→S | 保留 StandardScaler，仅把 PCA 换为固定随机正交投影 | seed 42；保存正交误差；判断主方向作用 |
| 投影 | A4 | 无 Scaler 的 PCA，T→S | 相对 A0 只取消 StandardScaler | seed 42；判断尺度校准作用 |
| 投影 | A5 | 共享瓶颈 | 教师固定 PCA→`d_l=C_s`；学生可学习 1×1 Conv→同维度 | seed 42；推理移除 adapter；判断学生坐标自由度 |
| 投影 | A5-FT | A5 端到端微调 | 从 A5 最优 probe checkpoint（adapter 已在 probe 前移除）出发，其余与 A0-FT 完全相同 | seed 42；fine-tune mIoU；当前单次 probe 最优候选的端到端结果 |
| 投影 | A6 | 完整教师空间，S→T | 无 PCA；学生 1×1 Conv 从 `C_s` 扩到 `C_t` | seed 42；判断完整教师空间是否过难 |
| 知识 | K0 | V2 监督基线 | `L=L_seg`，受控重跑 S2-0 | 3 seed；与 K1-K3 共用初始化和 batch 顺序 |
| 知识 | K1 | 仅 feature KD | `L=L_seg+lambda_feat*L_feat`，使用唯一 `A_best`，不用 logits | 3 seed；报告 `K1-K0` |
| 知识 | K2 | 仅 logits KD | `L=L_seg+lambda_logit*L_logit`，教师 head 已训练并冻结 | 3 seed；报告 `K2-K0` |
| 知识 | K3 | feature+logits KD | 合并 K1/K2，权重不变 | 3 seed；报告 `K3-K1` 和 `K3-K2` |
| 知识 | K3-G | K3 梯度夹角审计 | 不改变 K3 的初始化、训练目标或权重；重跑 `seed=42`，在预注册的固定 batch 和 step 上记录 CE/feature/logits 对 OS=4/8/16 学生特征的两两 gradient cosine | 跑满 80k；覆盖 warm-up 结束、中期和后期；输出各层 `cos(CE,feat)`、`cos(CE,logit)`、`cos(feat,logit)`，区分目标冲突与 feature 梯度尺度过小 |
| 知识 | K4 | A0 初始化 + logits KD | 从 A0 `seed=42` 最优 probe checkpoint（backbone+head）出发，按 A0-FT 的 80k CE 微调协议训练；唯一新增项为 K2 的 logits KD（`T=4`、`lambda_logit=0.5`、前 4000 step warm-up），不使用 feature KD | 首轮 seed 42；报告 `K4-A0-FT` 和 `K4-K2`；若相对 A0-FT 的增益小于当前 K 组 seed 波动则停止，否则扩展 3 seed |
| 关系 | R0 | K1 复现 | CE+`A_best` feature KD，无关系项 | seed 42；排除代码分支差异 |
| 关系 | R1 | 图像关系 | R0+masked GAP 后的 batch `B×B` cosine matrix MSE | seed 42；记录关系梯度和 batch 大小 |
| 关系 | R2 | 空间关系 | R0+池化至 8×16 后的 `128×128` token 关系 | seed 42；重点报告边界/小目标和显存 |
| 关系 | R3 | 两种关系 | R0+R1+R2；各自 `lambda_rel=0.03` 起步并 warm-up | seed 42；关系梯度为 feature 梯度的 5%–20% |
| 关系 | R4 | relation-only | CE+关系项，移除逐点 feature MSE | 1 seed，仅诊断欠约束，不作主候选 |
| 分布 | D0 | K1 复现 | 无新增分布项 | seed 42；分布实验锚点 |
| 分布 | D1 | CORAL | K1+共同瓶颈均值/协方差对齐 | seed 42；有效才扩 3 seed |
| 分布 | D2 | MMD/SWD | K1+MMD 或 SWD，首轮只选一种 | seed 42；记录带宽或投影数 |
| 分布 | D3 | 对抗辅助 | K1+池化瓶颈小判别器，必须保留 feature MSE | 3 seed；若方差增大或判别器饱和则停止 |
| 分布 | D4 | adversarial-only | CE+对抗项，移除 feature MSE | 1 seed，仅诊断，不作主候选 |
| 激活 | H0 | V2 ReLU6 | 在最佳 K/R 协议上保留原激活 | seed 42；mIoU 与 FP32/INT8 延迟 |
| 激活 | H1 | V2 Hardswish | 相对 H0 只换 inverted residual 内激活 | seed 42；mIoU 与 FP32/INT8 延迟 |
| 激活 | H2 | V2 SiLU | 相对 H0 只换激活；linear bottleneck 仍无激活 | seed 42；mIoU 与 FP32/INT8 延迟 |
| 激活 | H3 | V2 h-GELU | 相对 H0 只换激活；保存公式与部署算子图 | seed 42；精度、延迟及算子可融合性 |
| V3复验 | M3-A0 | V3 固定 PCA | 按 V3 通道重新拟合 A0，禁止复用 V2 PCA | 3 seed；无标签 probe mIoU |
| V3复验 | M3-Abest | V3 最佳投影 | 迁移 V2 的投影机制，但所有维度/PCA 按 V3 重建 | 3 seed；检验跨学生可迁移性 |
| V3复验 | M3-K0 | V3 CE-only | 受控重跑 S3-0 | 3 seed；V3 2×2 基线 |
| V3复验 | M3-K1 | V3 feature KD | CE+M3-Abest feature KD | 3 seed；报告相对 M3-K0 增益 |
| V3复验 | M3-K2 | V3 logits KD | CE+同一冻结教师 pixel-logit KD | 3 seed；报告相对 M3-K0 增益 |
| V3复验 | M3-K3 | V3 feature+logits | 合并 M3-K1/K2，权重不变 | 3 seed；同时报告 R-ASPP/LR-ASPP 与效率差异 |

A0、A5、A6 和 seed 42 的另一领先 A 组需扩展到 3 seed，并补做统一端到端 fine-tune。A0-FT/A1-FT/A5-FT 是 A 组的端到端微调对照：它们从对应的最优 probe checkpoint 出发（A5 在 probe 前已移除训练期 adapter），解冻整个 backbone，用与 S2-0 完全一致的 80k step、SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)、poly(power=0.9) 和相同增强做端到端 pixel CE，不加任何 KD 损失。它们回答 A 组的核心问题：无标签特征预训练得到的 backbone，经监督微调后能否超过 S2-0 scratch 基线。首轮只做 seed=42，若明显优于或劣于 S2-0，再随 A0/A5 的三 seed 一并扩展。FT 阶段的 checkpoint 只按 dev mIoU 选择，效率指标沿用 probe 阶段口径。K3-G 只增加梯度夹角日志，不改变 K3 的科学定义；K4 使用 A0 probe 初始化并只加入 logits KD，用于分离“分阶段特征预训练初始化”和“在线响应蒸馏”的贡献。K2/K3 稳定有效后，才可单变量搜索 `T={2,8}` 或 `lambda_logit={0.25,1.0}`。D1/D2 无法改善时才做 D3；KPCA、自定义学生和复杂几何损失不进入第一轮。

## 3. 执行与解锁顺序

推荐顺序：

`P0 -> T0/T1 -> S2-F/S2-0/S3-0 -> A0-A6 -> K0-K3 -> K3-G -> K4 -> R1/R2 -> M3复验 -> H/D -> test_local`

`test_local` 是锁定的最终留出集（官方 val 的 500 张图），不是开发集。训练、checkpoint 选择、教师选择、PCA/层位/损失/温度/激活筛选只能使用 `dev_local`。只有在数据、代码、协议和候选名单全部冻结后，才允许进行一次最终 test 评估；查看 test 后不得再据此调整任何实验设置。

最终 test 的最小评估集合为：

- 选定的 T1 教师一个 checkpoint（按 `dev_local` 选出并锁定）；
- V2 的 S2-0 scratch 基线；
- 若纳入实用性基线，则评估 S2-P（ImageNette2-320 预训练），并与 scratch 分表报告；
- A/K 阶段最终预注册的 V2 候选，以及后续 V3 的预注册候选。

S2-0 和 S2-P 若已完成 3 个 seed，建议对 3 个 seed 都在同一 `test_local` 上评估并报告 `mean ± std`；不得只挑 test 最高的 seed。若只保存一个最终基线 checkpoint，必须在查看 test 前预先声明选择规则。

官方 test 的网站提交模型和本地 2530/445/500 协议必须分开报告。Cityscapes 官方 test 没有可用本地真值，不能在本地计算 mIoU。

## 4. 每次运行记录

每次必须保存：实验编号、配置和命令；数据清单/模型/PCA/Scaler/adapter 哈希；seed、初始化、global batch、optimizer、LR 和总 step；具体 taps/OS/通道；各 loss 与梯度范数；最佳 dev step；19 类 IoU 与逐图 confusion matrix；移除 adapter 后的参数量、MACs、显存和目标设备延迟；scratch/pretrained、single/multi-scale、dev/test/official-test 口径。
