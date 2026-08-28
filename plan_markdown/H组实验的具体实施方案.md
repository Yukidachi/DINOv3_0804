# H 组实验的具体实施方案：Cityscapes MobileNetV2 激活函数与位置消融

版本：2026-08-24  
适用范围：H0-H3，首轮只在 `MobileNetV2 + R-ASPP`、Cityscapes 本地划分上执行  
依据：`plan_markdown/K组实验的具体实施方案.md`、`plan_markdown/D组实验的具体实施方案.md`、`plan_markdown/R组实验的具体实施方案.md`、`plan_markdown/Cityscapes知识蒸馏实验详单.md`、`dino_s2_0.py`

本文把 H 组从“是否替换 ReLU6”细化为可直接实现的模型构造、激活位置、损失、训练、部署测量、验收、统计和停止规则。H 组首轮只研究 **Hardswish 的激活位置**，不同时搜索多种激活函数、损失权重或网络结构。

---

## 1. H 组要回答的问题

### 1.1 当前已锁定的前置事实

| 项目 | 已锁定结果 | 对 H 组的意义 |
|---|---:|---|
| K1 | `0.522032 ± 0.002194` mIoU，三 seed | feature-KD scratch 参考，不是 H 主模型锚点 |
| R2 | `0.53275 ± 0.00601` mIoU，三 seed | 已验证的空间关系协议 |
| R5 | `seed=42` mIoU=`0.573653` | 当前最强 scratch 组合；尚未完成三 seed |
| D1 | `0.526454 ± 0.006117` mIoU，三 seed | CORAL 有小幅同方向增量，但不是 H 的唯一变量 |
| D2 | `0.520336 ± 0.007948` mIoU，三 seed | SWD 当前配置停止，不作为 H 的基础损失 |

H 组默认以当前 R5 组合为精度筛选协议：

```text
L_R5 = L_seg + warmup(s) * (
          1.0 * L_feat
        + 0.3 * L_R2
        + 0.5 * L_logit
      )
T = 4
```

但 R5 目前只有 `seed=42`。因此 H 的 seed=42 运行只能是筛选证据；任何“稳定优于 R5”的结论都必须在 H0 与候选激活使用相同 seeds、相同初始化和相同 batch 顺序后再作出。H0 是 H 组的 matched anchor，不直接把 R5 单 seed 数值当作三 seed基线。

### 1.2 H 组共同科学问题

H 组回答：

1. `H1-H0`：把 Hardswish 替换到所有 inverted residual 激活位置，是否改变分割性能和部署效率；
2. `H2-H0`：只在 MobileNetV2 的 OS=16 后段替换 Hardswish，是否保留全量替换的精度收益并降低高分辨率激活开销；
3. `H3-H2`：在后段中只替换 depthwise convolution 后的激活，是否足以获得 Hardswish 的收益；
4. Hardswish 的收益究竟来自激活函数本身、深层位置，还是 inverted residual 内的具体放置位置。

H 组不回答：

- MobileNetV2 与 MobileNetV3 的整体架构优劣；
- Hardswish、SiLU、GELU、h-GELU 的完整激活函数网格；
- R5 中 feature、R2 和 logits 的联合最优权重；
- CORAL、SWD、MMD、GAN 与激活函数的交互；
- MobileNetV3-Small+LR-ASPP 的跨架构结论；
- 未查看 `test_local` 后的任何调参问题。

---

## 2. 对“只改后半网络”的回答

### 2.1 结论

**首轮应把“仅修改后段”作为主假设，但不能只运行后段方案。** H1 全部替换和 H3 depthwise-only 是必要对照：

- 只有 H2 高于 H0，才支持“后段替换足够”；
- H1 高于 H2，说明全网络位置可能仍有额外精度贡献，但要同时承担更高的高分辨率开销；
- H2 与 H3 接近，说明后段 depthwise 后的非线性可能是主要贡献位置；
- H1/H2/H3 均与 H0 性能相近，则不能宣称激活替换有效，只能报告部署差异。

MobileNetV3 的证据支持“复杂激活不必从浅层开始”，但它不是对当前 MobileNetV2+R-ASPP 分割系统的直接证明。H 组因此把该结论注册为可检验假设，而不是实现前提。

### 2.2 当前实现中的精确边界

当前 `dino_s2_0.py` 使用 `torchvision.models.mobilenet_v2(weights=None)`，backbone 为 `backbone.0..18`：

| 范围 | 当前结构 | H 组处理 |
|---|---|---|
| `backbone.0` | stem Conv-BN-ReLU6 | 所有 H 组保持 ReLU6 |
| `backbone.1..17` | inverted residual blocks | H1/H2/H3 按矩阵替换其中指定激活 |
| `backbone.18` | final 1×1 Conv-BN-ReLU6 | 所有 H 组保持 ReLU6 |
| R-ASPP `project` | head 内 Conv-BN-ReLU6 | 所有 H 组保持 ReLU6 |

每个 inverted residual 内：

- expansion `1×1 Conv-BN-ReLU6` 的激活路径为 `backbone.i.conv.0.2`；
- depthwise `3×3 Conv-BN-ReLU6` 的激活路径为 `backbone.i.conv.1.2`；
- linear bottleneck projection `backbone.i.conv.2` 只有 Conv-BN，**没有激活，所有 H 组必须保持无激活**。

因此 H 组首轮的可替换站点是 blocks `1..17` 内的 expansion/depthwise 激活，共 34 个候选站点；block 1 的 expansion ratio 为 1，实际只有 depthwise 激活。

`dino_s2_0.py` 的 output-stride 转换只在 blocks `14..17` 的 depthwise convolution 上修改 stride/dilation：block 14 的 stride 从 2 改为 1，blocks 14..17 使用 dilation=2。故本方案将 `backbone.14..17` 定义为 **OS=16 后段**。不能使用“网络后半段”作为唯一配置字段，必须把 block index 写入 `config.json`。

### 2.3 为什么不是直接修改 backbone.7..17

`backbone.7..13` 虽然已经进入中后层，但仍包含 OS=8 的高分辨率特征，且 `backbone.14..17` 才是当前 OS=16、低空间分辨率、膨胀 depthwise 的明确后段。为了让 H2 直接回答“低分辨率深层激活是否足够”，首轮边界固定为 `14..17`。

如果 H2 显示收益但仍存在明显效率/精度折中，后续可以另行登记 `H2-midlate`（blocks `7..17`），不能把它事后并入 H2 的定义。

---

## 3. 论文依据与证据边界

以下均为论文或论文官方页面，不使用博客、厂商文章或教程作为主要依据。

### 3.1 直接相关或强相关论文

1. **Searching for MobileNetV3**, Howard et al., ICCV 2019.  
   官方论文：[CVF Open Access](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html)，[arXiv:1905.02244](https://arxiv.org/abs/1905.02244)。
   
   MobileNetV3 将 hard-swish 选择性用于较深的网络部分，浅层保留较便宜的 ReLU。该论文是 H2 的直接架构先例，但其任务、网络宽度、硬件和训练协议与本项目不同，不能直接迁移精度数字。

2. **Dynamic ReLU**, Chen et al., ECCV 2020.  
   官方论文：[ECCV Proceedings](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640341.pdf)，[arXiv:2003.10027](https://arxiv.org/abs/2003.10027)。
   
   该工作研究动态激活在 inverted bottleneck 内不同位置的作用，说明 expansion、depthwise 和 projection 不是等价的放置位置。它支持 H3 这种 placement ablation，但不证明固定 Hardswish 必须只放在 depthwise 后。

3. **Activate or Not: Learning Customized Activation**, Ma et al., CVPR 2021.  
   官方论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Activate_or_Not_Learning_Customized_Activation_CVPR_2021_paper.html)，[arXiv:2009.04759](https://arxiv.org/abs/2009.04759)。
   
   该论文研究可学习激活和移动网络中的激活替换，支持“激活不是与 block 结构无关的独立旋钮”。它不能替代本项目对 Hardswish、block scope 和移动端实测延迟的独立验证。

4. **MobileOne: An Improved One millisecond Mobile Backbone**, Vasu et al., CVPR 2023.  
   官方论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Vasu_MobileOne_An_Improved_One_Millisecond_Mobile_Backbone_CVPR_2023_paper.html)，[arXiv:2206.04040](https://arxiv.org/abs/2206.04040)。
   
   该工作强调移动端实际延迟、重参数化和算子部署，而不只是 FLOPs。它支持 H 组必须报告真实 latency、peak memory 和 export/operator compatibility，不能只用 MACs 推断 Hardswish 是否更快。

5. **RepViT: Revisiting Mobile CNN From ViT Perspective**, Wang et al., CVPR 2024.  
   官方论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html)，[arXiv:2307.09283](https://arxiv.org/abs/2307.09283)。
   
   RepViT 展示了移动网络中 block、激活、归一化和 stage 设计的耦合。它是移动 CNN 的架构类比证据，支持按 stage 研究激活位置，但不是对当前 MobileNetV2 segmentation head 的直接验证。

6. **MobileNetV4: Universal Models for the Mobile Ecosystem**, Qin et al., CVPR 2024.  
   官方论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Qin_MobileNetV4_Universal_Models_for_the_Mobile_Ecosystem_CVPR_2024_paper.html)，[arXiv:2404.10518](https://arxiv.org/abs/2404.10518)。
   
   该工作从多种移动硬件和模型组件协同设计出发，支持 H 组将硬件/导出测量纳入主验收。它不直接规定 MobileNetV2 的 H2 block 边界。

### 3.2 文献综合结论

论文证据共同支持三点：

1. 激活函数的收益取决于 block 位置和 stage，而不是简单“全网替换越多越好”；
2. 浅层高分辨率激活对内存访问和真实延迟更敏感；
3. 移动端结论必须结合目标硬件和算子融合实测。

论文没有直接证明“MobileNetV2+R-ASPP 一定应只替换 blocks 14..17”。因此 H2 是合理的主假设，H1/H3 是必要的可辨识对照。

---

## 4. H0-H3 实验矩阵

首轮固定目标激活为 Hardswish，只改变替换位置：

| 编号 | Stem/final | inverted residual expansion | inverted residual depthwise | 目标 |
|---|---|---|---|---|
| H0 | ReLU6 | ReLU6，blocks 1..17 | ReLU6，blocks 1..17 | R5-compatible activation anchor |
| H1 | ReLU6 | Hardswish，blocks 1..17 | Hardswish，blocks 1..17 | 全部 in-block 替换 |
| H2 | ReLU6 | Hardswish，blocks 14..17 | Hardswish，blocks 14..17 | 后段 expansion+depthwise 替换，主假设 |
| H3 | ReLU6 | ReLU6，blocks 1..17 | Hardswish，blocks 14..17 | 后段 depthwise-only 替换 |

所有 H 组：

- linear bottleneck projection 不加激活；
- `backbone.0`、`backbone.18` 和 R-ASPP head 保持 ReLU6；
- 不改变 block 数、通道数、stride、dilation、feature tap、R-ASPP 或输出尺寸；
- 不改变 R5 的 loss、teacher、PCA、R2、logits、数据和 optimizer；
- H0 必须与 R5 的 ReLU6 模型拓扑一致，作为 H 组内部 anchor。

### 4.1 后续激活函数扩展

首轮 H0-H3 只使用 Hardswish，以隔离位置变量。只有 H2 通过 seed=42 的实现、精度和效率筛选后，才可另行注册：

| 编号 | 激活 | 位置 | 目的 |
|---|---|---|---|
| H4 | SiLU | blocks 14..17 expansion+depthwise | 比较平滑非 hard 激活 |
| H5 | GELU | blocks 14..17 expansion+depthwise | 比较高表达力激活 |
| H6 | h-GELU | blocks 14..17 expansion+depthwise | 比较部署友好的近似激活 |

H4-H6 不能在 H0-H3 首轮中隐式加入，也不能用不同激活函数替代 H1/H2/H3 后再解释位置效应。

---

## 5. 所有 H 运行必须一致的协议

### 5.1 数据、标签和增强

- `train_local=2530`、`dev_local=445`、`test_local=500`；组合清单 SHA-256：

  ```text
  033161572be28a6de295e0c5dfb62d83cd4d0a18b6039321347c58ab28b9d3c2
  ```

- 训练视图：随机缩放 `[0.5,2.0]`、随机裁剪 `512×1024`、水平翻转；
- 标签：`labelIds -> trainIds 0..18`，其余为 `ignore_index=255`；
- dev：原分辨率 `1024×2048`，单尺度、无水平翻转；
- 不加入 class-mix、copy-paste、伪标签、类别权重、Dice、Focal、辅助头或多尺度推理；
- `test_local` 不参与训练、checkpoint 选择、激活选择、候选筛选或 seed 扩展；所有 H 运行写入 `test_local_evaluated=false`。

### 5.2 教师与 R5 loss

H0-H3 固定使用：

```text
result/T1_DINOv3_RASPP/seed_3407/t1_dinov3_raspp_teacher.pth
teacher SHA-256 = 73cb1d3161c746d1b4ea30918ec6a1f0de5e3a4952c000cf85ddf95f3ccaddeb
```

教师始终 `eval()`、冻结、不进入 optimizer、不包 DDP，并在 `torch.no_grad()` 中前向。teacher feature、projected feature target 和 teacher logits 必须 detach。

H 首轮默认 loss：

```text
L_H = L_seg + warmup(s) * (
        1.0 * L_feat
      + 0.3 * L_R2
      + 0.5 * L_logit
    )
T = 4
warmup_steps = 4000
```

`L_feat`、`L_R2`、`L_logit` 的正式定义分别复用 K/R 组，不得在 H 组复制出不同 reduction：

- `L_feat`：A0 fixed StandardScaler+PCA，OS=4/8/16 等权 BCHW MSE；
- `L_R2`：R 组 native feature 的 masked 8×16 token relation，`lambda_R2=0.3`；
- `L_logit`：全分辨率 masked pixel-logit KL，`T=4`、`lambda_logit=0.5`。

H 组不加入 CORAL、SWD、MMD、adversarial 或新的关系项。

### 5.3 学生、shape 和 output stride

学生固定为 `MobileNetV2 + R-ASPP`、`weights=None`、`output_stride=16`：

| 层 | 学生 tap | `512×1024` 形状 |
|---|---|---|
| OS=4 | `backbone.3` | `[B,24,128,256]` |
| OS=8 | `backbone.6` | `[B,32,64,128]` |
| OS=16 | `backbone.17` | `[B,320,32,64]` |
| R-ASPP input | `backbone.18` | `[B,1280,32,64]` |

H 代码不得因为激活替换改变 `backbone.14..17` 的 stride/dilation contract。`align_corners=False`、19 类输出和原分辨率 logits 必须保持不变。

### 5.4 共同初始化与 batch 顺序

H0-H3 必须使用同一套 matched 初始化协议：

1. 每个 seed 从 K 组 shared scratch initialization 开始；
2. 不加载 R5、K4、D1、A0-FT 或其他训练完成的学生 checkpoint；
3. H0-H3 同一 seed 的 step=0 `student_state_sha256` 必须一致；
4. H0-H3 同一 seed 的 DataLoader generator、DistributedSampler、增强状态和前 N 个 batch 必须一致；
5. 不同 seed 的 initialization hash 必须不同；
6. H0-H3 唯一模型变量是激活名称和激活 placement。

注意：H0-H3 的第一阶段建议使用与 R5 相同的 scratch 训练入口重新受控复现，而不是直接加载 R5 checkpoint。R5 当前只有 seed=42，不能作为 H0 的三 seed checkpoint 来源。

### 5.5 预算与优化器

| 项目 | 固定值 |
|---|---|
| world size | 2 |
| per-GPU batch | 2 |
| accumulation | 2 |
| effective optimizer global batch | 8 |
| optimizer | SGD，lr=`0.01`，momentum=`0.9`，weight decay=`1e-4` |
| scheduler | poly，power=`0.9`，min LR ratio=`0.01` |
| optimizer steps | `80,000` |
| AMP | 开启 |
| deterministic | 开启 |
| checkpoint 选择 | 只按 dev mIoU |
| eval 间隔 | 每 5,000 optimizer steps 附近的首个 epoch 边界 |

---

## 6. 激活函数的正式定义与替换规则

### 6.1 Hardswish

首轮 H1-H3 使用 PyTorch/torchvision 的标准 Hardswish 语义；配置中必须同时记录公式和实现名称：

\[
\operatorname{h\text{-}swish}(x)=x\frac{\operatorname{ReLU6}(x+3)}{6}.
\]

不得把 `SiLU`、近似 GELU 或自定义 hard-swish 误记为 Hardswish。若实现使用 fused/mobile backend，必须先用 reference tensor 与公式比较。

### 6.2 精确替换站点

H 训练入口应在构建 MobileNetV2 后、shape audit 前执行 deterministic activation replacement，并记录替换清单：

```text
all_inverted_residual =
  for block in backbone.1..17:
    block.conv.0.2  # 若该 expansion path 存在
    block.conv.1.2  # depthwise path

late_blocks = backbone.14..17
```

替换规则：

- H1：替换 blocks `1..17` 内所有存在的 `conv.0.2` 和 `conv.1.2`；
- H2：只替换 blocks `14..17` 内所有存在的 `conv.0.2` 和 `conv.1.2`；
- H3：只替换 blocks `14..17` 内的 `conv.1.2`；
- 所有 H：不替换 `backbone.0.2`、`backbone.18.2`、R-ASPP head activation 和任何 `conv.2` projection；
- 未命中的模块路径必须保持 ReLU6，不能因为模块类型相同而全局替换。

建议沿用 K 组的 variant builder/hook 模式：以 `dino_s2_0.build_backbone()` 为基础构造 H backbone，再由 H spec 注入激活，避免复制训练循环。

### 6.3 参数与部署约束

Hardswish、ReLU6、SiLU、GELU 和固定 hard-GELU 均为无参数激活。H0-H3 的总参数量、可训练参数量、feature tap 和 R-ASPP head 必须相同。若某实现引入 learnable activation 参数，必须另建编号，不得放入 H1-H3。

---

## 7. 正式训练前的验收和测试

正式 80k 训练前，必须先完成 H0-H3 的构造、数值、梯度和部署 smoke test。

### 7.1 结构与 placement 测试

- H0 的激活模块拓扑与原始 MobileNetV2 完全一致；
- H1 的替换站点是全部 in-block eligible sites；
- H2 的替换站点只属于 blocks `14..17`；
- H3 的替换站点只属于 blocks `14..17` 的 depthwise path；
- H1-H3 替换站点数量和模块路径写入 `config.json`；
- stem、final 1×1、R-ASPP head 仍为 ReLU6；
- 所有 linear bottleneck `conv.2` 不包含 activation；
- H0-H3 的 backbone length 都是 19；
- `backbone.14..17` 的 stride/dilation 与 H0 完全一致；
- OS=4/8/16/R-ASPP input shape 与基线一致。

### 7.2 激活数值测试

- Hardswish reference formula 与实现最大绝对误差不超过预注册容差；
- `x= -3, 0, 3` 等分段边界值与 reference 一致；
- 正、负、大幅值输入均 finite；
- H0 ReLU6 输出与未改模型一致；
- H1-H3 只改变注册站点的输出；
- H0-H3 训练前 state hash、activation spec hash 和 replacement list hash 可复现。

### 7.3 损失与梯度测试

- CE、feature、R2、logits 和 total loss 均 finite；
- teacher 参数无 grad；A0 projection 无 grad；
- 学生 backbone、head 和被替换激活前后的卷积路径有 grad；
- warm-up 在 step=1 为 `1/4000`，step=4000 达到 1；
- H0-H3 的 loss reduction、mask、R2 pair denominator 和 KL `T²` 与 R5 reference 一致；
- 辅助项有效梯度连续三个注册审计点超过 CE 的 2 倍时停止该运行并排查，不得事后静默调权重；
- 激活替换不能改变 teacher、PCA、R2 或 logits target。

### 7.4 训练恢复测试

- accumulation 前 N-1 个 micro-batch 使用 `no_sync()`；
- scheduler 每 optimizer step 只前进一步；
- resume 恢复 step、LR、AMP scaler、generator、best checkpoint 和 activation spec；
- rank 0 独占写文件，collective 顺序一致；
- smoke test 后所有 rank 返回码为 0。

---

## 8. 精度与部署效率测量

H 组不以理论 MACs 作为唯一效率指标。

### 8.1 精度指标

每个运行报告：

- 19 类 mIoU、mAcc、pixel accuracy；
- 每类 IoU；
- small-object mIoU；
- boundary F1；
- best step、训练曲线和 finite 状态；
- 19×19 confusion matrix 与逐图 confusion JSONL。

### 8.2 运行时指标

仅测学生 MobileNetV2+R-ASPP，不计入 teacher、PCA、R2、logits 或其他训练期模块：

- 参数量和 trainable 参数量；
- FP32 MACs/GMACs；
- 目标设备 batch=1 latency 的 p50、p90 和标准差；
- warm-up 次数、正式测量次数、输入分辨率和同步方式；
- peak CUDA memory 或目标设备峰值内存；
- ONNX/TensorRT/TFLite 等目标后端的导出结果；
- activation 是否成功融合、是否产生额外 kernel 或 fallback operator。

H0-H3 必须使用同一设备、同一软件版本、同一输入、同一 precision、同一测量顺序和同一 warm-up。若某后端不支持某激活，记录 export failure，不能用另一个后端的结果填补。

### 8.3 精度-效率解释

| 结果 | 允许结论 |
|---|---|
| H2 精度不劣于 H0 且延迟更低 | 支持后段替换在当前 V2+R-ASPP 上具有更优 Pareto 方向 |
| H1 精度高于 H2 但延迟明显更高 | 全量替换存在精度-效率折中，不能无条件称优 |
| H3 接近 H2 | 支持后段 depthwise 激活是主要候选位置之一 |
| H3 明显低于 H2 | expansion path 可能提供额外作用，不能只保留 depthwise |
| 精度提升但导出/融合失败 | 报告为部署不兼容或待验证，不报告为移动端收益 |
| CI 跨 0 或差值小于 seed 波动 | 描述为性能相近/证据不足 |

---

## 9. seed=42 筛选与三 seed 扩展

### 9.1 首轮执行顺序

```text
H0(seed=42)
-> H1(seed=42)
-> H2(seed=42)
-> H3(seed=42)
-> 只扩展通过筛选的候选
```

H0-H3 不得从前一个候选的 checkpoint 继续训练。每个候选都从同一个 seed=42 H shared scratch initialization 独立开始。

### 9.2 seed=42 候选门

候选必须满足：

1. 激活公式、placement、linear bottleneck、shape 和 R5 loss reference tests 全部通过；
2. loss、gradient、checkpoint reload 和训练过程 finite；
3. 相对 H0 的 mIoU、boundary F1、small-object mIoU 和逐图 paired bootstrap 已记录；
4. 若宣称精度改善，mIoU 增益应超过 K1 三 seed样本标准差 `0.002194`，或有明确机制指标改善且 paired evidence 不与其冲突；
5. efficiency、export 和 operator fusion 结果已记录；
6. `test_local_evaluated=false`。

单次最高 mIoU、未配对比较、改变 R5 loss、改变初始化或查看 `test_local`，都不能触发三 seed扩展。

### 9.3 三 seed 扩展规则

至少扩展：

- H0：`seed=42/3407/260805`；
- 通过筛选且具有清晰科学解释的最佳候选一个。

如果目标是正式回答“后段是否优于全量”，则 H1、H2 必须一起扩展；否则 H1/H2 不能作正式位置因果比较。H3 只有在 H2 通过且需要区分 depthwise placement 时扩展。

扩展时：

- 使用对应 seed 的 K shared scratch initialization；
- 保持 activation spec、R5 loss、数据、batch 顺序和预算不变；
- 不重新搜索激活位置；
- 不查看 `test_local`；
- 每个候选报告 `mean ± sample std`（ddof=1）。

---

## 10. 统计与结果解释

每个 H 运行报告：

- 每个 seed 的最佳 dev step 和全部指标；
- 三 seed `mean ± sample std`；
- 同 seed H 候选-H0 的 mIoU 差值；
- 445 张相同 dev 图像的 paired-bootstrap 95% CI；
- 每类 IoU、small-object mIoU、boundary F1 差值；
- activation/CE、feature/CE、R2/CE、logits/CE 梯度比例和 cosine；
- latency、peak memory、MACs、导出和融合状态。

H0-H3 的 paired bootstrap 必须按同 seed、同一批 445 张 dev 图像分别计算：先累加抽样图像的 19×19 confusion matrix，再计算 mIoU；不能平均逐图 mIoU，也不能把不同 training seed 混在一个 bootstrap 数据集。

允许的表述：

| 结果 | 允许结论 |
|---|---|
| H2-H0 三 seed 同方向且 CI 支持正向 | 当前 V2+R-ASPP 中后段激活替换有稳定增量证据 |
| H1-H0 正向且 H2-H0 相近 | 后段替换可能保留全量替换收益并降低部署成本 |
| H1 高于 H2 且效率代价明显 | 全量替换有精度收益，但存在部署折中 |
| H3 与 H2 相近 | depthwise placement 足以解释大部分收益的证据 |
| 差值小于 seed 波动或 CI 跨 0 | 性能相近/证据不足 |
| 精度和延迟方向相反 | 报告 Pareto trade-off，不宣布单一胜者 |
| 只有 seed=42 改善 | 仅为筛选信号，不能宣称稳定激活收益 |

不得宣称：

- MobileNetV3 已经证明 MobileNetV2 必须只改后半段；
- Hardswish 在所有移动设备都比 ReLU6 快；
- GELU/SiLU 的理论表达力必然转化为 Cityscapes mIoU；
- 单次 seed 结果证明激活函数普遍有效。

---

## 11. 审计产物和结果目录

建议使用独立目录：

```text
result/H_MobileNetV2_RASPP_server/
  shared_init/
    seed_42/student_init.pth(.sha256)
    seed_3407/student_init.pth(.sha256)
    seed_260805/student_init.pth(.sha256)
  H0/seed_42/
  H1/seed_42_hardswish_all_inverted_residual/
  H2/seed_42_hardswish_blocks_14_17/
  H3/seed_42_hardswish_depthwise_blocks_14_17/
```

每个 H 运行至少保存：

- `config.json`：H 编号、activation name、activation formula、placement scope、affected module paths、activation/replacement hash、R5 loss开关和权重、warm-up、seed、batch、80k预算、optimizer、AMP、deterministic；
- `feature_taps.json` 与 OS=16 shape/stride/dilation audit；
- `first_batch_audit.json`；
- shared-init、teacher、PCA、manifest 和 R5 relation/logit spec hash；
- `activation_replacement.json`：原模块类型、替换后模块类型、完整路径、block index、path type；
- `training_history.json`：CE、feature、R2、logits、total loss、warm-up、LR；
- `gradient_norms.jsonl`：组件梯度、比例、cosine、finite 状态和 activation spec hash；
- `last_checkpoint.pth` 及恢复状态；
- `best_checkpoint.pth` 及 `.sha256`；
- `dev_metrics.json`、19×19 confusion matrix、逐图 confusion JSONL；
- `efficiency.json`：参数量、MACs、latency、memory、export/fusion 状态；
- `software.json`；
- `test_local_evaluated=false`。

H 组还应保存论文证据表，至少包含论文标题、作者、年份、venue、官方链接、涉及的激活/位置、直接证据或类比证据、与本项目的差异。

---

## 12. 推荐实现与执行顺序

### 12.1 实现顺序

```text
1. 固化 H0-H3 ExperimentSpec、activation formula 和 placement policy
2. 复用 dino_s2_0.build_backbone()、R5 shared forward 和 K shared init
3. 实现 activation factory 与显式 block/path replacement
4. 添加 activation formula、module topology、linear bottleneck 和 shape reference tests
5. 添加 R5 loss、gradient、resume 和 DDP smoke test
6. 添加参数量、MACs、latency、memory、export/fusion audit
7. 运行 H0-H3 seed=42 短程预检和正式筛选
8. 对 H0 以及通过门控的候选扩展 seed=3407/260805
9. 汇总 matched 差值、bootstrap CI、梯度、部署和论文证据
10. 冻结 H 候选名单后再决定最终 test_local 评估
```

### 12.2 H 组停止条件

- activation reference、placement 或 shape test 失败：停止正式训练，修复后从 step=0 开始；
- H0 无法复现 R5-compatible baseline：停止 H 组，先修复公共入口；
- 任意 H 候选出现 non-finite、teacher gradient、linear bottleneck 被激活或 output-stride contract 改变：该运行无效；
- 辅助损失梯度连续三个注册点超过 CE 的 2 倍：停止并检查 R5 loss/reduction，不在训练中动态调权重；
- 激活导出失败或后端不支持：停止该部署结论，保留精度结果并标记部署不可用；
- H0 与 H2 三 seed性能相近且 H2无效率优势：停止后段激活主线，不继续扩展更多激活函数；
- 查看 `test_local` 后不得继续调整 H 组设置。

### 12.3 H 组完成标准

H 组完成不是“某个激活函数 mIoU 最高”，而是：

1. H0-H3 的 activation placement 可由 module path 审计；
2. H0 与 R5-compatible 公共入口、初始化、batch 顺序和 loss reference 一致；
3. linear bottleneck、stem、final projection、R-ASPP 和 output-stride contract 未被误改；
4. 通过筛选的候选完成 matched 三 seed，或明确记录未扩展原因；
5. 最佳 checkpoint 可重载复现；
6. 精度、梯度、bootstrap、MACs、latency、memory 和导出状态均已报告；
7. 论文证据明确区分 MobileNetV3 直接先例、移动 CNN 类比和本项目自身证据；
8. `test_local` 在候选冻结前未查看；
9. 是否进入 H4-H6、D5 或最终 test 有明确的基于证据的解锁结论。
