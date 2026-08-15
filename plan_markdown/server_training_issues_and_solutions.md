# S2-0 服务器训练问题与解决方案

版本：2026-08-06  
适用入口：`dino_s2_0_server.py`  
适用环境：Linux 服务器、`torchrun`、多 GPU DDP、Cityscapes `train_local/dev_local`

本文记录服务器版 S2-0（MobileNetV2 + R-ASPP）训练中已经观察到的问题、证据、修复方式和后续编程约束。后续修改训练入口、DataLoader 或 DDP 生命周期时，必须先对照本文的检查清单。

## 1. 当前结论

服务器 smoke test 的模型计算已经通过。曾经的失败发生在 smoke test 输出之后的进程退出阶段，而不是数据、模型 shape、前向、反向或 loss 计算阶段。

当前服务器入口的关键设计为：

| 领域 | 当前约定 |
|---|---|
| 启动 | `torchrun --standalone --nproc_per_node=2 dino_s2_0_server.py` |
| 设备 | 每个 rank 绑定 `LOCAL_RANK` 对应的 CUDA 设备 |
| 分布式 | 多 GPU 使用 NCCL、`DistributedDataParallel` 和 `DistributedSampler` |
| DataLoader | 服务器默认 `spawn`，默认 `pin_memory=False`；`persistent_workers` 只在 worker>0 时启用 |
| 梯度累积 | 非最后一个 micro-batch 使用 `DDP.no_sync()` |
| 全局 batch | `batch_size_per_gpu * accumulation_steps * world_size` |
| 输出 | 使用 `result/S2_0_MobileNetV2_RASPP_server/`，避免覆盖普通 S2-0 结果 |
| 退出 | 停止 worker、同步 CUDA、释放 DDP 相关对象，再销毁 process group |

## 2. 服务器基线环境

profiling 记录 `result/s_2_0_profile_spawn_no_pin.json` 显示的环境：

- Python 3.10.19，PyTorch 2.5.1，torchvision 0.20.1。
- CUDA 12.4，cuDNN 9.17.0，NVIDIA GeForce RTX 4090，显存约 24 GB。
- Linux 5.15，CPU 22 核。
- 训练输入 crop 为 `512x1024`，评估分辨率为 `1024x2048`。
- AMP 开启，deterministic 开启，cuDNN benchmark 和 TF32 关闭。

这些版本和开关需要写入结果 JSON；不要只依赖服务器当前环境，否则后续无法判断性能或精度变化来自代码还是运行时。

## 3. 已发现的问题

### 3.1 普通入口不能直接当作服务器 DDP 入口

`dino_s2_0.py` 面向单进程/普通诊断流程，服务器需要同时处理 rank、GPU 绑定、全局 batch、梯度同步和多进程退出。因此服务器使用独立的 `dino_s2_0_server.py`，不要在普通入口中临时拼接 DDP 逻辑。

服务器入口必须满足：

1. 从 `RANK`、`WORLD_SIZE`、`LOCAL_RANK` 读取 torchrun 环境。
2. 在创建模型前执行 `torch.cuda.set_device(local_rank)`。
3. 只在 `world_size>1` 时初始化 NCCL process group。
4. 所有 rank 执行相同数量、相同顺序的 collective；仅 rank 0 写 checkpoint 和评估结果。

### 3.2 pinned-memory 路径在服务器上表现异常

`result/s_2_0_profile.json` 记录了原始 pinned-memory 配置的瓶颈：

- `data_wait` 平均约 `224.81 ms`，占同步阶段 `77.9%`。
- synthetic/e2e 吞吐为 `29.02/3.68`，比例约 `7.89x`。
- 平均 GPU 利用率约 `12.6%`，说明 GPU 经常等待主机侧数据管线。
- worker sweep 中 `num_workers=4` 最快，约 `7.64 samples/s`，但整体仍受数据等待限制。

服务器曾出现 pinned-memory 路径使 worker 近似串行、吞吐明显下降的现象。因此服务器入口默认关闭 pinned memory；不要因为 CUDA 通常推荐 pin memory 就直接改回默认值，必须用实际服务器 profiling 证明有效。

### 3.3 worker 数量和启动方法不能照搬笔记本

关闭 pin memory、使用 `spawn` 的 sweep 结果如下（`result/s_2_0_profile_spawn_no_pin.json`，batch size=2）：

| workers | steady batch 吞吐 | samples/s |
|---:|---:|---:|
| 0 | 3.77 batch/s | 7.55 |
| 2 | 7.31 batch/s | 14.61 |
| 4 | 10.80 batch/s | 21.59 |
| 8 | 22.27 batch/s | 44.54 |

结论是 worker>0 有明显收益，8 个 worker 在该 profiling 环境最快；但这是独立 DataLoader sweep，不等于完整 DDP 训练的最终最优值。实际训练仍需结合 CPU、磁盘、显存和多 rank 争用重新测量。

建议顺序：先用 `num_workers=0` 做功能 smoke，再用 `spawn` 测试 `2/4/8`，最后锁定一个配置。`persistent_workers` 只有在跨 epoch 重启开销确实明显时才开启。

### 3.4 全局 batch 与梯度累积容易算错

`--batch-size` 是每个 GPU 的 local batch，不是全局 batch。正确关系为：

```text
global_batch_size = batch_size_per_gpu * accumulation_steps * world_size
```

例如 2 GPU、每卡 batch=2、目标 global batch=8 时，accumulation steps 应为 2，而不是继续使用默认值 4。`--global-batch-size` 必须能被 `batch_size * world_size` 整除；入口会主动检查这一条件。

梯度累积期间必须：

- 最后一个 micro-batch 才同步梯度；
- 其他 micro-batch 使用 `model.no_sync()`；
- 每个 optimizer step 只执行一次 `scaler.step/update`、`optimizer.zero_grad` 和 scheduler step；
- `DistributedSampler.set_epoch(epoch)` 在每个 epoch 开始调用。

### 3.5 smoke test 后 rank 0 发生 SIGSEGV

曾执行：

```bash
torchrun --standalone --nproc_per_node=2 dino_s2_0_server.py \
  --device cuda --smoke-test --batch-size 1 \
  --global-batch-size 2 --num-workers 0
```

输出包含：

```text
[OK] server DDP smoke test: ... logits=(1, 19, 512, 1024), loss=2.994007
rank: 0
exitcode: -11
Signal 11 (SIGSEGV)
```

这说明前向、反向和 loss 都完成；rank 1 的 SIGTERM 是 torchrun 在 rank 0 崩溃后进行的联动清理。结合崩溃时序，最可能的原因是退出阶段的资源生命周期竞争：

- CUDA/NCCL 反向 collective 仍有异步工作；
- DDP reducer 仍持有 process group 和 CUDA bucket 引用；
- DataLoader iterator 或 persistent worker 仍在后台运行；
- 原代码在这些对象仍存活时直接调用 `dist.destroy_process_group()`。

`nll_loss2d_forward_out_cuda_template` 的 deterministic warning 只是警告，不是本次 SIGSEGV 的直接证据。

## 4. 已实施的解决方案

### 4.1 独立服务器入口

`dino_s2_0_server.py` 将服务器配置与普通 S2-0 分开：

- 默认 `multiprocessing_context="spawn"`；
- 默认 `pin_memory=False`；
- 默认输出目录带 `_server` 后缀；
- 记录 world size、每卡 batch、累积步数、worker、AMP、deterministic 和数据锁信息。

### 4.2 DDP 训练实现

模型使用：

```python
DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    broadcast_buffers=True,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
)
```

训练循环用 `no_sync()` 处理梯度累积，并在 epoch、评估、checkpoint 和最终退出处安排 rank 同步。任何新增 collective 都必须在所有 rank 上走同一分支，否则可能死锁或触发 NCCL 异步错误。

### 4.3 有序退出

当前入口增加了 `_shutdown_loader()` 和 `_synchronize_cuda()`，并在 `finally` 中执行：

1. 显式关闭 DataLoader persistent worker iterator。
2. `torch.cuda.synchronize(device)`，等待 CUDA/NCCL 工作完成。
3. 成功路径中跨 rank barrier。
4. 释放 `selected_model`、scaler、scheduler、optimizer 和 DDP model。
5. 再次同步 CUDA 和 barrier。
6. 最后才调用 `dist.destroy_process_group()`。

异常路径不额外执行 barrier，避免某个 rank 已失败时其他 rank 永久等待；仍会尽力停止 worker、同步和释放本地对象。

## 5. 标准运行流程

### 5.1 第一步：单次 DDP smoke

先使用零 worker，排除 worker 生命周期问题：

```bash
torchrun --standalone --nproc_per_node=2 dino_s2_0_server.py \
  --device cuda --smoke-test \
  --batch-size 1 --global-batch-size 2 \
  --num-workers 0 --no-pin-memory
```

成功标准：

- 所有 rank 正常退出，shell 返回码为 0；
- 出现 `[OK] server DDP smoke test`；
- 没有 `SIGSEGV`、`ChildFailedError` 或 NCCL timeout；
- logits shape 为 `(1, 19, 512, 1024)`，loss 为 finite。

### 5.2 第二步：worker sweep

使用 profiling 脚本测量 `0/2/4/8` worker，固定输入尺寸、batch、AMP、deterministic 和 pin-memory 选项。服务器历史结果表明：

- 首次诊断可使用 `spawn + no pin + persistent_workers=false`；
- 完整训练再比较 `persistent_workers` 开关；
- 不以单次 first batch 时间选择 worker，至少比较 steady-state 和完整 epoch。

### 5.3 第三步：正式训练

保持 global batch=8 的示例：

```bash
torchrun --standalone --nproc_per_node=2 dino_s2_0_server.py \
  --seed 42 \
  --batch-size 2 --global-batch-size 8 \
  --num-workers 8 \
  --multiprocessing-context spawn \
  --no-pin-memory --persistent-workers
```

如果服务器内存、CPU 或磁盘压力较大，先降到 `--num-workers 4`，不要同时修改 batch、crop、AMP 和 worker，否则无法判断收益来源。

### 5.4 发生崩溃时的隔离顺序

1. `--num-workers 0 --no-pin-memory --no-persistent-workers`：隔离 DataLoader。
2. `--nproc_per_node=1`：隔离 DDP/NCCL。
3. 保持 DDP，添加 `NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL`：确认 collective 顺序和 NCCL 错误。
4. 检查 `nvidia-smi`、CUDA/PyTorch/cuDNN 版本、GPU 是否被其他任务占用。
5. 最后才做 `--no-deterministic` 性能 A/B；这不是 SIGSEGV 的首选修复，也不能混入正式精度实验。

注意：shell 中单独的 `--` 会终止 argparse 选项解析。命令换行使用反斜杠 `\`，不要在 `--global-batch-size` 前误放一个独立的 `--`。

## 6. 后续编程约束清单

### DDP 与设备

- [ ] 模型创建前绑定 `LOCAL_RANK`，禁止所有 rank 默认使用 `cuda:0`。
- [ ] 所有 rank 创建相同结构的模型和 optimizer。
- [ ] 所有 collective 的调用次数和顺序完全一致。
- [ ] rank 0 独占文件写入；写完后让其他 rank barrier。
- [ ] 评估和保存后再次 barrier，再进入下一轮训练。

### DataLoader

- [ ] worker 内不创建 CUDA tensor，不访问共享 CUDA context。
- [ ] `persistent_workers` 只能在 `num_workers>0` 时传入 true。
- [ ] 新增 worker 或 iterator 后，在退出前显式调用 `_shutdown_workers()`（通过统一 helper）。
- [ ] `pin_memory` 必须用 profiling 证明有效后再开启。
- [ ] worker 数量变更时记录 CPU、磁盘和 GPU 利用率。

### 训练和可复现性

- [ ] 明确记录 local batch、world size、accumulation 和实际 global batch。
- [ ] 保留 `DistributedSampler.set_epoch()`、seed 和 generator state。
- [ ] deterministic 开关、AMP、TF32、cuDNN benchmark 必须写入结果 JSON。
- [ ] 看到 warning 时先判断是否影响数值或退出码，不要把 warning 直接当成训练失败。

### 退出与异常

- [ ] 正常退出顺序必须是：停止 worker -> CUDA synchronize -> 释放 DDP/优化器 -> barrier -> destroy process group。
- [ ] 异常路径不要盲目新增 barrier；失败 rank 不能让其他 rank 永久阻塞。
- [ ] 训练入口必须有 smoke-test，且验证完整 forward/backward，而不只是加载模型。
- [ ] 正常退出的判定以所有 rank 返回码为准，不能只看 rank 0 的 `[OK]` 文本。

## 7. 证据与相关文件

| 文件 | 用途 |
|---|---|
| `dino_s2_0_server.py` | 服务器 DDP 训练入口和有序退出实现 |
| `scripts/profile_dino_s2_0.py` | worker、pin memory、GPU 利用率和端到端瓶颈 profiling |
| `result/s_2_0_profile.json` | 原始 pinned-memory profiling 证据 |
| `result/s_2_0_profile_spawn_no_pin.json` | `spawn + no pin` worker sweep 证据 |
| `dino_s2_0.py` | 普通 S2-0 单进程入口，不能直接替代服务器入口 |

每次后续改动都应在本文件追加“日期、命令、环境、结果和结论”，不要只修改代码而不更新运行约束。

## 8. S2-F 服务器入口（2026-08-07）

新增入口：`dino_s2_f_server.py`。它沿用本文件的 DDP、`spawn` worker、默认关闭 pinned memory、梯度累积和有序退出约束，但实验定义改为：`weights=None` 的 MobileNetV2 完整 backbone 冻结，只训练 19 类 R-ASPP head。冻结不仅包括参数的 `requires_grad=False`，也包括训练期间 backbone 的 BatchNorm 运行统计保持 `eval()`；入口还会在 smoke test 和每个 epoch 后校验冻结 backbone 的梯度与 SHA-256。

S2-F 是 A 组 probe 下界，默认使用冻结 head probe 的 40,000 个 optimizer steps，不使用 S2-0 的 80,000 步端到端预算。结果写入 `result/S2_F_MobileNetV2_RASPP_server/`，文件带 `s2_f_server` 标识，不覆盖 S2-0 结果。

服务器正式启动示例：

```bash
torchrun --standalone --nproc_per_node=2 dino_s2_f_server.py \
  --seed 42 --batch-size 2 --global-batch-size 8 \
  --num-workers 8 --multiprocessing-context spawn \
  --no-pin-memory --persistent-workers
```

本地功能验证（Windows、PyTorch 2.5.0、CUDA 可用、单进程；不替代 Linux 两卡 DDP 验证）：

```bash
python -B dino_s2_f_server.py --device cuda --smoke-test \
  --batch-size 1 --global-batch-size 1 --num-workers 0 \
  --no-persistent-workers --no-pin-memory --no-amp
```

结果：正常退出，logits 为 `(1, 19, 512, 1024)`，loss `2.944437`，未发现冻结 backbone 梯度。正式服务器运行前仍需执行 `num_workers=0` 的两卡 DDP smoke，再进行 worker sweep。

## 9. S2-P 服务器入口（2026-08-07）

新增入口：`dino_s2_p_server.py`。由于当前没有现成的 ImageNet-1K MobileNetV2 权重，该入口提供两个显式阶段：

1. `--stage imagenet-pretrain`：从本地 ImageNet-1K `train/类别目录/图片` 结构训练原生 MobileNetV2 分类模型，默认 90 epochs，并保存完整分类 checkpoint；如果存在 `val/类别目录/图片`，按 val top-1 选择最佳权重。
2. `--stage cityscapes`：读取上述 checkpoint 中的 `features.*`，加载到 Cityscapes OS=16 MobileNetV2 backbone，再按 S2-0 的 80,000 optimizer steps、SGD+poly、端到端 pixel CE 训练 R-ASPP。该阶段唯一实验变量是 ImageNet 初始化。

ImageNet 预训练：

```bash
torchrun --standalone --nproc_per_node=2 dino_s2_p_server.py \
  --stage imagenet-pretrain \
  --imagenet-root /data/imagenet \
  --imagenet-batch-size 128 --imagenet-global-batch-size 256 \
  --num-workers 8 --multiprocessing-context spawn \
  --no-pin-memory --persistent-workers
```

Cityscapes S2-P：

```bash
torchrun --standalone --nproc_per_node=2 dino_s2_p_server.py \
  --stage cityscapes \
  --imagenet-checkpoint result/ImageNet_MobileNetV2_server/seed_42/imagenet_mobilenetv2_best.pth \
  --seed 42 --batch-size 2 --global-batch-size 8 \
  --num-workers 8 --multiprocessing-context spawn \
  --no-pin-memory --persistent-workers
```

ImageNet 预训练只需完成一次；S2-P 的 Cityscapes `seed=42/3407/260805` 应复用同一个已锁定的 ImageNet checkpoint，并在结果中单独标记为 `locally trained ImageNet-1K MobileNetV2`，不能与官方 torchvision 权重混称。

## 10. 当前数据为 ImageNette-10（2026-08-07）

核验 `datasets/imagenette2-320/split_protocol.json`：当前数据是 10 类 ImageNette 子集，`train=8522`、`val=947`、`test=3925`。它不能作为详单中严格定义的 ImageNet-1K S2-P；`test/` 只保留为锁定测试集，不能用于预训练或 checkpoint 选择。

入口现已支持显式的 `--stage imagenette-pretrain`。该产物标记为 `ImageNette-10`，随后用于 Cityscapes 时自动记录为 `S2-P-ImageNette`，并使用独立输出目录 `result/S2_P_ImageNette_MobileNetV2_RASPP_server/`。它必须与严格 S2-P 分开统计。

ImageNette-10 预训练：

```bash
torchrun --standalone --nproc_per_node=2 dino_s2_p_server.py \
  --stage imagenette-pretrain \
  --imagenet-root datasets/imagenette2-320 \
  --imagenet-batch-size 128 --imagenet-global-batch-size 256 \
  --num-workers 8 --multiprocessing-context spawn \
  --no-pin-memory --persistent-workers
```

Cityscapes 诊断迁移：

```bash
torchrun --standalone --nproc_per_node=2 dino_s2_p_server.py \
  --stage cityscapes \
  --imagenet-checkpoint result/ImageNette_MobileNetV2_server/seed_42/imagenette_mobilenetv2_best.pth \
  --seed 42 --batch-size 2 --global-batch-size 8 \
  --num-workers 8 --no-pin-memory --persistent-workers
```
