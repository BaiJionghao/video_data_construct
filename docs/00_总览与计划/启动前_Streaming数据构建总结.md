# 启动前总结：Wan1.3B 流式后训练数据构建

## 1. 当前目标

这次不是从零训练视频基础模型，而是为 `Wan2.1-T2V-1.3B` 的流式后训练准备第一版可直接开跑的数据资产。

目标拆成两层：

1. 先得到一套 **立即可训练** 的 `video,prompt` 数据集；
2. 同时把目录结构和 sidecar 信息设计成 **后续可升级到 Self Forcing / Rolling Forcing 风格蒸馏**。

## 2. 为什么现在采用 synthetic teacher-data 路线

我最终没有把主路线放在 `OpenVid / VideoUFO / Pexels / Pixabay / CC 视频抓取` 上，原因是：

- 你最在意的是版权和可控性；
- 你现在已有本地 `Wan2.1-T2V-1.3B` checkpoint；
- 论文里的 `Rolling Forcing` 本质上就是 prompt-driven post-training，而不是重新依赖海量真视频；
- 真实视频数据源即使表面许可更宽松，仍然有上传者权限、肖像权、商标和上游 license 继承的问题。

所以当前主决策是：

- 主训练资产：`prompt bank + teacher-generated videos + sidecar manifests`
- 真实视频：后续只做小规模补充，不做主料仓

## 3. 本地资源判断

已确认：

- GPU：`8 x NVIDIA A100-SXM4-80GB`
- 当前基本空闲
- 本地模型已存在：
  - `/vepfs-mlp2/c20250518/241506050/ckpts/wan21/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors`
  - `/vepfs-mlp2/c20250518/241506050/ckpts/wan21/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth`
  - `/vepfs-mlp2/c20250518/241506050/ckpts/wan21/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`
  - `/vepfs-mlp2/c20250518/241506050/ckpts/wan21/Wan2.1-T2V-1.3B/google/umt5-xxl`

这意味着：

- 不需要先下载 teacher 权重
- 可以直接做 8 卡并行 teacher dataset 生成

## 4. 本次已经落实的工程骨架

### 4.1 Prompt taxonomy

新增：

- `data_construct/configs/rf_prompt_taxonomy_v1.json`

作用：

- 用内部组合式 taxonomy 生成 prompt bank
- 避免直接依赖外部 prompt 数据集 license
- 保持 subject / scene / motion / camera / style 的可控覆盖

### 4.2 Prompt bank 生成脚本

新增：

- `data_construct/scripts/build_prompt_bank.py`

作用：

- 生成 `train.jsonl` 和 `val.jsonl`
- 默认规模：
  - `train=20000`
  - `val=1000`
- 每条记录包含：
  - `prompt_id`
  - `prompt`
  - `negative_prompt`
  - `seed_hint`
  - `tags`

### 4.3 8 卡 teacher 视频生成脚本

新增：

- `data_construct/scripts/render_teacher_dataset.py`

作用：

- 读取 prompt bank
- 通过 `torchrun --nproc_per_node=8` 自动分片
- 每个 rank 独占 1 张 GPU
- 使用本地 `Wan2.1-T2V-1.3B` 生成视频
- 输出：
  - `videos/*.mp4`
  - `records/*.json`
  - `logs/manifest_rankXX.jsonl`
  - `logs/failures_rankXX.jsonl`

设计重点：

- 支持断点续跑
- 已生成样本默认跳过
- 支持 `--max-items` 做 smoke test

### 4.4 metadata 汇总与校验

新增：

- `data_construct/scripts/build_metadata.py`
- `data_construct/scripts/verify_teacher_dataset.py`

作用：

- 生成 `metadata.csv`
- 汇总可直接喂给当前 `DiffSynth-Studio` 的训练入口
- 校验视频、records、失败日志和缺失文件

### 4.5 一键 8 卡启动脚本

新增：

- `data_construct/run/launch_teacher_dataset_8gpu.sh`

作用：

- 自动：
  1. 建 prompt bank
  2. 8 卡并行生成
  3. 汇总 metadata
  4. 运行校验

## 5. 当前数据格式决策

第一阶段输出格式定为：

```text
data_construct/artifacts/datasets/wan13b_teacher_v1_train/
  videos/
  records/
  logs/
  metadata.csv
```

其中：

- `metadata.csv` 是当前训练最直接可用格式
- `records/*.json` 保留更多信息，方便后面升级成 RF 风格蒸馏

## 6. 为什么先生成 mp4，不先存大规模 latent cache

这是一个刻意的取舍：

- `mp4 + prompt` 现在就能直接训练
- 大规模 latent / ODE cache 会迅速放大存储成本
- 在训练器还没有完全升级成 `Rolling Forcing` 之前，先大规模缓存 latent 的收益不高

所以当前顺序是：

1. 先把 teacher video dataset 跑起来
2. 再挑热点子集补 latent / rollout cache
3. 再接更忠实的 `Self Forcing / Rolling Forcing` loop

## 7. 生成参数决策

当前 teacher 生成参数定为：

- model: `Wan2.1-T2V-1.3B`
- resolution: `832x480`
- frames: `81`
- fps: `16`
- `num_inference_steps=50`
- `cfg_scale=6.0`
- `sigma_shift=8.0`
- `tiled=True`

原因：

- 贴近 `Wan2.1-1.3B` 的推荐使用区间
- 与 `Rolling Forcing` 论文的 5 秒 480P 设定接近
- 对后续流式 student post-training 更一致

## 8. 吞吐与规模预估

目前已经拿到两组 smoke 基线：

- `1 worker/GPU`:
  - 8 卡并行，8 条样本
  - 平均单样本 `168.7s`
  - 折算总体吞吐约 `170.7 samples/hour`
- `2 workers/GPU`:
  - 16 worker 并行，16 条样本
  - 平均单样本 `383.3s`
  - 折算总体吞吐约 `150.3 samples/hour`

因此当前机器和参数下，正式生成更优先采用：

- `NPROC_PER_NODE=8`
- `WORKERS_PER_GPU=1`

补充 benchmark：

- 已新增 `docs/03_Teacher视频生成/模型生成延迟基准.md`
- 其中记录了 `Wan2.1-T2V-1.3B`、`Wan2.1-T2V-14B`、`Helios-Distilled` 的单卡单条样本生成耗时对比。
- 在 `832x480, 81 frames, 50 steps` 下：
  - `Wan2.1-T2V-1.3B`: `166.139s`
  - `Wan2.1-T2V-14B`: `848.428s`
  - 14B 约为 1.3B 的 `5.1x` 时间成本。
- `Helios-Distilled` 在 `640x384, 99 frames` 官方尺寸下端到端约 `60s`，但当前本地脚本在 `832x480` 下未跑通。

也就是说：

- 不追求单卡把 80G 显存“塞满”
- 追求 8 卡总体 `samples/hour` 最高
- 当前看 `1 worker/GPU` 比 `2 workers/GPU` 更划算

### 8.1 壁钟时间估算

按 `170.7 samples/hour` 粗估：

- `1024` 条约 `6.0h`
- `2048` 条约 `12.0h`
- `4096` 条约 `24.0h`
- `5000` 条约 `29.3h`
- `8192` 条约 `48.0h`
- `20000` 条约 `117.1h`，约 `4.9` 天
- `21000` 条约 `123.0h`，约 `5.1` 天

### 8.2 磁盘占用估算

根据当前 smoke 结果，单条 `mp4` 平均约 `0.39 MB`。这个值会随画面复杂度波动，但可作为首估：

- `4096` 条视频约 `1.6 GB`
- `20000` 条视频约 `7.8 GB`
- `21000` 条视频约 `8.2 GB`

再加上 `records/*.json` 与日志，整体仍处于比较轻量的量级。

## 9. 启动策略

为了不把 8 卡机器直接押到一个长任务上而没有校验，我准备按两步启动。

### 9.1 第一步：GPU 负载与吞吐 benchmark

目标：

- 用短任务复核 `1 worker/GPU` 与 `2 workers/GPU` 的壁钟吞吐
- 观察 GPU util、显存占用和稳定性
- 以真实 `samples/hour` 选择正式参数

建议规模：

- `1 worker/GPU`: `MAX_ITEMS=16`
- `2 workers/GPU`: `MAX_ITEMS=32`

这样每个 worker 至少处理 2 条样本，比单条 smoke 更能反映稳态吞吐。

### 9.2 第二步：首批正式集

当前更合适的第一批正式集不是直接 `20000`，而是先启动一个 **bootstrap set**：

- `train=4096`
- `val=256`

目标：

- 先得到一批足够做首轮流式后训练的数据
- 把训练链路、样本质量、loss 行为先跑通
- 如果训练和验证都正常，再无缝扩展到 `20000+1000`

原因：

- `4096` 条大约 `24h` 可完成，反馈周期更友好
- 对第一轮 student post-training 已经足够看趋势
- 避免一上来就押上 5 天机器时间后才发现 prompt 分布或样本质量需要调

### 9.3 第三步：扩展到 v1 全量集

当 bootstrap set 验证通过后，再扩展到：

- `train=20000`
- `val=1000`

这会是当前方案下较合理的 v1 正式规模。

## 10. 启动命令

### 10.1 Benchmark: 1 worker/GPU

```bash
PROMPT_BANK_DIR=/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/prompt_banks/rf_prompt_bank_v1 \
DATASET_ROOT=/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/datasets/wan13b_teacher_v1_bench_w1g1 \
MAX_ITEMS=16 \
NPROC_PER_NODE=8 \
/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_teacher_dataset_8gpu.sh
```

### 10.2 Benchmark: 2 workers/GPU

```bash
PROMPT_BANK_DIR=/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/prompt_banks/rf_prompt_bank_v1 \
DATASET_ROOT=/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/datasets/wan13b_teacher_v1_bench_w2g2 \
MAX_ITEMS=32 \
NPROC_PER_NODE=8 \
WORKERS_PER_GPU=2 \
/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_teacher_dataset_8gpu.sh
```

### 10.3 Bootstrap full run

```bash
PROMPT_BANK_DIR=/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/prompt_banks/rf_prompt_bank_v1 \
DATASET_ROOT=/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/datasets/wan13b_teacher_v1_bootstrap4096 \
TRAIN_COUNT=4096 \
VAL_COUNT=256 \
MAX_ITEMS=4096 \
NPROC_PER_NODE=8 \
/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_teacher_dataset_8gpu.sh
```

### 10.4 v1 full run

```bash
PROMPT_BANK_DIR=/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/prompt_banks/rf_prompt_bank_v1 \
DATASET_ROOT=/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/datasets/wan13b_teacher_v1_train20k \
TRAIN_COUNT=20000 \
VAL_COUNT=1000 \
MAX_ITEMS=20000 \
NPROC_PER_NODE=8 \
/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_teacher_dataset_8gpu.sh
```

## 11. 成功标准

这次启动后，我会重点关注：

1. 8 卡是否都稳定进入推理
2. 是否能稳定写出 `mp4` 和 `record json`
3. `metadata.csv` 是否能无缝生成
4. 样本质量是否达到“可做第一轮 student post-training”的水平

如果 smoke test 正常，后面最直接的下一步就是：

- 启动 full run
- 然后基于产出的 `metadata.csv` 去接 `DiffSynth-Studio/examples/wanvideo/model_training/train.py`

## 12. 备注

这里总结的是高层工程推理、取舍和执行顺序，不是逐 token 的内部思维转录。这样更适合作为项目文档，也更方便后续继续扩展和复现。
