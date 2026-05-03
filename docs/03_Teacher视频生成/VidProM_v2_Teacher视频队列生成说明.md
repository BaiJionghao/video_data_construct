# RF VidProM v2 Teacher Video 队列式生成说明

日期：2026-05-03

本文档记录如何用多台火山云自定义任务共同生成 `rf_vidprom_prompt_bank_v2/train.jsonl` 对应的 teacher videos。核心目标是：所有机器运行同一份固定 shell，自动从共享 prompt bank 抢任务，避免大面积重复生成，并且任务中断后可以安全重跑。

## 1. 新增文件

新增队列式渲染脚本：

```text
scripts/render_teacher_dataset_queue.py
```

新增火山云固定入口：

```text
run/launch_rf_vidprom_v2_teacher_queue.sh
```

火山云自定义任务里只需要运行：

```bash
/bin/bash /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_rf_vidprom_v2_teacher_queue.sh
```

脚本内部使用绝对路径 Python：

```text
/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python
```

因此不依赖自定义任务环境里是否能直接找到 `python`。

## 2. 默认输入输出

默认 prompt bank：

```text
/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/train.jsonl
```

默认输出目录：

```text
/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train/
```

输出结构：

```text
rf_vidprom_v2_wan13b_teacher_train/
  videos/
    rfvp2_train_00000000.mp4
    ...
  records/
    rfvp2_train_00000000.json
    ...
  locks/
    *.lock/
  logs/
    queue_worker_*.jsonl
  worker_logs/
    *.log
  tmp/
  metadata.csv
  queue_status.json
```

一条样本只有在同时存在以下两个文件时才算完成：

```text
videos/<prompt_id>.mp4
records/<prompt_id>.json
```

如果任务中断时留下了半截临时文件，后续重跑不会把它当成完成样本。

## 3. 多机并发机制

每个 prompt 是一个原子任务。worker 抢任务时会创建：

```text
locks/<prompt_id>.lock/
```

目录创建是共享文件系统上的原子操作，因此多台机器同时抢同一个 prompt 时，通常只有一个会成功。

锁目录里有：

```text
lease.json
```

包含：

- `worker_id`
- `hostname`
- `pid`
- `token`
- `updated_at`
- `expires_at`
- `lease_seconds`

worker 生成视频期间会定期更新 heartbeat。默认：

```text
QUEUE_LEASE_SECONDS=1800
QUEUE_HEARTBEAT_SECONDS=30
```

如果火山云任务被中断，heartbeat 会停止。超过 `QUEUE_LEASE_SECONDS` 后，其他 worker 会把旧锁移动到：

```text
locks/_stale/
```

然后重新接管该 prompt。这样不会出现永久死锁。

## 4. 重复生成和原子提交

重复生成可能在两种情况下发生：

1. 共享文件系统延迟导致两个 worker 短时间内都认为自己可以生成；
2. 某个旧 worker 被判断为 stale 后又恢复执行。

脚本对重复生成的处理是：

- 生成阶段先写入 `tmp/<worker_id>/`；
- 生成结束后检查自己是否仍持有锁；
- 如果锁已经失效，就丢弃临时文件；
- 如果目标样本已经由其他 worker 完成，也丢弃临时文件；
- 否则用 `os.replace` 原子提交视频和 sidecar。

因此偶发重复最多浪费少量算力，不会破坏最终数据集。最终以 `records/` 和 `videos/` 的同名文件为准。

## 5. 默认正式命令

火山云自定义任务使用固定 shell：

```bash
/bin/bash /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_rf_vidprom_v2_teacher_queue.sh
```

默认会：

- 自动检测本机 GPU 数量；
- 每张 GPU 启动 1 个 worker；
- 使用 Wan2.1-T2V-1.3B；
- 生成 `480x832`、`81 frames`、`16 fps`、`50 steps` 的 mp4；
- 所有 worker 一直运行到全局 2000 条完成；
- 完成后写 `metadata.csv` 并运行 `verify_teacher_dataset.py`。

## 6. 常用环境变量

如果需要覆盖配置，在火山云自定义任务里设置环境变量即可。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `WAN_PYTHON` | `/vepfs-mlp2/.../miniconda3/envs/wan/bin/python` | Python 绝对路径 |
| `MODEL_ROOT` | `/vepfs-mlp2/.../ckpts/wan21/Wan2.1-T2V-1.3B` | Wan checkpoint |
| `PROMPT_BANK_PATH` | v2 `train.jsonl` | prompt bank |
| `DATASET_ROOT` | `artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train` | 输出目录 |
| `NPROC_PER_NODE` | `auto` | 每台机器使用几张 GPU |
| `WORKERS_PER_GPU` | `1` | 每张 GPU 启几个 worker |
| `BATCH_SIZE` | `1` | 单次 forward prompt 数 |
| `MAX_ITEMS` | 空 | smoke test 时限制前 N 条 |
| `QUEUE_LEASE_SECONDS` | `1800` | 锁多久没 heartbeat 后视为 stale |
| `QUEUE_HEARTBEAT_SECONDS` | `30` | heartbeat 间隔 |
| `QUEUE_IDLE_SLEEP` | `10` | 没抢到任务时等待秒数 |
| `QUEUE_MAX_JOBS_PER_WORKER` | `0` | 单 worker 最多生成几条，0 表示不限 |
| `RUN_FINALIZE` | `1` | 完成后是否生成 metadata 并 verify |
| `DRY_RUN` | `0` | 只测试队列/锁，不真正生成视频 |

示例：只用 4 张 GPU：

```bash
NPROC_PER_NODE=4 /bin/bash /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_rf_vidprom_v2_teacher_queue.sh
```

示例：smoke test 前 16 条，不加载模型，只测试锁和恢复逻辑：

```bash
MAX_ITEMS=16 DRY_RUN=1 DATASET_ROOT=/tmp/rf_vidprom_v2_queue_smoke \
  /bin/bash /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_rf_vidprom_v2_teacher_queue.sh
```

注意：`DRY_RUN=1` 会写 placeholder mp4，只用于测试队列，不要用于正式数据目录。

## 7. 状态检查

查看当前完成进度：

```bash
/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python \
  /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/scripts/render_teacher_dataset_queue.py \
  --prompt-bank /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/train.jsonl \
  --output-dir /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train \
  --status-only
```

输出中重点看：

```json
{
  "target_count": 2000,
  "complete": 1234,
  "incomplete": 766,
  "active_locks": 8,
  "stale_locks": 0,
  "done": false
}
```

如果发现 stale lock，也可以手动清理：

```bash
/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python \
  /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/scripts/render_teacher_dataset_queue.py \
  --prompt-bank /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/train.jsonl \
  --output-dir /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train \
  --cleanup-stale-locks \
  --status-only
```

正常情况下不需要手动清理，因为 worker 抢任务时会自动接管 stale lock。

## 8. 中断后的重跑

如果火山云任务被杀，直接重新提交同一个固定 shell 即可：

```bash
/bin/bash /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_rf_vidprom_v2_teacher_queue.sh
```

脚本会自动跳过已完成样本，只补未完成样本。未完成定义是：

- 没有视频；
- 或没有 sidecar；
- 或 sidecar JSON 损坏；
- 或视频文件过小。

不需要恢复生成一半的视频；这类样本会重新生成。

## 9. 完成后的校验

默认 shell 会在全局完成后运行：

```bash
build_metadata.py --require-video
verify_teacher_dataset.py
```

也可以手动运行：

```bash
/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python \
  /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/scripts/build_metadata.py \
  --dataset-dir /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train \
  --require-video

/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python \
  /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/scripts/verify_teacher_dataset.py \
  --dataset-dir /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train
```

期望最终：

```text
video_count: 2000
record_count: 2000
missing_videos_count: 0
metadata_exists: true
```

## 10. 当前建议

正式生成 v2 teacher videos 时使用：

```text
run/launch_rf_vidprom_v2_teacher_queue.sh
```

不要用旧的 `render_teacher_dataset.py` rank 切片方式跑多台火山云机器。旧脚本适合单节点固定 `WORLD_SIZE`，多台自定义任务同时运行时容易重复覆盖同一批 rank；新的 queue 脚本才适合共享目录上的多机抢任务模式。
