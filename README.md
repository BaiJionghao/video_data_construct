# Video Data Construct

面向 Rolling Forcing / streaming video 后训练的数据构建工作区。当前重点是从 VidProM 构建可控 prompt bank，并用多机队列生成 Wan teacher videos。

```text
VidProM prompts
  -> clean / filter / oversample
  -> LLM rewrite for 5s single-shot video
  -> safety post-filter
  -> final prompt bank
  -> queued teacher video generation
  -> video-level train manifest
```

## 当前状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| VidProM v2 prompt bank | 已完成 | `train=2000`、`val=200`，fallback/unusable 为 0 |
| LLM 配置 | 已配置为本地私有文件 | `configs/llm_config.json` 不入 git，模板见 `configs/llm_config.example.json` |
| 多机 teacher 生成 | 已完成入口 | 火山云自定义任务可运行固定 shell，多机共享队列抢任务 |
| v3 prompt 策略 | 设计中 | 下一步是 canonical tags、prompt quality、quota-aware sampling |
| 固定评测集 | 设计中 | 计划构建 `val_rf_stream_suite_v1`，不再把随机 val 当主评测 |

## 快速入口

### 1. 查看文档

主索引：

[docs/文档索引.md](docs/文档索引.md)

最常用文档：

- [VidProM v2 Prompt Pipeline 编排记录](docs/01_Prompt构建/VidProM_v2编排记录.md)
- [VidProM v2 Teacher 视频队列生成说明](docs/03_Teacher视频生成/VidProM_v2_Teacher视频队列生成说明.md)
- [Teacher 视频质检方案](docs/03_Teacher视频生成/Teacher视频质检方案.md)
- [RF/Streaming Prompt 评测与采样调研报告](docs/02_评测与采样/RF_Streaming_Prompt评测与采样调研报告.md)

### 2. 重新生成 VidProM v2 prompt bank

```bash
cd /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct

/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python \
  scripts/build_vidprom_prompt_pipeline.py \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v2 \
  --work-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v2_work \
  --train-count 2000 \
  --val-count 200 \
  --oversample-factor 1.30 \
  --motion-rich-ratio 0.20 \
  --workers 16 \
  --resume-stages \
  --llm-config configs/llm_config.json
```

### 3. 火山云多机生成 v2 teacher videos

火山云自定义任务里只需要运行固定 shell：

```bash
/bin/bash /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_rf_vidprom_v2_teacher_queue.sh
```

默认输入：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/train.jsonl
```

默认输出：

```text
artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train/
```

队列脚本会用共享文件系统租约锁避免重复生成，并支持任务中断后重跑。

### 4. 对 teacher videos 做自动质检

```bash
cd /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct

/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python \
  scripts/qc_teacher_videos.py \
  --dataset-dir artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train \
  --workers 16 \
  --sample-frames 9 \
  --overwrite
```

质检会输出 `qc/metadata_qc_pass.csv`、`qc/metadata_qc_review.csv` 和 `qc/metadata_qc_reject.csv`。

## 目录结构

```text
data_construct/
  configs/
    rf_prompt_taxonomy_v1.json
    llm_config.example.json
    llm_config.json              # local only, ignored

  scripts/
    build_vidprom_prompt_pipeline.py
    build_vidprom_prompt_bank.py
    extend_vidprom_prompt_bank_with_llm.py
    post_filter_llm_prompt_bank.py
    render_teacher_dataset_queue.py
    render_teacher_dataset.py
    qc_teacher_videos.py
    build_metadata.py
    verify_teacher_dataset.py

  run/
    launch_rf_vidprom_v2_teacher_queue.sh
    launch_teacher_dataset_8gpu.sh
    launch_rf_2k_teacher_proxy.sh

  docs/
    文档索引.md
    00_总览与计划/
    01_Prompt构建/
    02_评测与采样/
    03_Teacher视频生成/

  artifacts/                     # generated, ignored
```

## Git 管理口径

纳入 git：

- 代码脚本；
- 运行入口；
- 文档；
- 非敏感配置；
- 可公开参考资料。

不纳入 git：

- `artifacts/` 下的 prompt bank、teacher videos、logs；
- `configs/llm_config.json` 中的真实 API 配置；
- `__pycache__/`、临时文件、模型权重、视频输出。

## 推荐下一步

1. 用 v2 prompt bank 生成 2k teacher videos。
2. 对 teacher videos 做可用性筛选，落 video-level train manifest。
3. 构建固定评测集 `val_rf_stream_suite_v1`。
4. 落地 v3：canonical tags、prompt_quality、quota-aware sampling。
