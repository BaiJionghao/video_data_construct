# Teacher 视频质检方案

日期：2026-05-03

本文档说明如何对 `rf_vidprom_v2_wan13b_teacher_train` 这批 teacher videos 做第一轮自动质检，并形成后续训练可用的 manifest。

## 1. 质检目标

视频质检不应该一上来就追求“完全判断 prompt 是否语义对齐”。第一轮更务实：

```text
先剔除明显坏样本
  -> 标出可疑样本供人工抽查
  -> 通过样本形成可训练 manifest
  -> 后续再接 VBench / VLM / 人工细评
```

当前自动质检覆盖：

- 文件是否存在；
- sidecar JSON 是否可读；
- mp4 是否可打开；
- 视频尺寸是否符合预期；
- fps / frame count / duration 是否符合预期；
- 是否黑屏、白屏、空白低细节；
- 是否过暗、过亮；
- 是否低运动或疑似冻结；
- 是否存在强全局闪烁。

自动质检不会直接判断：

- prompt-video 语义是否完全对齐；
- 主体身份是否长期一致；
- 复杂动作是否真实；
- 人脸、手部、物理细节是否足够好。

这些应该进入第二轮人工抽查或 VLM/VBench 评测。

## 2. 新增脚本

```text
scripts/qc_teacher_videos.py
```

推荐使用 Wan 环境 Python，因为默认系统 Python 没有 `cv2/imageio`：

```text
/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python
```

## 3. 快速 smoke

先跑 20 条确认脚本和依赖正常：

```bash
cd /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct

/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python \
  scripts/qc_teacher_videos.py \
  --dataset-dir artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train \
  --limit 20 \
  --workers 4 \
  --overwrite
```

## 4. 全量自动质检

```bash
cd /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct

/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python \
  scripts/qc_teacher_videos.py \
  --dataset-dir artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train \
  --workers 16 \
  --sample-frames 9 \
  --overwrite
```

默认期望：

```text
width:      832
height:     480
frames:     81
fps:        16
duration:   about 5.06s
```

如果后续生成参数变化，需要显式改：

```bash
--expected-width 832 \
--expected-height 480 \
--expected-num-frames 81 \
--expected-fps 16
```

## 5. 输出文件

默认写到：

```text
artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train/qc/
```

主要产物：

| 文件 | 作用 |
| --- | --- |
| `teacher_video_qc_summary.json` | 汇总统计 |
| `teacher_video_qc_report.md` | Markdown 报告 |
| `teacher_video_qc.jsonl` | 每条视频的详细指标 |
| `metadata_qc_pass.csv` | 自动通过样本，可直接作为第一版训练 manifest |
| `metadata_qc_review.csv` | 可疑样本，建议人工抽查 |
| `metadata_qc_reject.csv` | 自动拒绝样本 |
| `pass_prompt_ids.txt` | pass prompt_id |
| `review_prompt_ids.txt` | review prompt_id |
| `reject_prompt_ids.txt` | reject prompt_id |

## 6. 判定口径

### 6.1 Hard reject

这些问题直接拒绝：

- `missing_video`
- `video_too_small`
- `open_failed`
- `sidecar_json_error`
- `dimension_mismatch`
- `fps_mismatch`
- `frame_count_mismatch`
- `duration_mismatch`
- `too_few_decodable_frames`
- `black_or_white_video`
- `blank_low_detail_video`

这类样本不建议进训练。

### 6.2 Review warning

这些问题先进入人工复核，不默认丢弃：

- `small_video_file`
- `very_dark_video`
- `very_bright_video`
- `low_detail_video`
- `low_motion_or_freeze`
- `high_global_flicker`

原因是有些 prompt 本来就是低运动、暗场、极简画面，自动指标容易误判。对 RF/streaming 训练来说，低运动样本不是一定没用，但比例不应太高。

## 7. 训练 manifest 使用建议

最保守：

```text
只用 metadata_qc_pass.csv
```

稍微激进：

```text
metadata_qc_pass.csv
  + 人工确认后的 metadata_qc_review.csv 子集
```

不建议：

```text
直接使用原始 metadata.csv
```

因为原始 metadata 只表示视频文件生成成功，不表示视频质量适合训练。

## 8. 人工抽查建议

自动质检后建议人工看三类样本：

1. `metadata_qc_reject.csv` 全部或至少 50 条，确认阈值没有过严；
2. `metadata_qc_review.csv` 至少 100 条，决定哪些 warning 可接受；
3. `metadata_qc_pass.csv` 随机 100 条，看整体 teacher 质量和 prompt 对齐。

人工重点看：

- prompt 主体是否出现；
- 动作是否符合 prompt；
- 是否有明显闪烁、崩坏、鬼影；
- 人/动物/车辆身份是否稳定；
- camera motion 是否自然；
- 是否出现文字、logo、水印、公众人物等残留。

## 9. 后续升级

第一版自动 QC 是 low-level video health check。下一步可以继续加：

- contact sheet / HTML 人工审阅页面；
- VBench 的 subject consistency、background consistency、motion smoothness、dynamic degree；
- VLM judge 评估 prompt-video alignment；
- RF/streaming 专用的 drift、chunk boundary、identity persistence 指标。
