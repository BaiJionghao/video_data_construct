# VidProM v2 Prompt Pipeline 编排记录

日期：2026-05-03

本文档记录最新版 VidProM prompt 编排方式。核心结论是：`VidProM clean-only`、`LLM 扩写`、`post-filter` 分开实现没有问题，但最终可用数量不能只在 clean-only 阶段对齐。因为 LLM reject/fallback 和后处理过滤都会继续丢样本，如果只先抽 2000/200，最后 safe bank 很可能少于目标数量。

## 1. 当前发现

已有 v1 数据的实际数量是：

| 阶段 | train | val |
| --- | ---: | ---: |
| `rf_vidprom_prompt_bank_v1` clean-only | 2000 | 200 |
| `rf_vidprom_prompt_bank_v1_llm_extended_safe` | 1953 | 197 |

也就是说，分离式流程已经产生了最终数量短缺：

```text
train 缺 47 条
val   缺 3 条
```

原因不是 clean 阶段抽样失败，而是后续阶段继续筛掉了：

- LLM reject/fallback 样本；
- 公众人物、品牌、IP 或敏感内容残留；
- 暴力、战争、灾难等不适合作为干净 teacher data 的样本；
- 正向 prompt 中的 `no text / no logos` 类否定句需要后处理。

因此，最终推荐口径应改为：

```text
先 oversample 生成候选池
  -> LLM 扩写
  -> post-filter 得到 safe pool
  -> 从 safe pool 再精确抽 train/val 目标数量
```

## 2. 新增顶层入口

新增脚本：

```text
scripts/build_vidprom_prompt_pipeline.py
```

它不替代原有三个阶段脚本，而是把它们组织成一个数量闭环：

```text
01_clean_pool:
  scripts/build_vidprom_prompt_bank.py

02_llm_extended:
  scripts/extend_vidprom_prompt_bank_with_llm.py

03_safe_pool:
  scripts/post_filter_llm_prompt_bank.py

final prompt bank:
  build_vidprom_prompt_pipeline.py 内部做 deterministic final selection
```

默认最终输出：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/
```

默认中间目录：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v2_work/
```

## 3. 数量对齐策略

新 pipeline 的关键是把 `train-count` 和 `val-count` 解释为最终 safe prompt 数量，而不是 clean-only 数量。

默认会按：

```text
pool_count = max(target_count * 1.25, target_count + 50)
```

先构建 clean pool。例如目标是 2000/200 时，默认先抽：

```text
train clean pool: 2500
val clean pool:   250
```

然后经过 LLM 和 post-filter 后，从 safe pool 中按固定 seed 抽回：

```text
train final: 2000
val final:   200
```

如果 safe pool 仍然不足，脚本会直接失败并提示提高 `--oversample-factor` 或 `--min-oversample-extra`，不会静默写出短缺版本。

## 4. 字段和追溯

最终 JSONL 会重新编号，保证 teacher generation 侧文件名连续：

```text
rfvp2_train_00000000
rfvp2_train_00000001
...
```

同时保留来源 ID：

```json
{
  "prompt_id": "rfvp2_train_00000000",
  "source_prompt_id": "rfvppool_train_00001234",
  "pre_selection_source": "vidprom_filtered_llm_extended_v1",
  "source": "vidprom_filtered_llm_extended_safe_v2",
  "prompt_program_version": "rf_vidprom_prompt_pipeline_v2"
}
```

最终目录会写出：

```text
train.jsonl
val.jsonl
pipeline_stats.json
```

`pipeline_stats.json` 记录目标数量、safe pool 数量、最终抽样数量、motion-rich 覆盖、各阶段 stats 和 warnings。

## 5. 推荐正式命令

LLM 配置推荐放在：

```text
configs/llm_config.json
```

支持的字段名：

```json
{
  "LLM_BASE_MODEL": "gpt-5-mini",
  "LLM_API_URL": "http://host:port/v1",
  "LLM_API_KEY": "..."
}
```

脚本也兼容小写字段 `model`、`base_url`、`api_key`。`LLM_API_KEY` 不会作为命令行参数传给子进程，避免出现在 shell 历史或进程命令里。

生成 2k + 200 final safe prompt bank：

```bash
cd /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct

python scripts/build_vidprom_prompt_pipeline.py \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v2 \
  --work-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v2_work \
  --train-count 2000 \
  --val-count 200 \
  --oversample-factor 1.25 \
  --motion-rich-ratio 0.20 \
  --workers 16 \
  --llm-config configs/llm_config.json
```

如果开启 lineage 检查后拒绝更多样本，可以提高到：

```bash
--oversample-factor 1.40
```

或：

```bash
--min-oversample-extra 200
```

## 6. 可复用模式

如果已经有一个 safe pool，只想从里面精确抽最终数量：

```bash
python scripts/build_vidprom_prompt_pipeline.py \
  --finalize-only \
  --safe-pool-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended_safe \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended_safe_final1900 \
  --train-count 1900 \
  --val-count 190
```

如果按 2000/200 从当前 v1 safe pool 抽，脚本会失败，因为 safe pool 只有 1953/197。这是预期行为。

## 7. 本次代码调整

本次除了新增 pipeline 外，还做了三个小修正：

1. `scripts/build_vidprom_prompt_bank.py`
   - `ASPECT_RE` 现在能识别 `--ar9:16`、`-ar16:9` 等无空格画幅参数。
2. `scripts/extend_vidprom_prompt_bank_with_llm.py`
   - 并发 LLM 时不再在 worker 线程里直接写 `stats` 的 fallback reason，避免统计竞争。
3. `scripts/post_filter_llm_prompt_bank.py`
   - 新增 `--check-lineage-text/--no-check-lineage-text`；
   - 可选检查 `raw_prompt`、`clean_prompt`、`pre_llm_prompt`；
   - 更稳地清理正向 prompt 中的 `no text / no logos / no signage` 类否定句。
4. `scripts/extend_vidprom_prompt_bank_with_llm.py` 和 `scripts/build_vidprom_prompt_pipeline.py`
   - 新增 `--llm-config`；
   - 默认会读取 `configs/llm_config.json`；
   - 兼容 `LLM_BASE_MODEL`、`LLM_API_URL`、`LLM_API_KEY`；
   - pipeline 只把 config 路径传给扩写子进程，不把 API key 写入命令行。

## 8. 烟测记录

已完成一次不调用外部 LLM 的 dry-run 烟测：

```bash
python scripts/build_vidprom_prompt_pipeline.py \
  --output-dir /tmp/rf_prompt_pipeline_smoke_final \
  --work-dir /tmp/rf_prompt_pipeline_smoke_work \
  --train-count 6 \
  --val-count 3 \
  --oversample-factor 4 \
  --min-oversample-extra 0 \
  --workers 2 \
  --dry-run \
  --overwrite-existing
```

结果：

```text
clean pool: train 24, val 12
safe pool:  train 23, val 12
final:      train 6,  val 3
```

同时通过：

```bash
python -m py_compile \
  scripts/build_vidprom_prompt_bank.py \
  scripts/extend_vidprom_prompt_bank_with_llm.py \
  scripts/post_filter_llm_prompt_bank.py \
  scripts/build_vidprom_prompt_pipeline.py
```

## 9. 当前建议

后续 teacher generation 不建议直接使用：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended_safe/
```

原因是它数量不满 2000/200，且 prompt_id 有过滤后的缺口。建议用新版 pipeline 生成：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/
```

再把 teacher generation 的 `PROMPT_BANK_DIR` 指向 v2 目录。

## 10. 2026-05-03 v2 重新生成状态

已完成 v2 重新生成：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/
```

本次命令：

```bash
python scripts/build_vidprom_prompt_pipeline.py \
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

本次参数：

```text
target final: train 2000, val 200
clean pool:   train 2600, val 260
oversample:   1.30x
seed:         20260503
motion-rich:  20%
LLM model:    gpt-5-mini
workers:      16
```

阶段行数：

```text
clean pool:   train 2600, val 260
LLM output:   train 2600, val 260
safe pool:    train 2522, val 248
final output: train 2000, val 200
```

LLM 扩写统计：

| split | input | extended | reject/fallback |
| --- | ---: | ---: | ---: |
| train | 2600 | 2563 | 37 |
| val | 260 | 255 | 5 |

post-filter 统计：

| split | input | kept | rejected | scrubbed no-text |
| --- | ---: | ---: | ---: | ---: |
| train | 2600 | 2522 | 78 | 58 |
| val | 260 | 248 | 12 | 4 |

最终输出校验：

```text
train.jsonl: 2000 条，prompt_id rfvp2_train_00000000 - rfvp2_train_00001999
val.jsonl:   200 条，prompt_id rfvp2_val_00000000 - rfvp2_val_00000199
fallback:    0
unusable:    0
warnings:    []
```

并行数建议：

- `--workers 16`：当前推荐，速度和接口稳定性比较平衡。
- `--workers 24`：如果 16 没有 429/timeout，可以尝试。
- `--workers 32`：只建议在确认 endpoint 限流和并发稳定后使用。

如果遇到 429、连接重置或大量 timeout，就降回 `--workers 8` 或增加 `--sleep 0.03`。

## 11. 训练集使用建议

需要区分三个概念：

```text
prompt bank:
  给 teacher generation 使用的 prompt 清单。

teacher video set:
  用 prompt 生成出来的视频候选。

train manifest:
  通过视频质检后，真正喂给 RF/streaming 训练的数据清单。
```

当前 `rf_vidprom_prompt_bank_v2/train.jsonl` 适合作为第一轮 2k teacher generation 的训练 prompt source，但它还不是最终训练集。最终训练集应该在视频生成之后再做一次筛选，把 teacher 生成失败、动作不对齐、主体严重漂移、闪烁严重或安全/IP 残留的样本剔除。

短期建议：

```text
1. 用 rf_vidprom_prompt_bank_v2/train.jsonl 生成 2000 条 teacher video。
2. 对 teacher video 做可用性筛选，形成 train manifest。
3. v2/val.jsonl 只作为 dev sanity，不作为主评测。
4. 主评测单独构建 val_rf_stream_suite_v1，并禁止进入 train。
```

如果要扩大到 16k/20k，建议不要继续只调大 `--train-count` 随机抽样，而是升级到 v3 采样策略：

```text
70% VidProM natural safe prompts
20% VidProM RF challenge-enriched prompts
10% internal RF stress prompts
```

其中 challenge-enriched 应该按规范化 RF 维度设置最低覆盖：

| 维度 | train 最低比例建议 |
| --- | ---: |
| camera_parallax | >= 10% |
| cyclic_motion | >= 8% |
| fine_interaction | >= 8% |
| environment_dynamics | >= 8% |
| multi_entity | >= 6% |
| occlusion_reentry | >= 5% |
| identity_persistence | >= 5% |
| physics_material | >= 5% |
| long_context_anchor | >= 5% |

这些标签可以重叠，所以比例总和可以超过 100%。训练集应保留真实用户主分布，但要保证 RF/streaming 关键难例不是“碰运气出现”。

对 Rolling Forcing 类任务，推荐训练数据也拆成两类：

| 类型 | 用途 | 建议 |
| --- | --- | --- |
| short teacher clips | 学基础 text-video alignment、动作质量和主体一致性 | 主分布优先，单 prompt 通常 1 seed |
| streaming continuation units | 学历史条件、长程一致性和 chunk 边界稳定 | 从 hard RF 子集生成多 chunk 或长视频，并记录 chunk index/history length |

下一版训练集构建不应直接把 prompt bank 当训练 manifest，而应新增一个视频级 manifest，例如：

```json
{
  "sample_id": "rftrain_v3_00000001",
  "prompt_id": "rfvp3_train_00000001",
  "prompt": "...",
  "teacher_video_path": "...",
  "teacher_model": "...",
  "seed": 12345,
  "canonical_tags": {
    "motion_type": "camera_motion",
    "rf_challenge_tags": ["camera_parallax", "identity_persistence"]
  },
  "video_quality": {
    "usable": true,
    "alignment_score": 0.82,
    "temporal_consistency_score": 0.80,
    "reject_reason": ""
  },
  "sample_weight": 1.0,
  "split": "train"
}
```

因此当前实际路线是：

```text
v2:
  先用于 2k teacher generation / 小规模训练闭环。

v3:
  增加 canonical tags、prompt_quality、quota-aware sampling，
  构建 16k-20k train prompt bank。

train manifest:
  在 teacher video 生成和视频质检之后再落盘，
  这是最终训练真正读取的文件。
```

## 12. v2 teacher video 多机生成入口

为了在火山云自定义任务中多机并发生成 v2 teacher videos，已新增队列式渲染入口：

```text
run/launch_rf_vidprom_v2_teacher_queue.sh
```

火山云自定义任务只需要运行固定 shell：

```bash
/bin/bash /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct/run/launch_rf_vidprom_v2_teacher_queue.sh
```

该入口默认读取：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/train.jsonl
```

默认输出：

```text
artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train/
```

它使用共享文件系统上的租约锁：

```text
locks/<prompt_id>.lock/lease.json
```

每个 worker 抢到一个 prompt 后生成对应视频，生成期间定期 heartbeat。任务被中断后锁会在 `QUEUE_LEASE_SECONDS=1800` 后变成 stale，其他 worker 会自动接管，所以不会永久死锁。最终视频和 sidecar 都先写入 `tmp/`，再用原子替换提交到 `videos/` 和 `records/`。

详细说明见：

```text
docs/03_Teacher视频生成/VidProM_v2_Teacher视频队列生成说明.md
```
