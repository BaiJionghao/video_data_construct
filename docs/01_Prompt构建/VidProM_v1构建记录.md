# VidProM v1 Prompt Bank 构建记录

日期：2026-05-03

本文档记录 `rf_vidprom_prompt_bank_v1` 的实际落地情况：写了什么 Python 代码、如何调用、功能是什么、生成了哪些文件、达到了什么效果，以及当前版本的局限。

## 1. 产物位置

Prompt bank 输出目录：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1/
```

包含：

```text
train.jsonl
val.jsonl
stats.json
```

当前规模：

```text
train.jsonl: 2000 条
val.jsonl:   200 条
```

输入 VidProM 文件：

```text
/vepfs-mlp2/c20250518/241506050/data/VidProm/VidProM_unique.csv
```

## 2. 新增 Python 代码

新增脚本：

```text
scripts/build_vidprom_prompt_bank.py
```

这个脚本负责从 VidProM CSV 构建 RF prompt bank。当前版本是 clean-only 版本，不调用外部 LLM，所以结果完全可复现、可离线运行。

主要功能：

1. 读取 `VidProM_unique.csv`。
2. 对 `prompt` 做基础清洗：
   - `html.unescape`；
   - 去除 URL、附件残留、`<img>`、cdn 链接；
   - 去除 `-ar`、`--ar`、`-motion`、`-camera`、`-gs` 等平台参数；
   - 去除 `-neg ...` 后的 negative prompt 片段；
   - 合并多余空格和换行。
3. 做自动过滤：
   - prompt 词数限制；
   - safety score 阈值；
   - text/font/logo/title/poster/watermark 等文字生成倾向；
   - aspect ratio / screen size / vertical / horizontal 等画幅参数；
   - negative prompt 残留；
   - 部分名人、品牌、IP、宗教和成人词 denylist。
4. 自动分桶：
   - `vidprom_natural`；
   - `vidprom_motion_rich`。
5. 自动打粗粒度 tags：
   - `subject_type`；
   - `scene_type`；
   - `motion_type`；
   - `camera_motion`；
   - `motion_intensity`；
   - `challenge_tags`；
   - `length_bucket`。
6. 按目标比例分层采样 train/val。
7. 写出 `train.jsonl`、`val.jsonl`、`stats.json`。

脚本使用 reservoir sampling 保存候选池，不会把 100 多万条候选全部读入内存。

## 3. 调用方式

本次实际调用命令：

```bash
cd /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct

python scripts/build_vidprom_prompt_bank.py \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1 \
  --train-count 2000 \
  --val-count 200 \
  --seed 20260503 \
  --motion-rich-ratio 0.20
```

默认输入路径就是：

```text
/vepfs-mlp2/c20250518/241506050/data/VidProm/VidProM_unique.csv
```

如需换路径，可加：

```bash
--vidprom-csv /path/to/VidProM_unique.csv
```

常用参数：

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `--train-count` | 2000 | train prompt 数量 |
| `--val-count` | 200 | val prompt 数量 |
| `--seed` | 20260503 | 随机采样种子 |
| `--min-words` | 8 | 最短词数 |
| `--max-words` | 80 | 最长词数 |
| `--max-safety` | 0.05 | safety 分数最大阈值 |
| `--motion-rich-ratio` | 0.20 | motion-rich prompt 占比 |
| `--keep-aspect-prompts` | false | 是否保留含画幅/尺寸提示的 prompt |

## 4. 输出 JSONL 格式

每条记录形如：

```json
{
  "prompt_id": "rfvp_train_00000000",
  "split": "train",
  "lang": "en",
  "prompt": "...",
  "negative_prompt": "...",
  "source": "vidprom_filtered_clean_v1",
  "source_uuid": "...",
  "raw_prompt": "...",
  "clean_prompt": "...",
  "license": "VidProM",
  "seed_hint": 123,
  "prompt_hash": "...",
  "safety_scores": {
    "toxicity": 0.0,
    "obscene": 0.0,
    "identity_attack": 0.0,
    "insult": 0.0,
    "threat": 0.0,
    "sexual_explicit": 0.0
  },
  "tags": {
    "source_bucket": "vidprom_motion_rich",
    "subject_type": "scene",
    "scene_type": "city",
    "motion_type": "subject_translation",
    "camera_motion": "unknown",
    "motion_intensity": "medium",
    "challenge_tags": ["basic_motion"],
    "length_bucket": "medium"
  }
}
```

其中：

- `prompt` 是清洗后的最终正向 prompt。
- `raw_prompt` 保留 VidProM 原始 prompt，方便追溯。
- `clean_prompt` 当前等同于 `prompt`，为以后加入 LLM extension 预留字段。
- `source_uuid` 对应 VidProM 原始记录的 uuid。
- `safety_scores` 保留 VidProM 原始安全分数。
- `tags` 是脚本自动生成的粗粒度标签。

## 5. 本次构建效果

全量扫描统计：

| 指标 | 数值 |
| --- | ---: |
| 读取 CSV 记录数 | 1,672,243 |
| 通过清洗/过滤且去重后候选 | 627,178 |
| reservoir 中 natural 候选 | 10,560 |
| reservoir 中 motion-rich 候选 | 2,640 |
| train 输出 | 2,000 |
| val 输出 | 200 |

主要拒绝原因：

| 原因 | 数量 |
| --- | ---: |
| 过短 | 582,760 |
| aspect/size hint | 153,657 |
| unsafe score | 128,105 |
| negative prompt fragment | 51,520 |
| text/logo request | 36,352 |
| 过长 | 34,741 |
| sensitive/IP term | 26,102 |
| attachment/url/html 残留 | 16,682 |
| 清洗后为空 | 831 |
| 重复 clean prompt | 14,315 |

Train 分布：

| 维度 | 分布 |
| --- | --- |
| source_bucket | `vidprom_natural`: 1600，`vidprom_motion_rich`: 400 |
| subject_type | scene 1043，human 546，environment 177，animal 108，vehicle 65，object 61 |
| scene_type | unknown 1296，nature 294，indoor 147，city 135，fantasy 90，water 38 |
| motion_type | static_low_motion 1477，subject_translation 176，environment_dynamics 155，camera_motion 133，fine_interaction 59 |
| motion_intensity | low 1586，medium 319，high 95 |
| word length | min 8，max 78，avg 18.96 |

Val 分布：

| 维度 | 分布 |
| --- | --- |
| source_bucket | `vidprom_natural`: 160，`vidprom_motion_rich`: 40 |
| subject_type | scene 119，human 53，environment 11，animal 7，vehicle 6，object 4 |
| motion_type | static_low_motion 145，subject_translation 20，environment_dynamics 16，camera_motion 10，fine_interaction 9 |
| word length | min 8，max 74，avg 19.91 |

## 6. 当前版本达到了什么效果

这版 `rf_vidprom_prompt_bank_v1` 已经完成了从 VidProM 到 RF prompt bank 的第一版落地：

- 主分布从内部随机规则填充切换到了真实用户 prompt 来源。
- 去掉了大量明显不适合 Wan2.1 teacher generation 的样本。
- 保留了 raw prompt 和 source uuid，后续可以追溯。
- 保留了 safety scores，后续可以收紧或复查。
- 输出格式可以直接替代当前 prompt bank 给 teacher generation 脚本使用。
- train/val 的 motion-rich 比例固定为 20%，比完全随机抽样更可控。
- `stats.json` 记录了过滤和采样结果，便于复现实验。

## 7. 当前局限

这仍然是 clean-only v1，不是最终推荐训练版。主要局限：

- 没有 LLM 扩写，所以很多 prompt 仍然偏短，缺少明确 5 秒时间结构。
- 自动 tags 是关键词规则生成，准确率有限。
- denylist 只能过滤一部分名人/IP/品牌/敏感实体，仍可能漏掉长尾实体。
- 真实用户 prompt 中有 image-style prompt，会有少量残留。
- 当前 `vidprom_natural` 占 80%，所以 `static_low_motion` 仍然偏多。
- val split 只是同源抽样，不是固定人工 RF temporal suite。

因此，这一版适合：

- 替代 v1 taxonomy prompt 做一次更真实分布的 2k teacher proxy run；
- 做人工抽查和过滤规则迭代；
- 作为 LLM extension 输入候选池；
- 作为后续 16k ODE prompt bank 的基础。

不建议把它直接当成最终 RF 训练 prompt bank。

## 8. 建议下一步

建议接下来做三件事：

1. 人工抽查 train 中 100 条、val 中 50 条，记录仍需过滤的问题类型。
2. 加一版 LLM extension，把短 prompt 归一化成单镜头 5 秒视频 prompt。
3. 单独构建 `val_rf_temporal_suite_v1`，作为长期固定评测集。

如果要马上进行 teacher generation，可以先用这个 `rf_vidprom_prompt_bank_v1` 跑一小批，例如 100-200 条，和当前 `rf_prompt_bank_v1` 的视频质量做人工对比，再决定是否扩大到 2k 或 16k。

## 9. LLM 扩写脚本

已新增 LLM 扩写脚本：

```text
scripts/extend_vidprom_prompt_bank_with_llm.py
```

使用说明见：

```text
docs/01_Prompt构建/VidProM_LLM扩写指南.md
```

该脚本读取 `rf_vidprom_prompt_bank_v1`，调用 OpenAI-compatible chat completions 接口，将 `prompt` 扩写为更适合 5 秒 Wan2.1 teacher generation 的单镜头视频 prompt。脚本支持 `--dry-run`、`--limit` 和 `--resume`。

已经完成 dry-run 链路验证：

```bash
python scripts/extend_vidprom_prompt_bank_with_llm.py \
  --input-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1 \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_dryrun \
  --splits train,val \
  --limit 3 \
  --dry-run
```

dry-run 只验证输入输出流程，不具备安全/版权/语义判断能力；正式扩写需要提供 `api_key`、`base_url` 和 `model`。
