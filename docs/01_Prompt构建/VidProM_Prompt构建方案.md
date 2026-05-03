# 基于 VidProM 的 Rolling Forcing Prompt 数据构建方案

日期：2026-05-03

本文档基于本地文件 `/vepfs-mlp2/c20250518/241506050/data/VidProm/VidProM_unique.csv`，重新梳理后续 RF/Rolling Forcing prompt 数据应该如何构建。目标是从当前“内部规则随机填充”的 v1 prompt bank，升级到更接近 Rolling Forcing/Self Forcing 做法的 `VidProM -> 过滤 -> 归一化 -> 扩写/标注 -> 分层采样` 流程。

## 1. 本地 VidProM 文件概况

本地文件：

```text
/vepfs-mlp2/c20250518/241506050/data/VidProm/VidProM_unique.csv
```

文件大小约 366MB。用 CSV parser 解析得到 1,672,243 条记录，其中非空 prompt 1,672,241 条。`wc -l` 显示约 1,807,735 行，是因为 CSV 中存在带换行的 quoted prompt，所以真实记录数应以 CSV parser 为准。

字段包括：

```text
uuid,prompt,time,toxicity,obscene,identity_attack,insult,threat,sexual_explicit
```

这对我们很有用：VidProM 已经提供了安全相关分数，可以作为第一层自动过滤依据。

## 2. 抽样统计结论

对全量 CSV 做了一次轻量统计：

| 指标 | 数值 |
| --- | ---: |
| CSV 记录数 | 1,672,243 |
| 非空 prompt | 1,672,241 |
| 字符长度 p10/p25/p50/p75/p90/p95/p99 | 22 / 37 / 66 / 127 / 247 / 380 / 845 |
| 词数 p10/p25/p50/p75/p90/p95/p99 | 4 / 6 / 11 / 21 / 39 / 58 / 126 |
| max safety score p50/p90/p95/p99/max | 0.004 / 0.077 / 0.205 / 0.687 / 0.998 |
| 含 text/font/logo/title 等文字生成倾向 | 68,600，约 4.10% |
| 含 9:16/16:9/aspect/vertical 等画幅指令 | 234,420，约 14.02% |
| 含 logo/brand/watermark | 13,276，约 0.79% |
| 含非 ASCII 字符 | 68,110，约 4.07% |

进一步粗过滤后：

| 条件 | 剩余数量 |
| --- | ---: |
| 8-80 词 | 1,097,782 |
| max safety score <= 0.05 | 1,457,025 |
| max safety score <= 0.02 | 1,281,817 |
| 不含明显 text/font/logo/title 等文字倾向 | 1,602,450 |
| 不含画幅/尺寸指令 | 1,437,387 |
| 不含 URL/附件/html img 等残留 | 1,632,983 |
| 不含 `-neg`/negative prompt/坏 anatomy 等负面提示片段 | 1,644,575 |
| 含显式运动/镜头词的 prompt | 461,681 |
| 综合过滤 A：8-80 词、安全 <=0.05、无文字倾向、无附件、无负面提示片段 | 887,141 |
| 综合过滤 B：综合过滤 A 且无画幅/尺寸指令 | 711,245 |
| 综合过滤 B 且含显式运动/镜头词 | 217,391 |

这些数字说明：VidProM 本地文件足够大，即使用较严格的规则，也有足够候选支撑 2k、16k 甚至更大规模的数据构建。

## 3. 为什么不能直接全量使用 VidProM

VidProM 的优点是接近真实用户 prompt 分布，但它不是可以直接全量投喂 teacher model 的干净训练集。抽样中能看到这些问题：

- 很多 prompt 很短，例如 `spaceship, universe`、`rock concert lights`，时间结构不足。
- 有些 prompt 是图像/logo/字体设计需求，例如要求生成某段文字、font、title、poster。
- 有不少 prompt 带平台参数，例如 `-ar 16:9`、`-motion 4`、`-neg ...`、`Message: 1 Attachment`。
- 有些 prompt 包含名人、品牌、影视角色、IP 角色或宗教/政治敏感实体。
- 有些 prompt 语法很乱，或者混入 html/img/emoji 残片。
- 很多 prompt 对视频时序要求不明确，只是静态画面描述。

所以正确路线不是“直接用 VidProM”，而是“把 VidProM 当作真实 prompt 来源，经过过滤、清洗、扩写和分层采样后再用”。

## 4. 总体构建路线

推荐将后续 prompt 数据构造成三类来源的混合：

```text
VidProM raw prompts
  -> 自动清洗
  -> 安全/版权/平台参数过滤
  -> 视频性过滤
  -> LLM/VLM 扩写和归一化
  -> 自动标签
  -> 分层采样
  -> rf_vidprom_prompt_bank_v1

内部 RF stress taxonomy
  -> 有约束规则生成
  -> 人工/LLM 校验
  -> rf_temporal_stress_prompts_v1

固定验证 prompt suite
  -> 人工设计
  -> 长期不变
  -> val_rf_temporal_suite_v1
```

最终训练 prompt bank 可以是：

- 70%：VidProM 清洗后自然分布 prompt。
- 20%：VidProM 中带显式运动/镜头/交互的 motion-rich prompt。
- 10%：内部构造的 RF temporal stress prompt。

如果目标是严格复现 Rolling Forcing，可进一步提高 VidProM 占比，并尽量使用 Rolling Forcing/Self Forcing 同源的 filtered/extended prompt 文件。

## 5. 第一阶段：清洗和过滤

### 5.1 基础清洗

对 `prompt` 字段做：

- `html.unescape`，把 `&quot;` 等 HTML entity 还原。
- 去除首尾空格、重复空格、异常换行。
- 去除明显平台残留：`Message: 1 Attachment`、`<img ...>`、URL、cdn 链接。
- 去除或单独解析平台参数：`-ar 16:9`、`--ar`、`-motion`、`-camera`、`-gs`、`-neg`。
- 对 `-neg` 后面的负面提示不要拼进正向 prompt，应丢弃或转移到 `negative_prompt_raw` 字段。

### 5.2 安全过滤

建议第一版阈值：

```text
max(toxicity, obscene, identity_attack, insult, threat, sexual_explicit) <= 0.05
```

如果后续发现脏样本仍多，可以收紧到 `<= 0.02`。按当前统计，`<=0.05` 后仍有 145.7 万条，`<=0.02` 后仍有 128.2 万条，因此收紧阈值不会造成数据不足。

### 5.3 长度过滤

建议第一版：

```text
8 <= word_count <= 80
```

原因：

- 少于 8 词通常太短，动作、场景、主体信息不足。
- 超过 80 词经常包含故事段落、多个镜头、复杂描述或平台噪声，不适合直接用于 5 秒 Wan2.1 T2V 生成。

长 prompt 不一定要全部丢弃，也可以进入 LLM 摘要队列，压缩成单镜头 5 秒视频 prompt。

### 5.4 内容过滤

建议过滤或降权：

- 明确要求生成文字、字体、logo、poster、title、watermark 的 prompt。
- 含名人、现实公众人物、品牌、影视/IP 角色的 prompt。
- 含 URL、附件、html、emoji 残片的 prompt。
- 明显不是视频生成需求的 prompt。
- 物理上不可能、语义矛盾严重、难以评估的 prompt。

画幅参数如 `16:9`、`9:16` 不一定要硬过滤。因为 Wan2.1 当前生成分辨率固定为 832x480，建议：

- 如果 prompt 内容好，只删除画幅参数；
- 如果 prompt 几乎只有画幅或平台参数，则丢弃；
- 原始参数保存在 `raw_generation_params` 字段，方便追溯。

## 6. 第二阶段：视频化扩写和归一化

VidProM 里很多 prompt 接近用户原始输入，短、散、风格化强。用于 teacher data 前，建议做一次“视频化扩写”，但不要扩得太长。

目标格式：

```text
{style}. {main_subject} {start_state} in/at {scene}. Over five seconds, {temporal_action}; by the end, {end_state}. The camera uses {camera_motion}, with {lighting}. Keep {consistency_constraint}.
```

扩写规则：

- 保留原 prompt 的主体、场景、风格意图。
- 补足一个明确的 5 秒动作过程。
- 补足镜头语言，但不要过度堆砌电影词。
- 增加一致性约束，例如主体外观、背景几何、物体位置。
- 如果原 prompt 已经很好，只做轻微清洗，不强行改写。
- 如果原 prompt 太长，把它压缩成单镜头单事件。
- 如果原 prompt 涉及多个故事事件，只保留最适合 5 秒视频的一段。

示例：

原始 prompt：

```text
a plane fly across the sky
```

扩写后：

```text
Realistic video. A small airplane enters from the left side of a clear sky and flies steadily across the frame. Over five seconds, it moves toward the distant horizon while thin clouds drift slowly behind it. The camera uses a locked wide shot with soft daylight. Keep the airplane shape and cloud positions coherent.
```

原始 prompt：

```text
leaves falling, slowly blinking, gently turning the pages, a girl is sitting under a tree with a book in her hands and reading it, gentle wind, anime style
```

扩写后：

```text
Anime-style video. A girl sits under a tree holding an open book. Over five seconds, she blinks, turns one page, and a few leaves drift down in the wind around her. The camera uses a locked medium shot with soft afternoon light. Keep her pose, the book, and the tree background consistent.
```

## 7. 第三阶段：自动标签体系

每条最终 prompt 建议保留这些字段：

```json
{
  "prompt_id": "rfvp_train_00000001",
  "split": "train",
  "lang": "en",
  "source": "vidprom_filtered_extended_v1",
  "source_uuid": "...",
  "raw_prompt": "...",
  "clean_prompt": "...",
  "prompt": "...",
  "negative_prompt": "...",
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
    "subject_type": "vehicle",
    "scene_type": "outdoor",
    "motion_type": "object_translation",
    "camera_motion": "locked_wide",
    "motion_intensity": "medium",
    "challenge_tags": ["basic_motion", "long_context_anchor"],
    "length_bucket": "medium"
  }
}
```

重点 tags：

- `source_bucket`：`vidprom_natural`、`vidprom_motion_rich`、`internal_rf_stress`。
- `subject_type`：human、animal、vehicle、object、environment、scene、abstract。
- `scene_type`：indoor、outdoor、city、nature、water、sky、studio、fantasy。
- `motion_type`：subject_translation、camera_motion、cyclic_motion、fine_interaction、environment_dynamics、multi_entity、static_low_motion。
- `camera_motion`：locked、pan、tilt、zoom、tracking、dolly、handheld、unknown。
- `motion_intensity`：low、medium、high。
- `challenge_tags`：RF 关心的时序难点。

自动标签可以先用规则/关键词做第一版，再用 LLM/VLM 做更稳的第二版。

## 8. 第四阶段：分层采样

对于 2k prompt bank，建议：

| 来源 | 比例 | 数量 |
| --- | ---: | ---: |
| VidProM filtered natural | 70% | 1,400 |
| VidProM filtered motion-rich | 20% | 400 |
| Internal RF temporal stress | 10% | 200 |

对于 16k prompt bank，建议：

| 来源 | 比例 | 数量 |
| --- | ---: | ---: |
| VidProM filtered natural | 70% | 11,200 |
| VidProM filtered motion-rich | 20% | 3,200 |
| Internal RF temporal stress | 10% | 1,600 |

如果要更贴近 Rolling Forcing 复现，可以改成：

| 来源 | 比例 |
| --- | ---: |
| VidProM filtered/extended | 90% |
| Internal RF temporal stress | 10% |

但不建议完全取消 internal stress prompts，因为它们可以保证遮挡、重现、身份保持、视差和长程一致性这些 RF 难点有足够覆盖。

## 9. 固定验证集设计

训练 prompt bank 可以随机抽样，但验证集必须固定。建议单独维护：

```text
artifacts/prompt_banks/val_rf_temporal_suite_v1/val.jsonl
```

规模建议 200-500 条，分为：

- 50 条 easy：单主体、简单背景、基础运动。
- 80 条 medium：镜头运动、周期动作、环境动态、简单交互。
- 70 条 hard：遮挡重现、身份保持、多主体、复杂背景、长上下文 anchor。

这些 prompt 不随训练集重采样变化。之后比较不同模型、不同 checkpoint、不同 RF 配置时，都用同一套 val suite。

## 10. 和当前 v1 规则填充方案的关系

当前 `rf_prompt_bank_v1` 的规则填充方案不需要立刻废弃，它仍然有三个用途：

- 作为 pipeline smoke test。
- 作为内部 RF stress prompt 的雏形。
- 作为 fallback，当 VidProM 某些 challenge category 覆盖不足时补样。

但它不应该继续作为主训练分布。更合理的位置是：

```text
VidProM = 主分布
内部有约束 taxonomy = 难例补充和可控覆盖
固定人工 val suite = 稳定评测
```

## 11. 建议落地文件结构

建议新增：

```text
configs/rf_prompt_taxonomy_v2.json
scripts/build_vidprom_prompt_bank.py
scripts/audit_prompt_bank.py
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1/
artifacts/prompt_banks/val_rf_temporal_suite_v1/
```

`build_vidprom_prompt_bank.py` 负责：

1. 读取 `VidProM_unique.csv`。
2. 清洗 raw prompt。
3. 应用安全、长度、内容过滤。
4. 可选调用 LLM 做扩写/归一化。
5. 自动打 tags。
6. 按目标比例抽样 train/val。
7. 写出 JSONL 和 stats。

`audit_prompt_bank.py` 负责：

1. 统计长度、来源、subject、motion、challenge 覆盖。
2. 检查重复 prompt。
3. 抽样输出可人工检查的 bad cases。
4. 检查是否仍有 URL、附件、`-neg`、画幅参数、文字/logo 指令。
5. 输出一份 markdown audit report。

## 12. 推荐的下一步

建议下一步先不急着重跑大规模 teacher generation，而是先做一个 `rf_vidprom_prompt_bank_v1`：

1. 从 VidProM 中过滤出候选池。
2. 先不调用 LLM，生成一个 clean-only 版本，规模 2k train + 200 val。
3. 人工抽查 100 条，确认过滤规则是否合理。
4. 再加 LLM 扩写，生成 extended 版本。
5. 对比 clean-only 和 extended 在 Wan2.1 teacher generation 中的质量。
6. 质量确认后，再用 extended 版本生成 2k 或 16k teacher data。

我的建议是：正式 RF reproduction 使用 VidProM filtered/extended 作为主数据，当前 v1 taxonomy 只作为 stress prompt 补充。这样最接近 Rolling Forcing 的做法，也能保留我们对 RF 长时序难点的主动覆盖。
