# VidProM v1 LLM 扩写质量报告

日期：2026-05-03

本文档记录使用 `gpt-5-mini` 对 `rf_vidprom_prompt_bank_v1` 做 LLM 扩写后的结果、质量分析和后续使用建议。

## 1. 本次运行

输入：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1/
```

输出：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended/
```

后处理安全版：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended_safe/
```

后处理脚本：

```text
scripts/post_filter_llm_prompt_bank.py
```

运行方式：

```bash
python scripts/extend_vidprom_prompt_bank_with_llm.py \
  --input-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1 \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended \
  --splits train,val \
  --temperature 0.2 \
  --workers 8 \
  --resume
```

说明：

- 模型：`gpt-5-mini`
- 温度：`0.2`
- 并发：`8`
- 为兼容 `gpt-5-mini`，脚本已把默认 `max_tokens` 提高到 `1000`，并默认使用 `reasoning_effort=minimal`。
- 正式跑完后，我已将脚本中的硬编码 key 移除，恢复为环境变量/命令行传参方式，避免密钥留在代码里。

## 2. LLM 扩写结果

`llm_extension_stats.json` 统计：

| split | 输入 | 成功扩写 | reject/fallback | 写出 |
| --- | ---: | ---: | ---: | ---: |
| train | 2000 | 1966 | 34 | 2000 |
| val | 200 | 198 | 2 | 200 |

成功扩写率：

- train：98.3%
- val：99.0%
- 总体：98.36%

reject/fallback 的主要原因是原始 VidProM prompt 中仍有真实公众人物、版权角色、暴力/灾难/性暗示等内容。这个结果是好事，说明 mini 模型确实能识别出一部分规则 denylist 没过滤掉的长尾问题。

## 3. Prompt 结构质量

LLM 扩写前后的词数变化：

| split | 扩写前平均词数 | 扩写后平均词数 | 扩写后范围 | 扩写后 p10/p50/p90 |
| --- | ---: | ---: | --- | --- |
| train | 18.96 | 51.76 | 8-72 | 47 / 52 / 58 |
| val | 19.91 | 52.26 | 9-69 | 47 / 53 / 58 |

结构覆盖统计：

| split | 含 5 秒时间结构 | 含 camera | 含 lighting | 含 consistency/coherent/keep |
| --- | ---: | ---: | ---: | ---: |
| train accepted | 1834 / 1966 | 1024 / 1966 | 1776 / 1966 | 1918 / 1966 |
| val accepted | 186 / 198 | 100 / 198 | 176 / 198 | 193 / 198 |

定性看，LLM 扩写后的 prompt 明显优于 clean-only 版本：

- 大多数 prompt 被改成了单镜头 5 秒视频描述。
- 动作、镜头、光照、一致性约束更加明确。
- prompt 长度集中在 50 词左右，适合 Wan2.1 teacher generation。
- 许多抽象/短 prompt 被转成了可生成的视频场景。

例如：

```text
原始：a girl is sitting in a bus station near the street and then one bus is coming

扩写：A teenage girl sits on a bench at a city bus stop beside a busy street; she looks at her phone as a single city bus approaches and pulls up. 5-second single-shot, slow dolly-in from slight wide to medium as bus enters frame. Soft late-afternoon light, shallow depth of field, keep girl and bus framing consistent.
```

```text
原始：Clumsy robots trying to cook in a kitchen, with comedic mishaps, in a cartoon style

扩写：A single-shot 5-second cartoon-style scene of two clumsy robots attempting to cook in a bright home kitchen, one juggling eggs while the other accidentally flips a pancake; playful small mishaps, light physical comedy. Medium close-up, slow handheld pan to follow action, warm soft morning lighting, consistent character proportions and simple cel-shaded colors.
```

## 4. 发现的问题

LLM 扩写版仍然不能直接无脑用于 teacher generation，主要问题有四类。

### 4.1 fallback 样本不应直接使用

train 有 34 条 fallback，val 有 2 条 fallback。fallback 当前会保留原始 clean prompt，因此里面可能仍包含公众人物、版权角色或敏感内容。

结论：teacher generation 时应排除 `llm_extension.used_fallback_prompt=true` 的样本。

### 4.2 个别扩写仍残留 IP/公众人物/风格名

例如有一条把原始 `Tim Burton film style` 扩成了 `Tim Burton-inspired style`。这说明 mini 模型大多数时候能拒绝或泛化，但不是 100% 稳。

结论：还需要后处理规则或二次审核。

### 4.3 少量暴力/敏感场景被改写成了可生成 prompt

例如原始 prompt 包含城市轰炸、燃烧城市、人物逃跑，模型将其改写成了更清晰的视频 prompt。这对普通视频生成可能可用，但不适合我们构造干净 teacher dataset。

结论：RF teacher 数据最好偏中性，避免战争、血腥、灾难、政治符号等复杂敏感语义。

### 4.4 正向 prompt 里偶尔出现否定句

有些扩写会写出 `ensure no readable text, logos, or signage are visible`。虽然语义上是好的，但放在正向 prompt 中可能让视频模型反而关注 `text/logos/signage`。

结论：这些内容应该放在 `negative_prompt`，不要放在正向 prompt。

## 5. 后处理安全版

我额外生成了一个后处理安全版：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended_safe/
```

复现命令：

```bash
python scripts/post_filter_llm_prompt_bank.py \
  --input-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended_safe \
  --splits train,val
```

规则：

1. 去掉所有 fallback/reject 样本。
2. 去掉少量公众人物/IP/品牌/艺术家风格残留。
3. 去掉少量暴力、战争、政治、灾难等敏感残留。
4. 从正向 prompt 中删除 `ensure no readable text/logos/signage...` 这类否定句。

后处理结果：

| split | LLM 输出 | safe kept | rejected | 清理正向 no-text 句 |
| --- | ---: | ---: | ---: | ---: |
| train | 2000 | 1953 | 47 | 1 |
| val | 200 | 197 | 3 | 0 |

拒绝原因统计：

| split | fallback/reject | public/IP/brand | sensitive/violent |
| --- | ---: | ---: | ---: |
| train | 34 | 6 | 15 |
| val | 2 | 2 | 1 |

注意：原因可能重叠，所以各原因数量之和会大于 rejected 总数。

## 6. 质量判断

我的判断：

```text
rf_vidprom_prompt_bank_v1_llm_extended:
  可用于分析和人工抽查，不建议直接做 teacher generation。

rf_vidprom_prompt_bank_v1_llm_extended_safe:
  可以作为下一轮小规模 teacher generation 的 prompt bank。
```

如果目标是先跑 100-200 条对比视频质量，我建议使用：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended_safe/train.jsonl
```

如果要扩大到 2k teacher data，建议先人工抽查 safe 版中的 100 条。因为它是自动后处理，不是人工金标。

## 7. 对 gpt-mini 是否够用的结论

这次运行说明：mini 级模型基本够做 prompt 扩写。

它做得好的地方：

- 输出 JSON 稳定，2,200 条没有大面积格式错误。
- 能把短 prompt 扩成 50 词左右的单镜头视频 prompt。
- 能显式加入 5 秒动作、镜头、光照和一致性描述。
- 能识别一部分规则过滤没覆盖到的公众人物/IP/敏感内容。

它不够稳的地方：

- 对长尾 IP/艺术家风格不是 100% 过滤。
- 对战争/灾难/政治敏感度不够保守。
- 偶尔会把“不要文字/logo”写进正向 prompt。
- 对 underage/girl/boy 这类词会保留，需要结合具体内容判断。

推荐最终流程：

```text
VidProM clean-only
  -> gpt-mini 批量 LLM 扩写
  -> 自动后处理过滤
  -> 人工抽查 100 条
  -> 小规模 teacher generation
  -> 再决定是否扩大到 2k/16k
```

## 8. 2026-05-03 追加：数量对齐和最新版编排

这次复查后需要修正一个流程口径：`VidProM clean-only`、`LLM 扩写`、`post-filter` 分开跑本身没有问题，但不能只在 clean-only 阶段对齐目标数量。

当前实际数量已经说明这个风险：

| 阶段 | train | val |
| --- | ---: | ---: |
| clean-only v1 | 2000 | 200 |
| LLM extended safe | 1953 | 197 |

也就是说，如果 teacher generation 需要 2000/200 条，当前 safe 版会短缺 47/3 条。原因是 LLM reject/fallback 和 post-filter 都会继续丢样本。

已新增顶层编排脚本：

```text
scripts/build_vidprom_prompt_pipeline.py
```

新版推荐组织方式：

```text
先 oversample 生成 clean pool
  -> LLM 扩写
  -> post-filter 得到 safe pool
  -> 从 safe pool 中 deterministic selection 到最终 2000/200
  -> reindex prompt_id，保留 source_prompt_id 追溯
```

详细记录见：

```text
docs/01_Prompt构建/VidProM_v2编排记录.md
```

推荐正式输出目录：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/
```

推荐命令：

```bash
python scripts/build_vidprom_prompt_pipeline.py \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v2 \
  --work-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v2_work \
  --train-count 2000 \
  --val-count 200 \
  --oversample-factor 1.25 \
  --motion-rich-ratio 0.20 \
  --workers 8
```

如果最终 safe pool 仍然不足，脚本会直接失败并提示提高 `--oversample-factor` 或 `--min-oversample-extra`，不会静默产出短缺 prompt bank。

本次 v2 已使用 `configs/llm_config.json` 重新生成完成：

```text
clean pool:   train 2600, val 260
LLM output:   train 2600, val 260
safe pool:    train 2522, val 248
final output: train 2000, val 200
```

最终 `rf_vidprom_prompt_bank_v2` 中 fallback 和 unusable 样本均为 0。

## 9. 后续改进

建议下一步改进脚本：

1. LLM reject 时不再 fallback 到原 prompt，而是标记 `usable_for_teacher_generation=false`。
2. 把 public/IP/sensitive 后处理规则内置进扩写脚本。
3. 在 system prompt 里更强地要求：不要在正向 prompt 里写 `no text/no logo`，这些只属于 negative prompt。
4. 对 hard cases 可用更强模型二次重写，而不是丢弃。
5. 输出一个 `accepted.jsonl` 和 `rejected.jsonl`，避免后续误用 fallback 样本。
