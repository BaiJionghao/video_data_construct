# VidProM Prompt Bank 的 LLM 扩写方案

日期：2026-05-03

本文档说明为什么需要 LLM 扩写、需要多强的模型、已经落实了什么脚本、怎么调用，以及拿到 `api_key`、`base_url`、`model` 后如何生成 LLM-extended prompt bank。

## 1. 需要多强的模型

我的判断：`gpt-mini` 这一级别通常够用，但要配合抽检和规则兜底。

原因是这项任务不是复杂推理，而是结构化文本改写：

- 把真实用户 prompt 改写成 5 秒、单镜头、text-to-video 友好的英文 prompt；
- 补齐主体、场景、动作、镜头、光照、一致性约束；
- 过滤明显不适合的视频生成输入；
- 输出固定 JSON schema。

这类任务 mini 模型一般可以胜任。真正需要更强模型的地方是：

- 识别长尾名人、品牌、影视/IP 角色；
- 判断 prompt 是否过于抽象、是否能转成单镜头视频；
- 处理很乱的用户 prompt；
- 更稳定地打 `subject_type`、`motion_type`、`challenge_tags`；
- 降低幻觉和过度改写。

推荐策略：

- 2k 级别实验：用 mini 模型即可，跑完后人工抽检 100 条。
- 16k ODE/teacher 数据：mini 也可以，但建议先抽检，再加 audit；预算允许时可用更强模型重写 hard cases。
- 最终论文/正式复现：建议 `mini 模型批量扩写 + 强模型抽样审查/重写问题样本`。

温度建议低一些：

```text
temperature = 0.2
```

这样输出更稳定，便于复现。

## 2. 已新增脚本

新增脚本：

```text
scripts/extend_vidprom_prompt_bank_with_llm.py
```

功能：

1. 读取已有 prompt bank，例如：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1/train.jsonl
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1/val.jsonl
```

2. 调用 OpenAI-compatible `/v1/chat/completions` 接口。
3. 要求 LLM 输出 JSON：

```json
{
  "prompt": "...",
  "reject": false,
  "reject_reason": "",
  "tags": {
    "subject_type": "...",
    "scene_type": "...",
    "motion_type": "...",
    "motion_intensity": "...",
    "challenge_tags": ["..."]
  }
}
```

4. 将 `prompt` 替换成 LLM 扩写后的 prompt。
5. 保留原始清洗 prompt 到 `pre_llm_prompt`。
6. 写出新的 JSONL 和 `llm_extension_stats.json`。
7. 支持 `--dry-run`，不调用模型也能测试链路。
8. 支持 `--resume`，中断后可继续。

脚本没有依赖 OpenAI SDK，只使用 Python 标准库 `urllib`。只要服务兼容 OpenAI chat completions 接口，就可以用。

## 3. Dry-run 验证

已经跑过 3 条样本 dry-run：

```bash
cd /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct

python scripts/extend_vidprom_prompt_bank_with_llm.py \
  --input-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1 \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_dryrun \
  --splits train,val \
  --limit 3 \
  --dry-run
```

输出位置：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_dryrun/
```

验证结果：

```text
train: 输入 3 条，写出 3 条
val:   输入 3 条，写出 3 条
```

注意：dry-run 只是模板扩写，不具备安全/版权/语义判断能力。它只用于验证输入输出流程。正式版本必须调用 LLM。

## 4. 正式调用方式

拿到 `api_key`、`base_url`、`model` 后，当前推荐写入本地 config 文件：

```text
configs/llm_config.json
```

这样 `api_key`、`base_url`、`model` 都不需要写进 shell 命令。仍然保留环境变量和显式参数作为兼容方式，但不作为当前推荐。

环境变量兼容方式：

```bash
export LLM_API_KEY='你的 key'
export LLM_BASE_URL='你的 base_url'
export LLM_MODEL='你的模型名'
```

先小规模测试 20 条：

```bash
cd /vepfs-mlp2/c20250518/241506050/code/video_code/data_construct

python scripts/extend_vidprom_prompt_bank_with_llm.py \
  --input-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1 \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended_test20 \
  --llm-config configs/llm_config.json \
  --splits train,val \
  --limit 20 \
  --temperature 0.2
```

抽查无明显问题后，跑完整 2k + 200：

```bash
python scripts/extend_vidprom_prompt_bank_with_llm.py \
  --input-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1 \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended \
  --llm-config configs/llm_config.json \
  --splits train,val \
  --temperature 0.2 \
  --sleep 0.05 \
  --resume
```

如果接口不是 `/v1/chat/completions`，但兼容 OpenAI 的 base url，只要传 `--base-url` 即可。脚本会自动补 `/v1/chat/completions`。

也可以直接显式传参：

```bash
python scripts/extend_vidprom_prompt_bank_with_llm.py \
  --input-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1 \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended \
  --base-url '你的 base_url' \
  --api-key '你的 key' \
  --model '你的模型名' \
  --splits train,val \
  --temperature 0.2 \
  --resume
```

但不推荐把 key 直接写在命令里。

### 4.1 使用 config 文件

现在脚本支持直接读取：

```text
configs/llm_config.json
```

推荐字段：

```json
{
  "LLM_BASE_MODEL": "gpt-5-mini",
  "LLM_API_URL": "http://host:port/v1",
  "LLM_API_KEY": "..."
}
```

也兼容小写字段：

```json
{
  "model": "gpt-5-mini",
  "base_url": "http://host:port/v1",
  "api_key": "..."
}
```

调用方式：

```bash
python scripts/extend_vidprom_prompt_bank_with_llm.py \
  --input-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1 \
  --output-dir artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended \
  --llm-config configs/llm_config.json \
  --splits train,val \
  --temperature 0.2 \
  --workers 16 \
  --resume
```

顶层 pipeline 同样支持 `--llm-config configs/llm_config.json`。推荐正式生成时走 pipeline，而不是单独跑扩写脚本。

## 5. LLM 扩写要求

脚本里的 system prompt 要求模型：

- 只返回合法 JSON；
- 保留原 prompt 的安全、可用意图；
- 改写成一个 5 秒、单镜头视频 prompt；
- 包含主体、场景、时间动作、镜头、光照、一致性约束；
- prompt 长度控制在 35-85 个英文词；
- 避免文字、字幕、logo、水印、UI、品牌、公众人物、版权角色、色情、血腥暴力、仇恨内容；
- 如果不能安全改写，则 `reject=true`。

输出的 prompt 更适合 Wan2.1 teacher generation，例如：

```text
Realistic video. A small airplane enters from the left side of a clear sky and flies steadily across the frame. Over five seconds, it moves toward the distant horizon while thin clouds drift slowly behind it. The camera uses a locked wide shot with soft daylight. Keep the airplane shape and cloud positions coherent.
```

## 6. 输出字段变化

LLM 扩写后的 JSONL 会保留原字段，并新增：

```json
{
  "pre_llm_prompt": "原 clean prompt",
  "extended_prompt": "LLM 扩写后的 prompt",
  "llm_extension": {
    "model": "...",
    "base_url": "...",
    "temperature": 0.2,
    "reject": false,
    "reject_reason": "",
    "used_fallback_prompt": false
  }
}
```

同时：

- `prompt` 字段会替换成扩写后的 prompt，方便直接给现有 teacher generation 脚本使用。
- `clean_prompt` 保留清洗后的原始 VidProM prompt。
- `raw_prompt` 仍保留 VidProM 原始 prompt。

如果 LLM reject 或输出不合格：

- `prompt` 回退为原 clean prompt；
- `llm_extension.used_fallback_prompt = true`；
- 统计写入 `llm_extension_stats.json`。

## 7. 质量检查建议

正式跑完后，建议检查：

```bash
python - <<'PY'
import json, random
path = 'artifacts/prompt_banks/rf_vidprom_prompt_bank_v1_llm_extended/train.jsonl'
rows = [json.loads(line) for line in open(path, encoding='utf-8')]
for row in random.Random(0).sample(rows, 20):
    print(row['prompt_id'])
    print('RAW:', row['pre_llm_prompt'])
    print('EXT:', row['prompt'])
    print('LLM:', row['llm_extension'])
    print()
PY
```

重点看：

- 是否仍有公众人物、品牌、IP 角色；
- 是否出现文字/logo/UI 生成要求；
- 是否过度改写原意；
- 是否有明确 5 秒动作；
- 是否为单镜头，而不是多段故事；
- 是否适合 Wan2.1 832x480、5s、16 FPS 的 teacher generation。

## 8. 下一步建议

等你提供 key、base_url、model 后，建议执行顺序：

1. 先跑 `--limit 20`。
2. 抽查 20 条。
3. 如果质量 OK，跑完整 `train,val`。
4. 抽查 100 条 train + 50 条 val。
5. 再决定是否用扩写版 prompt bank 跑新的 2k teacher generation。

我的建议模型选择：

```text
优先：便宜稳定的 mini 级模型
参数：temperature 0.2
兜底：对抽查失败类型补规则或用更强模型重写 hard cases
```

## 9. 2026-05-03 追加：最终数量建议使用 pipeline

单独运行本脚本只保证 LLM 阶段输入多少、写出多少；它不会保证后续 safe prompt bank 仍然满足最终 train/val 目标数量。当前 v1 已出现：

```text
clean-only: train 2000, val 200
safe:       train 1953, val 197
```

因此，正式构建 teacher generation 用的 prompt bank 时，建议改用顶层编排入口：

```text
scripts/build_vidprom_prompt_pipeline.py
```

该入口会先 oversample，再运行 LLM 扩写和 post-filter，最后从 safe pool 中精确抽回目标数量并重新编号。详细说明见：

```text
docs/01_Prompt构建/VidProM_v2编排记录.md
```

本次 v2 已使用 `configs/llm_config.json` 和 `--workers 16` 完整生成：

```text
artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/
train.jsonl: 2000
val.jsonl:   200
```
