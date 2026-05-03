# RF/Streaming Prompt 评测与采样调研报告

日期：2026-05-03

本文档回答一个核心问题：从很大的 VidProM prompt 池里随机抽样，是否足够科学？结论是：**作为第一版训练/teacher generation bootstrap 可以接受，但作为 RF/streaming 任务的评测 prompt 和后续大规模训练 prompt 仍然粗糙，需要升级为“自然分布 + 分层难例 + 固定评测集”的双轨方案。**

## 1. 结论摘要

当前 `rf_vidprom_prompt_bank_v2` 已经比内部随机 taxonomy 好很多：

- prompt 来源是真实用户分布 VidProM；
- LLM 扩写后具备 5 秒、单镜头、动作、镜头、光照、一致性约束；
- post-filter 后最终数量对齐到 `train=2000, val=200`；
- final 输出中 fallback 和 unusable 均为 0。

但它仍然不应被当成“科学评测集”：

- `val.jsonl` 仍然是同源随机抽样，不是固定能力 suite。
- LLM 生成的 tags 太自由，`subject_type`、`motion_type`、`challenge_tags` 高基数且混入风格/光照词，不利于分层评估。
- RF/streaming 关心的 failure modes，例如 long-horizon drift、identity persistence、background anchor、occlusion reentry、camera parallax、cyclic motion，目前没有被严格 quota 控制。
- 80% natural + 20% motion-rich 能保证基本分布，但不能保证 hard temporal cases 的覆盖。

推荐路线：

```text
训练 prompt bank:
  保留真实 VidProM 主分布，但改成 quota-aware stratified sampling。

固定评测 prompt suite:
  单独构建 val_rf_stream_suite_v1，不再从训练池随机抽。

prompt 质量评估:
  先做 prompt-only audit，再做 teacher/model generation audit。
```

一句话：**v2 可以继续用于下一批 2k teacher data；但在扩大到 16k/20k 之前，最好先落一个固定 RF streaming eval suite 和规范化 tag/schema。**

## 2. 外部调研概览

本次主要参考了公开 benchmark、项目页、论文页和官方仓库。它们共同说明一件事：成熟视频生成评测不会只随机抽 prompt，而是把 prompt 显式拆成能力维度、难度等级和评测指标。

| 工作 | Prompt / 评测设计 | 对我们的启发 |
| --- | --- | --- |
| Rolling Forcing | 任务目标是 streaming long video，重点是减少 error accumulation、保持 long-horizon consistency。论文摘要强调 attention sink 和 self-generated history 条件下的训练。 | 我们的 prompt 不应只测 5 秒生成质量，还要覆盖长程漂移、全局锚点、历史条件误差累积。 |
| Self Forcing | 官方仓库训练阶段下载 `vidprom_filtered_extended.txt`，并说明模型更适合 long/detailed prompts。 | 使用 LLM-extended VidProM 是合理路线；短 prompt 直接用于 streaming 模型不理想。 |
| VidProM | 1.67M unique text-to-video prompts + 6.69M videos，是真实用户 T2V prompt 来源。 | 适合作为主分布，但真实 prompt 必须过滤、扩写、去重、分层。 |
| VBench / VBench++ | 将视频生成质量拆成多个维度，为每个维度设计 prompt suite 和自动评测方法；VBench++ 扩展到 long video、trustworthiness 等。 | 要把“训练 prompt 分布”和“评测 prompt suite”分开。 |
| FETV | 619 prompts，按 major content、attribute control、prompt complexity 三个正交维度标注，并区分 temporal/spatial。 | prompt schema 应该有正交维度，而不是自由文本 tags。 |
| T2V-CompBench | 从 1.67M 真实用户 prompt 中抽取高频 nouns/verbs/adjectives，构造 1400 prompts，覆盖 7 类 compositionality；使用 MLLM、检测、跟踪指标，并做人类相关性验证。 | 可以用真实分布统计 + 控制构造，避免纯随机；prompt 应包含 active verbs。 |
| Movie Gen Bench | 1003 prompts，覆盖 human activity、animals、nature/scenery、physics、unusual subjects/activities，并标注 testing concepts 和 motion level。 | 我们也应给每条评测 prompt 标注 `testing_concepts` 和 `motion_level`。 |
| EvalCrafter | 700 prompts + 17 objective metrics + subjective user opinions。 | prompt suite 可以配自动指标，但小规模人工抽检仍然必要。 |
| PhyGenBench | 160 prompts，覆盖 27 个物理规律，先人工设计物理现象，再用 GPT-4o 增强多样性，最后人工质检。 | 物理/世界模型类 prompt 不应靠随机出现，而应由“规律 -> prompt -> 质检”构造。 |
| Video-Bench | 300 prompts，使用 MLLM 做多维评估并强调与人类判断对齐。 | 可以引入 MLLM judge，但要用少量人工标注校准。 |

## 3. 资料来源摘记

外部来源：

- Rolling Forcing arXiv: https://arxiv.org/abs/2509.25161
- Rolling Forcing project: https://kunhao-liu.github.io/Rolling_Forcing_Webpage/
- Self Forcing GitHub: https://github.com/guandeh17/Self-Forcing
- Self Forcing prompt file: https://huggingface.co/gdhe17/Self-Forcing/blob/main/vidprom_filtered_extended.txt
- VidProM project: https://vidprom.github.io/
- VBench GitHub: https://github.com/Vchitect/VBench
- FETV GitHub: https://github.com/llyx97/FETV
- T2V-CompBench project: https://t2v-compbench-2025.github.io/
- Movie Gen Bench GitHub: https://github.com/facebookresearch/MovieGenBench
- EvalCrafter GitHub: https://github.com/evalcrafter/EvalCrafter
- PhyGenBench GitHub: https://github.com/OpenGVLab/PhyGenBench
- PhyGenBench project: https://phygenbench123.github.io/
- Video-Bench project: https://video-bench.github.io/

本地资料：

- `docs/01_Prompt构建/Prompt池设计报告.md`
- `docs/01_Prompt构建/VidProM_Prompt构建方案.md`
- `docs/01_Prompt构建/VidProM_v2编排记录.md`
- `docs/00_总览与计划/Rolling_Forcing数据规模与Prompt示例.md`
- `artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/pipeline_stats.json`

## 4. “科学测评 prompt”应该测什么

对 RF/streaming 任务，prompt 评测至少要分两层。

### 4.1 Prompt-only 质量评估

这一层不生成视频，只评估 prompt bank 本身是否健康。

建议指标：

| 指标 | 含义 | 推荐做法 |
| --- | --- | --- |
| 安全/IP | 是否含公众人物、品牌、版权角色、敏感暴力等 | 规则 + LLM lineage check + 人工抽检 |
| 视频性 | 是否真的能生成 5 秒单镜头视频，而不是静态图、logo、poster、抽象概念 | LLM judge 输出 `video_actionability_score` |
| 时间结构 | 是否有主体、起始状态、动作过程、结束趋势或连续运动 | 规则 + LLM schema |
| RF 难度 | 是否覆盖遮挡、重现、身份保持、视差、周期运动、多主体、环境动态 | 规范化 challenge tags |
| 多样性 | 是否重复、近重复、同一概念过密 | prompt_hash + embedding cluster + max-per-cluster |
| 覆盖率 | 各类 subject、scene、motion、difficulty 是否达到目标比例 | quota table |
| 可生成性 | 是否过长、多事件、多镜头、互相矛盾 | LLM reject + word count + single-shot check |
| 评测可解释性 | 是否能从 prompt 中明确判断模型该成功什么 | expected_behavior / failure_mode 字段 |

### 4.2 Generation-dependent 评估

这一层生成视频，然后评估模型输出。

短视频通用指标：

- visual / imaging quality；
- aesthetic quality；
- text-video alignment；
- dynamic degree；
- motion smoothness；
- temporal flickering；
- subject consistency；
- background consistency；
- object/action binding；
- spatial relationship；
- physical commonsense。

RF/streaming 专用指标：

- drift slope：随时间窗口推进，质量或 alignment 的下降斜率；
- anchor consistency：初始主体、背景地标、颜色调性是否长期保持；
- identity persistence：人物/动物/车辆在遮挡、转身、镜头运动后是否保持身份；
- background geometry stability：长时间 camera motion 下背景结构是否漂移；
- cyclic motion stability：步态、波浪、雨雪、机器运转是否周期稳定；
- chunk boundary artifacts：AR block 接缝处是否出现跳变、闪烁、语义断裂；
- prompt adherence over time：前中后段是否都还符合 prompt，而不是前 5 秒符合、后面漂掉；
- exposure-bias symptoms：后段是否出现颜色过饱和、主体变形、场景崩坏。

## 5. 当前 v2 的诊断

当前 v2 最终输出：

```text
train: 2000
val:   200
```

source bucket：

| split | natural | motion-rich |
| --- | ---: | ---: |
| train | 1600 | 400 |
| val | 160 | 40 |

词数：

| split | min | max | avg |
| --- | ---: | ---: | ---: |
| train | 38 | 70 | 52.14 |
| val | 42 | 64 | 52.25 |

优点：

- LLM 扩写长度集中，适合 Wan2.1 5 秒 teacher generation。
- final 版没有 fallback/unusable。
- natural/motion-rich 比例准确。
- post-filter 清理了 fallback、IP/敏感 lineage 和 no-text 正向句。

主要问题：

1. `val` 不是科学评测集，只是同分布抽样。
2. tags 不稳定：LLM 输出的 `subject_type` 包含 `human`、`person`、`people`、`single_person` 等大量近义标签。
3. `challenge_tags` 混入光照和风格，例如 `shallow_depth_of_field`、`soft_lighting`、`film_grain`，不都是 RF temporal challenge。
4. `motion_intensity` 出现 `low`、`medium`、`high` 之外的 `moderate`、`gentle`、`subtle` 等自由标签。
5. hard RF cases 不保证足量，例如遮挡重现、身份保持、长程 anchor、camera parallax 只是自然出现，不是 quota 控制。

判断：

```text
rf_vidprom_prompt_bank_v2:
  适合继续做 2k teacher generation / pipeline smoke / 第一轮训练试验。

  不适合单独作为科学评测集。

  如果扩大到 16k/20k，建议先做 tag canonicalization 和 quota-aware sampling。
```

## 6. 推荐的 prompt 体系

建议把 prompt 数据拆成三类，而不是只维护一个随机抽样池。

### 6.1 Train Pool: 自然主分布

用途：训练/蒸馏时提供接近真实用户分布的多样条件。

来源：

- VidProM filtered + LLM extended；
- 适量 internal stress prompts；
- 未来可加入自建/人工审核 prompt。

采样策略：

```text
60-70% real-user natural distribution
20-30% challenge-enriched VidProM
10-15% internal RF stress/programmatic prompts
```

这里的 challenge-enriched 不是随机 motion-rich，而是按 RF 维度分层：

- camera parallax；
- occlusion reentry；
- identity persistence；
- cyclic motion；
- fine interaction；
- environment dynamics；
- multi-entity；
- physics / material interaction；
- stable anchor / long context。

训练集和评测集的原则不同：训练集不应该做成小而固定的能力考试，而应该是一个足够大的、去重的、质量可控的“主分布 + 难例增强”集合。固定评测集负责稳定比较模型；训练集负责尽量覆盖真实用户分布和 RF/streaming 的常见失败模式。

推荐把训练数据分成两层：

```text
prompt train bank:
  只保存 prompt、规范化标签、质量分、来源和采样权重。

video train manifest:
  在 prompt 生成 teacher video 并通过视频质检后，记录真正进入训练的数据样本。
```

也就是说，`rf_vidprom_prompt_bank_v2/train.jsonl` 还只是 teacher generation 的 prompt source，不等价于最终模型训练 manifest。最终训练集应该在生成视频后再做一次筛选，过滤掉 teacher 模型生成崩坏、明显不对齐、闪烁严重、主体漂移严重的样本。

建议的训练样本字段：

```json
{
  "sample_id": "rftrain_v3_00000001",
  "prompt_id": "rfvp3_train_00000001",
  "prompt": "...",
  "canonical_tags": {
    "subject_type": "human",
    "scene_type": "city",
    "motion_type": "camera_motion",
    "rf_challenge_tags": ["camera_parallax", "identity_persistence"]
  },
  "prompt_quality": {
    "overall": 84,
    "temporal_specificity": 86,
    "rf_challenge_value": 78
  },
  "teacher_video": {
    "path": "...",
    "teacher_model": "...",
    "seed": 12345,
    "num_frames": 81,
    "fps": 16,
    "resolution": "480p"
  },
  "video_quality": {
    "usable": true,
    "alignment_score": 0.82,
    "motion_score": 0.76,
    "temporal_consistency_score": 0.80,
    "reject_reason": ""
  },
  "sample_weight": 1.0,
  "split": "train"
}
```

对 Rolling Forcing / streaming 训练，还建议把训练数据再拆成两种用途：

| 类型 | 作用 | 建议 |
| --- | --- | --- |
| short teacher clips | 学基础 text-video alignment、动作、主体一致性 | 覆盖主分布，单 prompt 通常 1 个 seed，优先扩大 prompt 多样性 |
| streaming continuation units | 学历史条件、长程 anchor、chunk 边界稳定 | 从 hard RF prompt 子集生成多 chunk 或长视频，记录 chunk index、history length、conditioning frames |

短期如果预算有限，优先做“更多不同 prompt 的 1-seed teacher clips”，不要把大量预算花在同一 prompt 多 seed 上。多 seed 更适合小规模稳定性分析或 eval，不适合作为第一版训练集扩容的主策略。

训练集和评测集要做严格去泄漏：

- `val_rf_stream_suite_v1`、`val_rf_stream_canary_v1` 的 prompt 不能进入 train；
- 与 eval prompt 同 `source_uuid`、同 normalized prompt hash、或 embedding cluster 过近的样本也应排除；
- internal stress prompt 如果有 train/eval 两版，要使用不同模板、不同主体、不同场景锚点。

推荐的 v3 训练 prompt bank 目标：

```text
rf_vidprom_prompt_bank_v3:
  train: 16k-20k prompts
  dev_random: 500-1000 prompts，仅用于训练过程 sanity，不作为主评测
  fixed eval: 单独使用 val_rf_stream_suite_v1，不从 train 随机切
```

v3 采样配比可以先用一个保守版本：

```text
70% VidProM natural safe prompts
20% VidProM hard RF / challenge-enriched prompts
10% internal RF stress prompts
```

等验证 hard prompt 对训练没有明显副作用后，再提高到：

```text
60-65% natural
25-30% challenge-enriched
10-15% internal stress
```

训练时可以给样本加 `sample_weight`，而不是只靠数量硬塞难例：

| 样本类型 | 初始 sample_weight |
| --- | ---: |
| natural safe | 1.0 |
| challenge-enriched | 1.1-1.3 |
| internal stress | 0.8-1.2 |
| 视频质检边缘但可用 | 0.5-0.8 |

这样能避免 hard case 过多导致训练分布偏离真实用户 prompt，同时又能让 RF 关键能力被模型看到足够多次。

### 6.2 Val Suite: 固定能力评测集

用途：不同 checkpoint、不同训练策略、不同采样策略之间稳定比较。

建议规模：

```text
val_rf_stream_suite_v1: 240-360 prompts
```

建议结构：

| 维度 | 数量建议 | 示例能力 |
| --- | ---: | --- |
| easy basic motion | 30 | 单主体平移、简单背景 |
| camera parallax | 30 | tracking、dolly、pan，背景几何稳定 |
| cyclic motion | 30 | 走路、跑步、波浪、雨雪、机器周期 |
| identity persistence | 30 | 转身、遮挡后重现、复杂背景中保持外观 |
| occlusion/reentry | 30 | 人/车/动物被遮挡后再次出现 |
| multi-entity | 30 | 双主体或群体动作，身份和空间关系不混淆 |
| fine interaction | 30 | 手、工具、小物体、材料变化 |
| environment dynamics | 30 | 水、火、雾、雪、反射、树叶 |
| physics/material | 30 | 碰撞、重力、液体、弹性、破碎 |
| long-context anchor | 30 | 背景地标、房间结构、颜色调性长期稳定 |
| prompt switch / interactive | 20-40 | streaming 中途改变动作或局部语义 |

每条 prompt 需要字段：

```json
{
  "prompt_id": "rfse_val_000001",
  "prompt": "...",
  "expected_behavior": "...",
  "primary_axis": "identity_persistence",
  "secondary_axes": ["occlusion_reentry", "camera_parallax"],
  "difficulty": "medium",
  "motion_intensity": "medium",
  "subject_type": "human",
  "scene_type": "city",
  "entity_count": 1,
  "camera_motion": "tracking",
  "expected_failure_modes": ["identity_drift", "background_warp"],
  "source": "internal_rf_eval_v1",
  "license": "internal"
}
```

### 6.3 Canary Suite: 每次都跑的小集

用途：快速检查训练是否倒退。

建议规模：

```text
20-40 prompts
```

特点：

- 每次实验固定；
- seed 固定；
- 覆盖最容易退化的几类：subject consistency、dynamic degree、flicker、chunk boundary、background anchor。

## 7. 推荐的 tag/schema 规范化

当前 LLM tags 太自由。建议引入 canonical schema，把自由标签映射到有限集合。

### 7.1 subject_type

```text
human
animal
vehicle
object
environment
scene
multi_entity
fantasy_creature
abstract
```

### 7.2 scene_type

```text
indoor
city
nature
water
sky
studio
industrial
fantasy
space
unknown
```

### 7.3 motion_type

```text
static_low_motion
subject_translation
camera_motion
cyclic_motion
fine_interaction
environment_dynamics
multi_entity_motion
transformation
physics_interaction
```

### 7.4 rf_challenge_tags

```text
basic_motion
camera_parallax
cyclic_motion
occlusion_reentry
identity_persistence
multi_entity
fine_interaction
environment_dynamics
physics_material
long_context_anchor
chunk_boundary_sensitive
prompt_switch
```

注意：光照、风格、画质不应该混入 `rf_challenge_tags`，应拆到：

```text
style_tags
lighting_tags
visual_quality_tags
```

## 8. Prompt Quality Score 建议

可以给每条 prompt 一个 `prompt_quality` 对象，用于过滤和排序。

建议分数：

```json
{
  "prompt_quality": {
    "overall": 86,
    "safety": 100,
    "single_shot_feasibility": 90,
    "temporal_specificity": 85,
    "rf_challenge_value": 75,
    "visual_clarity": 85,
    "generation_feasibility": 90,
    "diversity_value": 80,
    "reject_reason": ""
  }
}
```

权重建议：

| 子项 | 权重 | 说明 |
| --- | ---: | --- |
| safety | hard gate | 不安全直接拒绝 |
| single_shot_feasibility | 15 | 是否适合 5 秒单镜头 |
| temporal_specificity | 20 | 是否有动作过程和持续性 |
| rf_challenge_value | 25 | 是否覆盖 RF 关键难点 |
| visual_clarity | 10 | 主体、场景、镜头是否清晰 |
| generation_feasibility | 15 | 是否过难、多事件、矛盾 |
| diversity_value | 15 | 是否补足当前池子的稀缺类别 |

分数用途：

- `overall < 60`：拒绝或重写；
- `60-75`：可入 train，不入 eval；
- `75-90`：可入 train / candidate eval；
- `>90`：优先人工抽查，可入 fixed eval suite。

## 9. Sampling 策略建议

### 9.1 不建议

```text
从 1.67M VidProM 过滤后直接 reservoir random sample
```

这个策略的问题是：真实分布会压低稀有 hard temporal cases 的占比，尤其是 RF 最关心的长程一致性难例。

### 9.2 建议

使用两阶段采样：

```text
Stage A: 按自然分布形成大候选池
Stage B: 按 canonical axes 和 prompt_quality 做 quota-aware selection
```

训练集可以保留自然分布，但要给困难维度设置最低配额：

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

注意：这些标签可以重叠，所以比例总和可以超过 100%。

### 9.3 多样性控制

建议加入：

- prompt_hash 去重；
- normalized prompt 去重；
- embedding cluster 去重；
- 每个 cluster 最多取 N 条；
- 每个 source_uuid 只取 1 条；
- 每个 named entity / scene template 限流。

## 10. 评测协议建议

### 10.1 Prompt-only audit

每次生成 prompt bank 后先跑：

```text
coverage report
canonical tag distribution
prompt_quality distribution
duplicate / near-duplicate report
safety/IP lineage report
length distribution
expected failure mode coverage
```

### 10.2 Short video eval

对 `val_rf_stream_suite_v1` 生成 5 秒视频：

- 每条 prompt 1-3 个 seed；
- 固定模型参数；
- 记录 sidecar；
- 用 VBench/EvalCrafter 可用维度做自动评价；
- 抽样 50-100 条人工看 alignment、motion、flicker、subject consistency。

### 10.3 Streaming eval

对 `rf_stream_long_suite_v1` 生成更长视频：

```text
30s smoke
60s standard
3-5min stress
```

建议指标：

- 前/中/后窗口 text-video alignment；
- 窗口间 visual quality drift；
- subject embedding drift；
- background/anchor embedding drift；
- dynamic degree drift；
- temporal flicker；
- chunk boundary discontinuity；
- human pairwise preference。

### 10.4 实验报告

每次模型实验记录：

```json
{
  "prompt_suite": "val_rf_stream_suite_v1",
  "suite_hash": "...",
  "model_ckpt": "...",
  "seed_policy": "...",
  "generation_config": "...",
  "metrics": {
    "short_video": {},
    "streaming": {},
    "human_audit": {}
  }
}
```

## 11. 对现有流程是否需要优化

需要，但分优先级。

### 11.1 不需要马上推翻 v2

`rf_vidprom_prompt_bank_v2` 已经解决了最关键的数量、fallback、安全后处理问题。它可以继续作为下一批 2k teacher generation 的 prompt source。

### 11.2 需要马上补一个固定评测集

当前最该补的是：

```text
artifacts/prompt_banks/val_rf_stream_suite_v1/val.jsonl
```

这是评估 RF/streaming 方法是否进步的锚点。否则每次都从训练分布随机抽 val，实验波动会很难解释。

### 11.3 下一轮 pipeline 需要优化 tag 和采样

建议新增：

```text
configs/rf_prompt_eval_axes_v1.json
scripts/audit_prompt_bank.py
scripts/build_rf_stream_eval_suite.py
scripts/canonicalize_prompt_tags.py
```

并把 pipeline 的 final selection 从：

```text
source_bucket natural/motion-rich 20% quota
```

升级为：

```text
canonical RF axes + prompt_quality + diversity cluster quota
```

## 12. 推荐落地顺序

### Phase 1: 低成本补齐评测能力

1. 写 `configs/rf_prompt_eval_axes_v1.json`。
2. 写 `scripts/canonicalize_prompt_tags.py`，把 v2 的自由 tags 归一化。
3. 写 `scripts/audit_prompt_bank.py`，输出 markdown/json 覆盖报告。
4. 从 v2 safe pool + 少量 internal stress prompts 构建 `val_rf_stream_suite_v1`。
5. 人工抽查 `val_rf_stream_suite_v1` 全量或至少 100 条。

### Phase 2: 让训练采样更科学

1. 给 prompt bank 加 `prompt_quality`。
2. 使用 quota-aware sampling 重建 `rf_vidprom_prompt_bank_v3`。
3. 维持自然分布主体，但提高 hard temporal case 的最低覆盖。
4. 对比 v2/v3 生成视频质量和训练效果。

### Phase 3: 接入自动视频评测

1. 先接 VBench 的可用自定义维度，例如 subject/background consistency、motion smoothness、dynamic degree、aesthetic/imaging quality。
2. 再接 EvalCrafter 或自写 RF streaming drift metrics。
3. 最后为 long video 建立 `drift slope` 和 `chunk boundary` 指标。

## 13. 建议的最终目录结构

```text
configs/
  rf_prompt_eval_axes_v1.json
  rf_prompt_sampling_policy_v1.json

scripts/
  canonicalize_prompt_tags.py
  audit_prompt_bank.py
  build_rf_stream_eval_suite.py
  evaluate_streaming_prompt_suite.py

artifacts/prompt_banks/
  rf_vidprom_prompt_bank_v2/
  rf_vidprom_prompt_bank_v3/
  val_rf_stream_suite_v1/
  val_rf_stream_canary_v1/

docs/
  文档索引.md
  01_Prompt构建/
    VidProM_v2编排记录.md
  02_评测与采样/
    RF_Streaming_Prompt评测与采样调研报告.md
    RF_Streaming评测集v1构建记录.md
  03_Teacher视频生成/
    VidProM_v2_Teacher视频队列生成说明.md
```

## 14. 最终建议

短期：

```text
继续用 rf_vidprom_prompt_bank_v2 做下一批 2k teacher generation。
```

中期：

```text
不要再把随机 val 当作主要评测。
先构建 val_rf_stream_suite_v1，并用它固定比较不同 checkpoint。
```

长期：

```text
把 prompt pipeline 升级成 v3：
VidProM natural distribution
  + canonical RF tags
  + prompt quality score
  + quota-aware hard-case enrichment
  + fixed eval suite
```

这比“从超大 prompt 池随机采样”科学得多，也更贴近 Rolling Forcing / Self Forcing 这类 streaming/autoregressive 任务真正关心的能力。
