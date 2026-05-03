# Rolling Forcing 数据构造的 Prompt Bank 设计报告

日期：2026-05-03

这份报告说明当前仓库里的 prompt 是如何构成的，分析它是否适合 Rolling Forcing 风格的数据构造，并给出下一版 prompt bank 的改进建议。

## 1. 当前 Prompt 是怎么构成的

当前 prompt bank 由 `scripts/build_prompt_bank.py` 生成，基础配置来自 `configs/rf_prompt_taxonomy_v1.json`。

注意：现在的 prompt 不是直接从 VidProM 抽样来的，而是通过“随机槽位填充”组合出来的。每条 prompt 的生成流程如下：

1. 随机选择一个 `subject_group`。
2. 在该组内部随机选择 `subject`、`action`、`scene` 和 `detail`。
3. 从全局列表中随机选择 `style`、`camera`、`lighting`、`mood`，再随机抽取 3 个 `quality_tags`。
4. 用一个固定英文模板渲染：

```text
A {style} of {subject} that {action} {scene}. The shot uses {camera} with {lighting}, and the overall feeling is {mood}. {detail}. Emphasize {quality_tag_1}, {quality_tag_2}, and {quality_tag_3}.
```

最终写出的 JSONL 记录包含 `prompt_id`、`split`、`lang`、`prompt`、统一的 `negative_prompt`、`source`、`license`、`seed_hint` 和粗粒度 tags。当前 `source` 标记为 `internal_compositional_taxonomy_v1`，因此这批 prompt 是内部合成的、可复现的。

当前 taxonomy 包含：

- 6 种全局风格：documentary、cinematic live-action、travel footage、wildlife footage、lifestyle、stylized animation。
- 8 种镜头设置：handheld follow、side tracking、locked medium、dolly-in、low-angle tracking、wide establishing、shoulder-height moving、forward push-in。
- 8 种光照设置。
- 6 种 mood。
- 6 个 quality tags。
- 7 个 subject groups：家养动物、野生动物、人类活动、地面车辆、水体/天气、城市生活、手工/物体。

当前 `rf_prompt_bank_v1` 里有 20,000 条 train prompts 和 1,000 条 val prompts。由于生成时对 subject group 做均匀随机抽样，所以各 subject group 的数量大致均衡。

## 2. 当前设计的优点

当前设计适合做第一版 proxy 数据集，优点是：

- 确定性强、成本低、完全内部可控。
- 对动作、主体类别、镜头运动、光照、mood 有结构化覆盖。
- 每条 prompt 都记录 tags，后续可以做分层统计和诊断。
- 统一 negative prompt 可以规避一部分低价值输出，例如低质量、模糊、水印、logo、曝光失败、静止画面、重复主体、畸形人体、镜头抖动、杂乱背景等。
- prompt 中显式包含运动描述，比静态图像式 prompt 更适合 text-to-video 生成。

因此，用它来跑 2k smoke/proxy generation 是可以的。但如果目标是严肃复现 Rolling Forcing，它更适合作为 v1 脚手架，而不应该作为最终训练 prompt 分布。

## 3. 当前 Prompt Bank 的主要问题

### 3.1 全局随机槽位导致语义错配

最大问题是全局槽位彼此独立随机采样，导致一些组合虽然能被脚本生成出来，但语义上并不自然，甚至存在明显错配。

当前 bank 中已经出现的例子包括：

- 把 `high-detail wildlife footage` 套到 `coffee beans falling into a grinder` 上。
- 把 `a gardener carrying a watering can` 放在 `in a bakery kitchen` 里。
- `fresh noodles being stretched by hand that show clear step-by-step motion` 这类主谓/数的问题。
- 室内或桌面场景搭配 `cool dusk light` 等偏室外的光照描述。

这会影响 teacher model 的条件输入质量。模型可能仍然能生成看起来还可以的视频，但 text-video pair 的噪声会变大。对 RF 训练来说，这个问题尤其关键，因为下游模型要学的是长时序自回归稳定性，脏 prompt 会把昂贵的 teacher 数据价值打折。

### 3.2 时间结构描述偏弱

大多数 prompt 只描述一个连续动作，而不是一个 5 秒视频应该有的短时序计划。对于 5 秒、16 FPS 的 clip，更理想的 prompt 应该包含：

- 初始状态；
- 中间动作或交互；
- 结束状态或继续运动趋势；
- 镜头和动作之间的关系。

当前 prompt 常见描述是 “moves steadily”、“runs”、“show clear step-by-step motion” 这类单句动作。它们能触发运动，但不能系统性覆盖长程一致性、遮挡后重现、转身、视差、主体身份保持等 RF 关心的难点。

### 3.3 风格分布均衡，但不够接近真实模型使用分布

均匀平衡 style/camera/lighting 方便做覆盖，但真实 text-to-video prompt 并不是这样均匀分布的。Rolling Forcing 使用的是 filtered and LLM-extended VidProM prompt，而 VidProM 本身是大规模真实用户 text-to-video prompts。也就是说，训练 prompt 分布最好保留一部分真实 prompt 的自然性，同时再补充我们关心的可控覆盖。

### 3.4 缺少 RF 专门的难例标签

Rolling Forcing 关注的是自回归长视频生成、减少误差积累、在 streaming 过程中保持全局一致性。当前 prompt schema 没有显式标记这些 RF 难点：

- 遮挡与重现；
- 镜头平移和背景视差；
- 主体身份保持；
- 重复周期动作；
- 物体恒常性；
- 前景/背景交互；
- 多主体协同；
- 场景切换或 prompt-switch 压力测试。

缺少这些 tags，就很难确认数据是否真的覆盖了 RF 的核心 failure modes。

### 3.5 Val Split 不是真正的评测集

当前 val split 和 train split 来自同一个生成分布。这对检查 pipeline 没问题，但不够用于 RF 评估。我们需要固定、分层的 validation suite，包含 easy、medium、hard 时序案例，并且有一批 prompt 在不同实验之间完全保持不变。

## 4. 外部参考和启发

Rolling Forcing 官方仓库在训练部分下载了 Wan2.1-T2V-1.3B 作为 base model，Wan2.1-T2V-14B 作为 teacher model，并从 Self-Forcing release 下载 `vidprom_filtered_extended.txt` 作为训练 prompt。论文也强调长时序 streaming generation、attention sink，以及在 self-generated histories 条件下缓解 exposure bias。

VidProM 是很重要的参考分布，因为它包含 167 万条来自真实用户的 text-to-video prompts，并且包含多个 text-to-video diffusion model 生成的视频。VidProM 作者明确指出 text-to-video prompt 和 image prompt gallery 不同，所以使用 T2V 专门 prompt 来源更合适。

Wan2.1 官方仓库也建议使用 prompt extension 来增强视频细节。它提供 Dashscope/Qwen prompt extension，并说明扩展 prompt 可以丰富生成视频细节、提升质量。这和 Rolling Forcing 使用 filtered and LLM-extended VidProM prompt 的思路一致。

参考来源：

- Rolling Forcing 官方仓库：https://github.com/TencentARC/RollingForcing
- Rolling Forcing arXiv：https://arxiv.org/abs/2509.25161
- Self Forcing arXiv：https://arxiv.org/abs/2506.08009
- VidProM 项目页：https://vidprom.github.io/
- Wan2.1 官方仓库：https://github.com/Wan-Video/Wan2.1

## 5. 市面上的常见做法

调研下来，一个比较准确的结论是：市面上并不是单纯靠一种“规则填充”方案。不同目标对应不同 prompt 构造方式。

### 5.1 大规模训练/蒸馏：更常见是真实数据、真实 prompt、caption 或扩写 prompt

如果目标是训练或蒸馏 video generation model，公开工作更常见的做法是：

- 从真实视频-文本数据集中拿 caption，例如 WebVid、MSR-VTT、内部视频库 caption。
- 从真实用户 prompt gallery 中拿 prompt，例如 VidProM。
- 对短 caption/prompt 做过滤、重写、扩写，让它更像视频生成 prompt。
- 使用 LLM 或 VLM 做质量过滤、语义补全、动作细化。

VidProM 就是典型例子：它强调自己是大规模真实用户 text-to-video prompt 数据集，而不是图像 prompt 数据集的简单替代。Rolling Forcing / Self Forcing 使用 `vidprom_filtered_extended.txt`，本质上也是走“真实 prompt 分布 + 过滤 + LLM 扩展”的路线，而不是随机规则填充。

这类路线的优点是分布更接近真实使用场景，prompt 语气、长度、对象组合、描述习惯更自然。缺点是需要处理 license、脏数据、敏感内容、低质量 prompt、重复 prompt，以及 prompt 和视频内容是否真正对齐的问题。

### 5.2 评测 benchmark：更常见是维度化设计、模板化、LLM 生成和人工校验

如果目标是评测模型能力，规则、模板和人工设计就更常见。因为评测需要可控覆盖，而不是只追求自然分布。

几个代表性工作：

- VBench 把视频生成质量拆成 16 个维度，例如 subject consistency、motion smoothness、temporal flickering、spatial relationship 等。每个维度都有对应 prompt suite 和评测方法。
- FETV 将 prompt 按三个正交维度标注：major content、attribute control、prompt complexity，并区分 spatial/temporal 属性。它的 prompt 来源包括 WebVid、MSR-VTT 和人工构造的 unusual cases。
- T2V-CompBench 分析 VidProM 中 167 万真实用户 prompt，抽取高频 nouns、verbs、adjectives，然后构造 1,400 条 compositional prompts，覆盖 attribute binding、spatial relationship、motion binding、action binding、object interaction、numeracy 等组合能力。
- Movie Gen Bench 使用 1,003 条 prompts，覆盖 human activity、animals、nature/scenery、physics、unusual subjects/activities，并给 prompt 标注 testing concepts 和 motion level。
- SeqBench 专门评估 sequential narrative coherence，用 320 条精心设计的 prompt 覆盖 single/multi subject、single/multi action、strict/flexible/simultaneous temporal order。

这说明：规则填充不是没人用，而是更适合做“能力覆盖”和“评测难例”。但成熟 benchmark 通常不会做无约束随机拼接，而是会先定义能力维度，再设计或生成 prompt，最后做过滤、标注和人工/模型校验。

### 5.3 规则填充到底重要吗？

规则填充本身不坏。它的价值是可控、便宜、可复现、方便覆盖长尾组合。问题在于“无约束随机填充”会把语义噪声引入数据。

对我们的场景，prompt 重要性可以这样判断：

- 对 2k pipeline/proxy generation：重要，但不是阻塞项。当前 v1 足够用于测吞吐、检查 manifest/record/video 格式、验证 loader。
- 对 16k ODE initialization 或 RF training：比较重要。因为每条 teacher trajectory 都很贵，如果 prompt 本身语义冲突，数据成本会被浪费。
- 对最终评测：非常重要。评测 prompt 如果没有固定维度和难度分层，实验结果会很难解释。
- 对 Rolling Forcing 这类长视频/streaming 方法：更重要。RF 关心的是长期一致性、误差积累、KV cache 历史条件、遮挡重现、身份保持。prompt 如果只描述短动作，不会充分激活这些 failure modes。

所以结论不是“不要规则填充”，而是：

1. 不要把无约束随机规则填充当作最终训练分布。
2. 可以把规则系统升级成“有约束的 prompt program”。
3. 更理想的是混合使用真实 prompt 分布、LLM 扩写、规则化 challenge tags、人工/模型过滤。

### 5.4 做 video prompt 数据集时通常要考虑什么

公开工作和实际生成经验里，video prompt 数据集通常要考虑这些方面：

- 来源分布：真实用户 prompt、视频 caption、人工 prompt、LLM 生成 prompt 的比例。
- 文本-视频对齐：prompt 里的主体、动作、属性、场景是否真的能被生成视频表达。
- 时间性：是否包含动作过程、事件顺序、起止状态，而不仅是静态画面描述。
- 运动强度：low / medium / high motion 的覆盖。
- 主体类型：人、动物、物体、车辆、自然场景、城市、人群、抽象/幻想内容。
- 组合能力：多主体、多属性绑定、空间关系、动作绑定、对象交互、数量控制。
- 物理合理性：重力、碰撞、流体、材料形变、遮挡、反射等。
- 长程一致性：身份保持、背景稳定、物体恒常性、遮挡后重现。
- 镜头语言：shot scale、camera motion、viewpoint、景深、运动方向。
- 安全和版权：名人、品牌、影视角色、logo、文字、水印、暴力、色情、隐私。
- 语言和模型适配：英文/中文、多语言 prompt，是否匹配 base model 的训练偏好。
- 可评测性：每条 prompt 是否有 tags，能否按维度聚合结果。
- 质量控制：去重、长度分布、语法、矛盾检测、人工抽检、LLM judge/VLM judge。

### 5.5 对我们项目的直接判断

Rolling Forcing 的 prompt 构造方法确实不是当前这种随机规则填充。它更接近：

```text
VidProM 真实 prompt -> 过滤 -> LLM 扩写 -> 用于 ODE initialization 和 RF training
```

我们当前方法更接近：

```text
内部 taxonomy -> 随机槽位组合 -> 统一 negative prompt -> 生成 proxy teacher videos
```

因此，当前方法适合快速启动，但不适合长期作为最终版本。下一步最好做两条线：

1. 复现线：尽量下载/使用 `vidprom_filtered_extended.txt`，对齐 Rolling Forcing。
2. 增强线：设计 `rf_prompt_taxonomy_v2.json`，用有约束规则和 challenge tags 补 RF 关心的难例。

这两条线可以合并成一个最终 prompt bank：大部分样本来自 VidProM filtered/extended，少部分样本来自我们有意构造的 RF temporal stress prompts。

新增参考来源：

- VBench: https://arxiv.org/abs/2311.17982
- VBench GitHub: https://github.com/Vchitect/VBench
- FETV: https://github.com/llyx97/FETV
- FETV arXiv: https://arxiv.org/abs/2311.01813
- T2V-CompBench: https://t2v-compbench-2025.github.io/
- T2V-CompBench arXiv: https://arxiv.org/abs/2407.14505
- Movie Gen Bench: https://github.com/facebookresearch/MovieGenBench
- SeqBench: https://videobench.github.io/SeqBench.github.io/

## 6. 推荐的 Prompt 策略

下一版最好采用混合方案：一部分来自接近真实分布的 prompt 来源，一部分来自我们主动设计的 RF 难例覆盖。

### 6.1 使用三阶段流程

阶段 A：基础 prompt 来源

- 如果 license 和访问条件允许，优先使用 `vidprom_filtered_extended.txt`，尽量对齐 Rolling Forcing / Self Forcing 的做法。
- 如果暂时不使用外部 prompt，则继续使用内部 taxonomy，但必须加入兼容性约束。
- 额外保留一小批人工编写的 RF stress prompts，用于评估。

阶段 B：LLM 扩展和归一化

- 把短 prompt 扩展成简洁的视频 prompt，包含主体、动作、场景、镜头、光照和时间推进。
- 控制长度，避免堆砌过多电影化形容词。
- 过滤受版权保护的角色、品牌名、可见文字/logo、不安全内容，以及物理上明显不可能的指令。

阶段 C：过滤和分层采样

- 做语法和矛盾检查。
- 统计 subject type、motion type、camera motion、lighting、scene type、RF challenge tag 的覆盖。
- 按目标比例采样，而不是纯粹均匀随机拼槽位。

### 6.2 加入兼容性约束

下一版 taxonomy 应该做局部约束：

- `style` 应该按 subject group 限制。例如 `high-detail wildlife footage` 主要用于动物、野生动物、自然场景，不适合桌面物体。
- `scene` 应该和 subject 兼容。gardener 可以在 greenhouse 或 backyard，不应该随机出现在 bakery kitchen，除非 prompt 明确解释原因。
- `lighting` 应该和 scene 兼容。`neon reflections at night` 更适合雨夜城市、夜市、湿地面，不应随便配雪地松林，除非明确是 stylized。
- `action` 应该按语法和物理属性分类。不可数名词、复数主体、单数主体要用不同模板。
- `camera` 应该和 motion 兼容。locked medium shot 适合手工/桌面动作，tracking shot 更适合车辆、动物、行人。

### 6.3 增加时间难例标签

推荐增加这些 RF challenge categories：

- `basic_motion`：单主体清晰运动，背景简单。
- `camera_parallax`：移动镜头、稳定几何结构和深度视差。
- `cyclic_motion`：走路、跑步、海浪、雨、工具运动等周期动作。
- `occlusion_reentry`：主体短暂被遮挡后重新出现。
- `identity_persistence`：主体转身、改变方向或经过复杂背景时保持外观一致。
- `multi_entity`：两个或多个主体，各自动作可分辨且协调。
- `fine_interaction`：手、工具、小物体、材料形变。
- `environment_dynamics`：雨、雪、海浪、雾、草、反射等环境动态。
- `long_context_anchor`：稳定地标或初始物体在长时序中保持一致。

这些 tags 比单纯的 `style/camera/lighting` 更贴近 RF 的目标。

### 6.4 推荐训练采样比例

对下一批 2k 或 16k 数据，可以考虑：

- 25%：干净背景下的基础主体运动。
- 20%：镜头运动和视差。
- 15%：周期动作和重复动作。
- 15%：手、工具、物体的精细交互。
- 10%：环境动态。
- 10%：遮挡、重现、身份保持。
- 5%：多主体或人群场景。

这个比例既保证数据可学，又能覆盖足够多的困难时序案例。

### 6.5 v2 Prompt 模板建议

内部生成 prompt 的模板可以改成更接近下面这种形式：

```text
{style}. {main_subject} {start_state} in/at {scene}. Over five seconds, {temporal_action}; by the end, {end_state}. The camera uses {camera_motion}, with {lighting}. Keep {consistency_constraint}.
```

例子：

```text
Cinematic live-action video. A border collie starts beside a low wooden hurdle in a grassy park. Over five seconds, it runs forward, jumps over the hurdle, lands, and continues toward the camera. The camera uses a shoulder-height tracking shot with warm late-afternoon light. Keep the dog's coat pattern, body shape, and background trees consistent.
```

这个模板比 v1 随机性更低，但更符合视频监督数据的需求。

## 7. 具体实施计划

1. 创建 `configs/rf_prompt_taxonomy_v2.json`。
2. 为每个 subject group 添加 allowed styles、allowed lighting、allowed cameras 和 typed action templates。
3. 增加 `challenge_tags` 和目标采样权重。
4. 修改 `scripts/build_prompt_bank.py`，或者新增 `scripts/build_prompt_bank_v2.py`，先按 challenge category 采样，再采样兼容的 subject/scene/action。
5. 新增 `scripts/audit_prompt_bank.py`，报告：
   - group/style/camera/lighting/challenge 覆盖；
   - 重复 prompt；
   - 明显语法问题；
   - 已知语义不兼容；
   - prompt 长度分布。
6. 固定一份 `val_rf_temporal_suite.jsonl`，包含 200-500 条 easy/medium/hard RF prompts。
7. 如果要严格复现，优先下载并过滤 Rolling Forcing / Self Forcing 使用的 `vidprom_filtered_extended.txt`；内部 taxonomy 可用于补足缺失的 challenge categories。

## 8. 对当前 2k 生成任务的建议

当前 2k generation 可以继续跑，作为 pipeline/proxy dataset 是有价值的。它可以用于测吞吐、检查 record 格式、验证下游 loader 假设。

但我不建议把它当成最终 RF 复现数据。在投入大量 GPU 时间生成 16k ODE pairs 或更大的 teacher cache 之前，最好先生成 v2 prompt bank，引入兼容性约束和 RF challenge tags。prompt 质量是所有 expensive teacher data 的上游，prompt 做得更好，后面的生成数据才更值钱。
