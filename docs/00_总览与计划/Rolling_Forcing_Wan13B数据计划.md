# Rolling Forcing 风格流式视频生成数据方案

## 1. 结论先行

如果你的目标是：

1. 基于现有 `Wan2.1-T2V-1.3B` 做流式/实时视频生成的后训练；
2. 尽量规避真实视频版权风险；
3. 先拿到一批足够驱动实验和初版训练的数据；

那么 `Rolling Forcing` 的“数据获取”思路是合适的，而且我认为它是当前最适合你的主路线。

但这里有一个非常关键的边界：

- `Rolling Forcing` 适合的是 **post-training / distillation**，不是从零训练视频基础模型。
- 它真正依赖的主数据不是“海量真实视频”，而是 **prompt 分布 + 现有教师模型生成/采样得到的轨迹**。
- 如果你想“从零训一个新的 Wan 级别基础模型”，这条路不够；如果你想“把现有 Wan1.3B 改造成更适合流式生成的 student”，这条路非常对。

所以我的建议是：

- 主方案采用 `Rolling Forcing / Self Forcing / CausVid` 这一类 **prompt-driven distillation** 路线；
- 真实视频只保留一个 **小而干净、可审计、许可更清晰** 的补充集，用于 sanity check、轻量对齐、评估，而不是作为主训练料仓；
- 在你当前代码库里，先落一个 **“可直接训练的 teacher-generated video,prompt 数据集”**，同时保留升级到更忠实 `Rolling Forcing` 训练脚本的空间。

## 2. 我从论文里读到的关键信息

我阅读的是本地 PDF：

- `code/video_code/data_construct/Liu 等 - 2025 - Rolling Forcing Autoregressive Long Video Diffusion in Real Time.pdf`

和你这个任务直接相关的几点如下。

### 2.1 它并不是“从公开视频站抓训练视频”

论文在方法和实现部分明确表达的是：

- `Rolling Forcing` 以 `Wan2.1-T2V-1.3B` 为基座；
- 训练使用的是 few-step causal distillation；
- 训练目标来自 `DMD` 风格的整体视频分布匹配；
- 训练时条件历史是 **模型自己生成的 history**，不是纯 GT history；
- 论文甚至明确写到，这个方法不需要再做大规模真实视频数据训练。

更具体地说，论文实现细节中给出的信息是：

- 基座模型：`Wan2.1-T2V-1.3B`
- 分辨率：`832x480`
- 长度：`5s @ 16 FPS`
- `T = 5`
- 每个 chunk 含 `3 latent frames`
- 训练 temporal window：`27 latent frames`
- 训练步数：`3000`
- batch size：`8`
- 初始化时使用 `16k ODE solution pairs`
- prompt 来自 **filtered + LLM-extended VidProM**
- 训练策略是 `Self Forcing` 与 `Rolling Forcing` 混合，`50% / 50%`

### 2.2 它真正需要的数据是什么

如果忠实复现 `Rolling Forcing`，主数据其实是三类：

1. `prompt` 集合
2. 教师模型对应的采样/ODE 轨迹
3. student rollout 过程中产生的 self-generated histories

换句话说，它最核心的“数据资产”并不是 `.mp4` 本身，而是：

- prompt 分布
- 每个 prompt 的生成配置
- 教师/学生在训练时需要访问的 latent / noisy / denoised 轨迹

### 2.3 为什么它适合你现在的处境

因为你现在最缺的是：

- 没有高质量、无明显侵权风险的大视频集
- 但你已经有 `Wan2.1`
- 也有 `DiffSynth-Studio`

这正好匹配 `Rolling Forcing` 的优势：

- **把“难拿的真实视频数据”依赖降到最低**
- 主要依赖现有开源基础模型做 teacher
- 更像是“模型后训练的数据构造问题”，而不是“互联网抓视频清洗问题”

## 3. 对你目标的判断

### 3.1 适合的目标

`Rolling Forcing` 方案适合下面这个目标：

- 把 `Wan2.1-T2V-1.3B` 改造成更适合长时流式生成的模型
- 训练重点是减少 error accumulation、提高 streaming consistency、降低 latency

### 3.2 不适合的目标

它不适合下面这个目标：

- 从零开始训练一个通用视频基础模型
- 指望只靠 prompt 和 teacher rollout 获得跟原始 Wan 预训练等价的数据覆盖

如果你的目标其实是“训一个新的 base Wan1.3B”，那仍然需要真实视频大集，这条线就会回到 `OpenVid / VideoUFO / 自采 / 合成数据` 的传统问题。

## 4. 类似方案调研

下面只列跟你当前需求最相关的方案。

| 方案 | 核心思路 | 是否依赖真实视频作为主训练料 | 是否依赖教师视频模型 | 对你是否合适 |
| --- | --- | --- | --- | --- |
| CausVid | 把双向视频扩散模型蒸馏成因果 few-step 生成器 | 否，偏 prompt + teacher trajectory | 是 | 合适，适合作为最早 baseline |
| Self Forcing | 训练时用 self-generated histories 缓解 exposure bias | 否，偏 prompt + rollout | 是 | 合适，是 Rolling Forcing 的直接前身 |
| Rolling Forcing | 多帧 rolling denoising window + attention sink + non-overlap training | 否，主数据仍是 prompt/trajectory | 是 | 最合适，直接对应你的目标 |
| StreamDiT | moving buffer + from-scratch streaming T2V 架构 | 是，需要大规模视频训练 | 是，但路线更重 | 不适合作为第一阶段 |
| Resampling Forcing | teacher-free、从零训练 AR video diffusion | 是，需要大规模真实视频 | 否 | 不适合作为现在的主路线 |

### 4.1 CausVid

参考：

- arXiv: https://arxiv.org/abs/2412.07772
- 项目页： https://causvid.github.io/

要点：

- 从预训练双向视频扩散模型蒸馏成自回归 few-step 模型；
- 使用 `teacher ODE trajectories` 做 student 初始化；
- 依赖 prompt 分布和教师轨迹，而不是重新整理海量真视频；
- 已经证明“短视频 teacher -> 长视频 streaming inference”是可行的。

对你最有价值的地方：

- 它说明 **prompt + teacher trajectories** 作为训练资产是成立的；
- 这也是你现在最容易获得、版权风险最低的训练资产。

### 4.2 Self Forcing

参考：

- arXiv: https://arxiv.org/abs/2506.08009
- 项目页： http://self-forcing.github.io/

要点：

- 核心是让训练阶段也用 `self-generated outputs` 做条件历史；
- 明确针对 exposure bias；
- 训练里使用 KV cache rollout；
- 仍然是 post-training 思路，不要求你再去准备一个新的视频大集。

对你最有价值的地方：

- 如果短期内你还没有把 `Rolling Forcing` 的完整训练 loop 实现出来，`Self Forcing` 是最自然的前置版本。

### 4.3 Rolling Forcing

参考：

- arXiv: https://arxiv.org/abs/2509.25161
- 项目页： https://kunhao-liu.github.io/Rolling_Forcing_Webpage/

要点：

- 不再逐帧严格因果去噪，而是联合去噪一个 rolling window；
- 用 progressively increasing noise 降低误差累积；
- 用 global attention sink 保存最早帧的 KV，增强长期一致性；
- 训练时只在 non-overlapping windows 上回传梯度，降低显存成本；
- 混合 `Self Forcing` 和 `Rolling Forcing` 训练以稳定相机运动。

### 4.4 StreamDiT

参考：

- arXiv: https://arxiv.org/abs/2507.03745
- 项目页： https://cumulo-autumn.github.io/StreamDiT/

要点：

- 它是更“重型”的 streaming video 方案；
- 训练基于 moving buffer；
- 是新架构 + 大规模训练的味道；
- 更适合有大规模真实视频和更长训练预算的团队。

结论：

- 值得参考建模思想；
- 不适合作为你眼下第一阶段的数据方案。

### 4.5 Resampling Forcing

参考：

- arXiv: https://arxiv.org/abs/2512.15702

要点：

- 这是 teacher-free、end-to-end from-scratch 的路线；
- 价值在于“未来如果你真要摆脱 teacher，可以往这里看”；
- 但它重新把问题带回了“大规模真实视频数据”。

结论：

- 不是你现在最优先的选择。

## 5. 数据许可与侵权风险判断

先说结论：

- 如果你的优先级是“尽量少碰侵权问题”，**最安全的主路线不是去找公开视频大集，而是让训练主数据退化为 prompt + teacher-generated trajectories**。
- 真实视频只做小规模、许可可审计补充。

下面是几个常见来源的判断。

### 5.1 VidProM

参考：

- arXiv: https://arxiv.org/abs/2403.06098

要点：

- 这是 prompt 数据集，不是你必须保存视频的那种数据集；
- 论文说明项目在 `CC-BY-NC 4.0` 下发布。

判断：

- 适合做 **非商业实验阶段** 的 prompt bootstrap；
- 如果你未来有商业化诉求，不建议把它当最终主 prompt 库。

建议：

- 可以把它作为“提示词风格参考”和早期实验 prompt source；
- 真正长期可控的方案仍然是 **自写 / 内部生成 / 人工审核 prompt 库**。

### 5.2 OpenVid-1M

参考：

- 论文： https://arxiv.org/abs/2407.02371
- 数据卡： https://huggingface.co/datasets/nkp37/OpenVid-1M

要点：

- 数据卡写的是 `CC-BY-4.0`；
- 但作者同时写明 **intended for research and non-commercial purposes**；
- 且视频样本来自多个公开来源，仍需遵守其上游来源许可证。

判断：

- 这不是“零版权焦虑”的数据源；
- 能做研究实验，但不适合你作为“尽量避免侵权问题”的主答案。

### 5.3 VideoUFO

参考：

- 论文： https://arxiv.org/abs/2503.01739

要点：

- 来自 YouTube 官方 API；
- 论文声称检索的是 Creative Commons 视频；
- 论文写明数据和代码在 `CC BY 4.0` 下发布。

判断：

- 比杂乱网页抓取更规范；
- 但仍然存在两个现实问题：
  - 上传者是否真拥有可授予 CC 的权利；
  - 人像、商标、场馆、品牌等非著作权风险仍然存在。

所以它可以做研究补充，但不应作为你“最稳妥”的主料仓。

### 5.4 Pexels

参考：

- 许可说明： https://help.pexels.com/hc/en-us/articles/360042295174-What-is-the-license-of-the-photos-and-videos-on-Pexels
- 条款说明： https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions

关键信息：

- Pexels 明确写了：**不要用 API 构建数据集或训练 ML/AI 模型，除非得到明确许可**。

判断：

- 不建议作为训练集来源。

### 5.5 Pixabay

参考：

- 许可摘要： https://pixabay.com/service/license-summary/

关键信息：

- 允许免费使用、允许改编；
- 但官方也明确提醒：某些内容还可能涉及额外的版权、商标、隐私、人格权等第三方权利。

判断：

- 风险低于乱抓网路视频；
- 但仍不是“完全放心”的训练主集。

### 5.6 Wikimedia Commons / NASA / Blender Open Movies

参考：

- Wikimedia Commons 复用说明： https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia
- NASA 媒体使用指南： https://www.nasa.gov/nasa-brand-center/images-and-media/
- Blender Open Movies： https://video.blender.org/c/blender_open_movies

判断：

- 这三类更适合做 **小而干净、许可链条更清楚** 的补充视频集；
- 但也要注意：
  - Wikimedia 要逐条看 license 和 attribution；
  - NASA 对 logo、可识别人像、AI attribution 有额外限制；
  - Blender 开放电影更偏动画风格，分布与真实世界视频不同。

我的建议是：

- 它们适合做 `eval / sanity / 小规模对齐`；
- 不适合作为主大集去替代 `Rolling Forcing` 的 prompt-driven 主路线。

## 6. 推荐的总体方案

我建议采用三层结构。

### 6.1 A 轨：主路线，忠实 Rolling Forcing 思路

这是最推荐的主方案。

#### 数据定义

主数据不是“公开视频集合”，而是：

1. `prompt_bank`
2. `teacher_generation_manifest`
3. `optional teacher video cache`
4. `optional latent / ODE cache`

#### 推荐规模

如果目标是对齐论文量级并让 `Wan2.1-T2V-1.3B` 能开始训练：

- `train prompts`：`20k ~ 50k`
- `val prompts`：`1k ~ 2k`
- `ODE init pairs`：先做 `16k`
- teacher video cache：先缓存 `10k ~ 20k` 条就够起步

为什么这个量级够起步：

- 论文训练只有 `3000` steps，batch size `8`；
- 这对应的“每一步有效样本使用量”其实没有大到需要百万真视频；
- prompt 分布的多样性比“硬凑海量视频”更关键。

#### prompt 来源建议

优先级从高到低：

1. 你们自己写/扩写/审核的 prompt
2. 内部 LLM 扩写的 prompt
3. `VidProM` 仅用于非商业实验阶段的 bootstrap

我建议一开始就给 prompt 打标签：

- `subject`: human / animal / vehicle / object / landscape / abstract
- `motion`: static camera / pan / tilt / dolly / follow / chaotic
- `scene`: indoor / outdoor / city / nature / fantasy / sci-fi
- `style`: realistic / cinematic / animation / documentary / macro
- `risk`: logo / celebrity / crowd / child / text-heavy

#### 训练资产建议格式

建议在 `data_construct` 下预留这样的目录：

```text
data_construct/
  docs/
  manifests/
    rf_prompt_bank_train.jsonl
    rf_prompt_bank_val.jsonl
    rf_teacher_manifest.jsonl
  teacher_videos/
    train/
    val/
  teacher_cache/
    ode_pairs/
    rollout_cache/
```

`rf_prompt_bank_train.jsonl` 建议字段：

```json
{"prompt_id":"00000001","prompt":"a dog running through a park, handheld tracking shot, documentary style","lang":"en","subject":"animal","motion":"follow","scene":"outdoor","style":"documentary","source":"internal","license":"internal"}
```

`rf_teacher_manifest.jsonl` 建议字段：

```json
{"prompt_id":"00000001","video_path":"teacher_videos/train/00000001.mp4","seed":1234,"size":"832x480","frame_num":81,"sample_steps":50,"sample_shift":8,"guide_scale":6,"teacher_model":"Wan2.1-T2V-1.3B"}
```

### 6.2 B 轨：立刻能在你当前仓库里跑起来的近似方案

这是最适合“现在就落地”的工程版本。

原因很简单：

- 你当前 `DiffSynth-Studio/examples/wanvideo/model_training/train.py` 直接吃的是 `metadata.csv + video,prompt`
- 它还支持离线 `.pth` cache
- 但它 **并没有现成实现 Rolling Forcing 的完整训练 loop**

所以最现实的第一步不是强行复现论文训练器，而是：

1. 用 `Wan2.1-T2V-1.3B` 生成一批 `5s / 832x480 / 81 帧` teacher videos
2. 写成 `metadata.csv`
3. 先跑 `SFT` 或 `direct_distill` 风格训练
4. 再把训练 loop 升级到 `Self Forcing / Rolling Forcing`

当前仓库里我确认到的相关入口：

- `code/video_code/Wan2.1/generate.py`
- `code/video_code/DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py`
- `code/video_code/DiffSynth-Studio/examples/wanvideo/model_training/train.py`
- `code/video_code/DiffSynth-Studio/data/my_wan_lora_dataset/metadata.csv`

其中最小数据格式就是：

```csv
video,prompt
videos/xxx.mp4,Your prompt here
```

所以如果你要“先得到一批起码足够训练 Wan1.3B 的数据”，最直接的落地定义就是：

- 先做一个 `teacher-generated synthetic dataset`
- 它的主内容是 `video,prompt`
- 每条视频由 `Wan2.1-T2V-1.3B` 生成
- 后续再决定是拿它做普通微调，还是升级成 RF 风格蒸馏训练

#### 我建议的第一版规模

- `train`: `20,000` 条视频
- `val`: `1,000` 条视频
- 每条：`832x480`, `81 frames`

这个规模的意义：

- 足够先把训练、评估、数据清洗、存储和采样流程跑通；
- 不会像百万级那样立刻把算力和磁盘压爆；
- 也足够支持第一轮 `Wan1.3B` streaming post-training 实验。

#### 存储预估

粗略估计：

- 如果只存 `.mp4`，`20k` 条 `5s 480P` 视频大约是 `80GB ~ 250GB` 量级，取决于编码质量；
- 如果额外缓存 `.pth` latent / rollout，中位数可能会再上去数百 GB；
- 所以建议：
  - **默认只存 mp4 + manifest**
  - `.pth` cache 只缓存热点子集，比如 `2k ~ 5k`

### 6.3 C 轨：小规模真实视频补充集

这不是主路线，只是补充。

建议用途：

- sanity check
- 域偏差检查
- 少量真实视频对齐
- benchmark / validation

建议来源：

1. 自采视频
2. Wikimedia Commons 逐条审计视频
3. NASA 非人像/非品牌重点视频
4. Blender Open Movies 及其切片

建议规模：

- `2k ~ 10k` clips 就够

## 7. 具体落地建议

### 7.1 第一阶段：先造 prompt bank

目标：

- 先得到可控、可审计、可扩展的 prompt 主数据

建议数量：

- `20k` train
- `1k` val

建议规则：

- 避免名人、品牌、影视 IP、明显 logo
- 避免大量文本渲染诉求
- 避免高度 NSFW / 暴力 / 医疗 / 未成年人高风险内容
- 统一中英比例，建议先英文为主、中文补充
- 尽量覆盖不同 camera motion 和 scene type

### 7.2 第二阶段：生成 teacher video bank

生成配置建议先统一，减少变量：

- model: `Wan2.1-T2V-1.3B`
- size: `832x480`
- frame_num: `81`
- sample_steps: `50`
- guide_scale: `6`
- sample_shift: `8 ~ 12`

原因：

- 这基本贴近当前 `Wan2.1-1.3B` 推荐设置；
- 也和 `Rolling Forcing` 论文的 5 秒 480P 场景接近。

### 7.3 第三阶段：导出当前仓库可直接训练的数据

目录建议：

```text
data_construct/wan13b_teacher_dataset/
  metadata.csv
  videos/
```

`metadata.csv` 示例：

```csv
video,prompt
videos/00000001.mp4,a dog running through a park, handheld tracking shot, documentary style
videos/00000002.mp4,a small boat drifting on a lake at sunset, slow cinematic pan
```

这套数据可以直接对接当前 `DiffSynth-Studio` 训练入口。

### 7.4 第四阶段：为 Rolling Forcing 预留升级接口

如果后面你们实现更忠实的 RF 训练器，建议再增加：

- `prompt_id -> seed -> teacher ODE pair`
- `prompt_id -> video latent cache`
- `prompt_id -> training rollout cache`

这样就可以从“普通 synthetic video fine-tune”平滑升级到“真正 RF 风格 post-training”。

## 8. 我对“最少够用数据”的建议

如果问题是：

> 现在至少先准备多少，才算“足够给 Wan1.3B 开始训练”？

我的回答是：

### 8.1 最小可开工版本

- `10k` train prompts
- `500` val prompts
- `10k` teacher-generated mp4

用途：

- 跑通数据链路
- 验证 loss 能否下降
- 验证 student 是否有 streaming 改善趋势

### 8.2 比较稳妥的第一版

- `20k` train prompts
- `1k` val prompts
- `20k` teacher-generated mp4
- `2k ~ 5k` latent/trajectory cache

用途：

- 做第一轮正式实验
- 开始比较 `baseline / self-forcing-like / rolling-like`

### 8.3 更接近论文风格的版本

- `30k ~ 50k` prompts
- `16k` ODE init pairs
- `20k+` teacher-generated videos
- 训练过程中在线构造 self-generated histories

用途：

- 更认真地复现 `Rolling Forcing`

## 9. 关键风险

### 9.1 你现在最容易误入的坑

以为“Rolling Forcing 的数据构造 = 去网上找一大堆公开视频”。

这其实不是它的核心。

它的核心是：

- 用已有强 teacher 模型
- 用 prompt 分布驱动
- 用蒸馏和 self-generated history 解决 streaming 问题

### 9.2 真实视频许可并没有绝对零风险

即使是：

- CC
- stock
- 平台声明可复用

也仍可能有：

- 上传者无权授权
- 肖像权 / publicity rights
- 商标 / 品牌 / 场馆限制
- 上游 license 继承问题

所以这里的原则应该是：

- **最小化真实视频依赖**
- **让训练主资产变成 prompt 和 teacher rollout**

### 9.3 当前仓库还没有现成 Rolling Forcing 训练器

你现有仓库能直接做的是：

- `Wan` 推理
- `DiffSynth-Studio` 的常规视频训练 / 蒸馏

但还不能无修改地跑出论文同款 `Rolling Forcing`。

所以文档里的方案分成了：

- 现在能立刻落地的近似版本
- 后面升级成忠实 RF 的版本

## 10. 最终建议

如果只保留一句话：

> 不要先去追求“找一个绝对干净的公开视频大集”，而是先把 `Wan2.1-T2V-1.3B` 当 teacher，用 `prompt bank + teacher-generated clips + optional latent cache` 先做一套 synthetic post-training 数据方案。

我建议你接下来按这个顺序做：

1. 先建 `20k + 1k` 的 prompt bank
2. 用 `Wan2.1-T2V-1.3B` 生成 `20k` 条 `832x480 / 81-frame` teacher videos
3. 导出 `metadata.csv`
4. 先用当前 `DiffSynth-Studio` 跑通一版训练
5. 再实现更忠实的 `Self Forcing / Rolling Forcing` 训练 loop

这条路最符合你现在的资源状态，也最符合“先做起来、先规避版权风险、再逐步逼近论文”的目标。

## 11. 参考链接

- Rolling Forcing: https://arxiv.org/abs/2509.25161
- Self Forcing: https://arxiv.org/abs/2506.08009
- CausVid: https://arxiv.org/abs/2412.07772
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- StreamDiT: https://arxiv.org/abs/2507.03745
- Resampling Forcing: https://arxiv.org/abs/2512.15702
- VidProM: https://arxiv.org/abs/2403.06098
- VideoUFO: https://arxiv.org/abs/2503.01739
- OpenVid-1M: https://arxiv.org/abs/2407.02371
- OpenVid-1M 数据卡: https://huggingface.co/datasets/nkp37/OpenVid-1M
- Pexels 条款说明: https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions
- Pexels 许可说明: https://help.pexels.com/hc/en-us/articles/360042295174-What-is-the-license-of-the-photos-and-videos-on-Pexels
- Pixabay 许可摘要: https://pixabay.com/service/license-summary/
- Wikimedia Commons 复用说明: https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia
- NASA 媒体使用指南: https://www.nasa.gov/nasa-brand-center/images-and-media/
- Blender Open Movies: https://video.blender.org/c/blender_open_movies

## 12. 说明

这份文档的法律判断仅用于工程风险评估，不构成正式法律意见。真正进入对外发布、商用、模型开放下载前，建议再做一轮许可与权利链审计。
