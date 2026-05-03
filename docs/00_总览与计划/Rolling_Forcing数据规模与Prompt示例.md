# Rolling Forcing 数据量估算与 Prompt 样例

记录日期：2026-05-03

## 1. 对齐的论文配置

目标配置来自 Rolling Forcing 实验描述：

- base model: `Wan2.1-T2V-1.3B`
- video spec: `5s`, `16 FPS`, `832x480`
- ODE initialization: `16k ODE solution pairs`
- prompt source: filtered and LLM-extended VidProM
- chunk denoising:
  - `T = 5`
  - each chunk has `3 latent frames`
  - trained temporal window has `27 latent frames`
- training:
  - `3000` steps
  - batch size `8`
  - generator learning rate `1.5e-6`
  - fake score learning rate `4.0e-7`
  - generator update every `5` fake-score updates

本地数据生成 benchmark 口径：

- `Wan2.1-T2V-1.3B`
- `832x480`
- `81 frames`
- `50 inference steps`
- 单卡单条 `166.139s`
- 8 卡理想并行吞吐约 `173.35 samples/hour`

## 2. 数据量与耗时估算

### 2.1 最小必须离线构造的数据

论文里明确需要离线初始化的是：

- `16k ODE solution pairs`

如果把“一条 5 秒 832x480 base model 采样轨迹”作为一个 ODE pair 构造任务的近似成本，则：

| 数据项 | 数量 | 单卡耗时 | 8 卡耗时 | 说明 |
| --- | ---: | ---: | ---: | --- |
| ODE initialization pairs | `16,000` | `738.4h` / `30.8d` | `92.3h` / `3.85d` | 最小离线 teacher 轨迹数据 |
| Rolling Forcing training prompt draws | `3,000 x 8 = 24,000` | 近似无需预生成 | 近似无需预生成 | 训练时主要是 prompt 采样和 student rollout |
| 如果把训练 prompt 也全部预生成视频 | `24,000` | `1107.6h` / `46.1d` | `138.4h` / `5.77d` | 不是论文必需，只是备选缓存策略 |
| ODE pairs + 全部训练 prompt 预生成 | `40,000` | `1846.0h` / `76.9d` | `230.7h` / `9.61d` | 更重的“全缓存”策略 |

结论：

- 按当前机器和 1.3B 速度，`16k ODE solution pairs` 大约需要 `3.9` 天 8 卡连续生成。
- Rolling Forcing 训练阶段的 `24k` prompt draw 本身不等价于要离线生成 `24k` 条 teacher videos。
- 如果为了调试方便把训练 prompt 对应视频也全部预生成，总耗时会接近 `9.6` 天。

### 2.2 如果误用 14B 做同样数据

RF 配置本身使用 `Wan2.1-T2V-1.3B`，不是 14B。

但如果把同样的 `16k` ODE pair 采样换成 `Wan2.1-T2V-14B`，按本地 `848.428s/sample` 估算：

- 单卡：约 `3770.8h`，即 `157.1d`
- 8 卡：约 `471.3h`，即 `19.6d`

所以 14B 不适合作为这套 RF 初始化数据的大规模生成器，只适合做小规模高质量对照或评估集。

## 3. 存储量估算

### 3.1 只保存 mp4 proxy

当前 smoke 数据里，`832x480, 81 frames` 的 mp4 平均约 `0.39 MB/sample`。

如果只保存可视化 mp4 proxy：

| 数量 | 估算 mp4 存储 |
| ---: | ---: |
| `16,000` | `~6.1 GB` |
| `24,000` | `~9.1 GB` |
| `40,000` | `~15.2 GB` |

这只是视频预览/普通训练数据格式，不等价于 ODE latent cache。

### 3.2 保存 ODE latent pairs

按 Wan latent 粗估：

- latent channels: `16`
- latent frames: `21`
- latent size: `60 x 104`
- dtype: `bf16`
- 单个 latent tensor: `~4.0 MiB`

如果一个 ODE pair 保存两个 latent tensor：

- `16k` pairs: `~125 GiB`

如果每个 pair 额外保存噪声、条件状态或 teacher prediction，按三个 tensor 估：

- `16k` pairs: `~187 GiB`

实际存储会取决于是否保存：

- noisy latent
- denoised latent / velocity / score
- timestep / sigma
- text embedding
- history latent
- teacher rollout 中间状态

建议给 ODE 初始化数据预留至少 `200-300 GiB` 的空间；如果保存更多中间步骤，需要按保存 tensor 数线性放大。

## 4. Prompt 分布应体现的特征

Rolling Forcing / Self Forcing 类训练的 prompt 不应该只是静态画面描述。它更需要覆盖：

- 连续动作：主体要在 5 秒内发生明确运动；
- 可观察的时间推进：开始、持续、转向、靠近、离开、变化；
- 稳定主体：动物、人、车辆、物体应能跨帧保持身份；
- 摄影机运动：tracking、dolly、push-in、locked shot 等；
- 环境一致性：背景、道路、室内结构、光照不要大幅跳变；
- 中长程一致性：适合后续测试 streaming error accumulation；
- 少文字和商标：避免生成字幕、logo、可识别品牌；
- 运动难度分层：简单平移、转向、遮挡、重复动作、多人/多物体交互。

下面 prompt 样例以内部可控 prompt bank 风格为主，不直接依赖外部 VidProM 文本。

## 5. Prompt 样例

1. A cinematic live-action video of a cyclist wearing a windbreaker riding along a quiet city bike lane, gradually turning through an intersection while the camera follows from shoulder height. Use soft morning light, natural colors, coherent body motion, stable road perspective, and no visible text or logos.

2. A realistic lifestyle video of a border collie running across a grassy park, slowing down to jump over a low obstacle and landing cleanly before continuing forward. Use a low-angle tracking shot, warm late-afternoon light, readable fur motion, and consistent background details.

3. A documentary video of a baker in a small kitchen kneading dough, rotating the dough with both hands, then leaning forward to dust flour across the table. Use a locked medium shot, diffused indoor window light, plausible hand motion, and stable kitchen geometry.

4. A naturalistic travel footage video of a compact electric car driving along a coastal road, passing the camera at medium speed as sunlight reflects across the windows. Use a wide establishing shot, golden-hour sunlight, consistent wheel motion, and clean road perspective.

5. A high-detail wildlife footage video of a deer crossing a foggy meadow, pausing briefly to look around, then changing direction and walking toward tall grass. Use a slow side tracking shot, overcast natural light, subtle foliage motion, and a clean readable animal silhouette.

6. A cinematic live-action video of a dancer in casual rehearsal clothes performing a smooth repeating turn, stepping sideways, and continuing the movement without breaking rhythm. Use a locked medium shot, simple studio walls, crisp limb motion, and stable scene composition.

7. A realistic lifestyle video of a gardener carrying a watering can through a greenhouse, stopping beside a row of plants and pouring water in a steady arc. Use a gentle dolly-in shot, diffused indoor light, coherent arm movement, and consistent plant details.

8. A documentary video of a delivery scooter turning a corner on a rainy downtown block, its wheels sending small ripples through shallow puddles. Use a handheld follow shot, neon reflections at night, natural reflections, stable road geometry, and no readable storefront text.

9. A stylized animation video of a small off-road vehicle rolling across uneven ground in open countryside, bouncing lightly while keeping a steady forward path. Use a low-angle tracking shot, bright midday light, clear suspension motion, and coherent background hills.

10. A high-detail wildlife footage video of a crane bird walking beside a shallow stream, lowering its head toward the water and then lifting off for a short glide. Use a smooth forward push-in, cool dusk light, readable wing motion, and stable water reflections.

11. A cinematic live-action video of a painter working on a large canvas in a warm studio, stepping back to inspect the painting and then leaning in to add broad brush strokes. Use a shoulder-height moving shot, warm indoor light, plausible arm motion, and no visible signatures or text.

12. A naturalistic travel footage video of a yellow bus turning a corner on a quiet suburban street and continuing forward under trees. Use a wide establishing shot, soft morning light, consistent wheel rotation, stable building geometry, and natural colors.

13. A realistic lifestyle video of a fluffy orange cat chasing a bouncing toy across a bright living room, briefly losing sight of it behind a chair and then pouncing back into view. Use a handheld follow shot, diffused window light, coherent cat posture, and stable furniture placement.

14. A documentary video of a hiker with a backpack walking along a rocky hillside trail, stepping over stones while the camera slowly tracks beside them. Use overcast natural light, plausible foot placement, consistent terrain detail, and a calm observational mood.

15. A cinematic live-action video of two people carrying a long wooden board through a tool-filled maker space, coordinating their steps as they turn around a workbench. Use a locked wide shot, diffused indoor light, clean body mechanics, and stable object scale.

16. A high-detail object video of coffee beans falling into a grinder, bouncing and settling as a hand adjusts the container and the camera gently pushes closer. Use crisp texture detail, focused motion, consistent shadows, and no labels or brand marks.

## 6. 推荐落地策略

短期最实际的配置：

- 先构造 `16k` ODE initialization pairs；
- 同时维护一个更大的 prompt bank，例如 `24k-50k` 条，用于训练时采样；
- 不急着把所有训练 prompt 都预生成成 mp4；
- 每隔一段训练步数抽固定 prompt 做可视化评估，观察 streaming consistency、主体保持和误差累积。

如果当前 8 卡机器可连续占用：

- `16k` ODE pairs 预计约 `4` 天；
- 加上失败重试、I/O、脚本启动和校验，实际排期建议预留 `4.5-5` 天。

如果只先做一个小规模 RF smoke：

- `1k` ODE pairs：8 卡约 `5.8h`
- `2k` ODE pairs：8 卡约 `11.5h`
- `4k` ODE pairs：8 卡约 `23.1h`

建议先用 `1k-2k` pairs 打通 ODE cache 格式、训练读取和短训练流程，再扩到 `16k`。

## 7. 当前已启动的 2k 生成任务

2026-05-03 已启动一个 `2k` Wan2.1-1.3B teacher proxy 生成任务：

- launcher: `run/launch_rf_2k_teacher_proxy.sh`
- dataset root: `artifacts/datasets/rf_wan13b_2k_teacher_proxy`
- log: `artifacts/logs/rf_wan13b_2k_teacher_proxy.nohup.log`
- pid file: `artifacts/logs/rf_wan13b_2k_teacher_proxy.pid`
- workers: `8`
- per-rank items: `250`
- total target items: `2000`
- output format: `videos/*.mp4`, `records/*.json`, `logs/manifest_rank*.jsonl`, `metadata.csv`

重要说明：

- 这批数据是 `prompt + teacher generated video + sidecar` 的 teacher trajectory proxy。
- 它可以用于先打通 RF 复现周边流程、可视化检查、prompt 分布验证和训练数据读取路径。
- 它还不是论文里严格意义的 `causal attention masking on 16k ODE solution pairs` 的 latent cache。
- 真正用于 ODE initialization 的 Wan latent pair 格式还需要单独实现，不能直接把这批 mp4 数据冒充为最终 ODE pair cache。
