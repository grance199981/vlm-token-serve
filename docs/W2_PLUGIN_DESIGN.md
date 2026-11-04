# W2 插件设计：vLLM 树外视觉 token 压缩（2026-10-20）

> 依据：vLLM v0.32.0 源码（本地源码副本，以及 2×L40 48GB 服务器上安装的版本，两者行号一致），RATE PR #60963 的 diff（head `06158cbe2748`，仍是 open），DivPrune 官方实现（`vbdi/divprune`，`LLaVA/llava/model/llava_arch.py` 中的 `DivPrune`）。

## 1. vLLM 0.32 已有的"多模态裁剪"机制（插件复用它）

| 环节 | 位置 | 做什么 |
|---|---|---|
| 占位 token 数 | `qwen3_vl.py:1655` `_get_prompt_updates` 里的 `get_image_replacement_qwen3vl` | 预处理阶段决定一张图在 prompt 里占几个 `<image_pad>` |
| 附加原始位置 | `qwen3_vl.py:2499` `_postprocess_image_embeds_evs` | 打开裁剪后，每个视觉 token 的 embedding 后面会多拼 5 个通道：原始 M-RoPE 的 t、h、w、llm_grid_w，以及一个占位通道 |
| 是否启用 | `config/multimodal.py:751` `is_multimodal_pruning_enabled()`；V2 runner 的判断在 `v1/worker/gpu/model_states/mm_pruning.py:137` | 0.32 只认 `--video-pruning-rate` |
| 位置重算 | `qwen3_vl.py:2963` `recompute_mrope_positions` → `multimodal/video_prune/evs.py:154` | 按每个保留 token 自带的原始位置重写 M-RoPE，兼容 chunked prefill |
| 初始占位位置 | `qwen3_vl.py:2888` `_get_mrope_input_positions` | 0.32 默认每张图的 token 数等于完整网格；裁剪后 token 变少，需要改成截断网格 |
| DeepStack | `qwen3_vl.py:885–904` | 多层视觉特征按 token 拼在同一行里，所以按行裁剪时这些特征会自动一起裁掉 |

## 2. 插件需要重写的地方

| # | 重写什么 | 怎么做 | 参照 |
|---|---|---|---|
| 1 | 打开裁剪机制 | 插件加载时给 `MultiModalConfig.is_multimodal_pruning_enabled` 打补丁：只要 `VTS_METHOD` 不是 none，就返回 True | RATE PR 对 config 的改动 |
| 2 | 占位 token 数 | 继承处理器，重写 `_get_prompt_updates` 里的图像替换函数，返回 `keep_count(N, h, w)` | RATE PR |
| 3 | 选 token | 继承模型，重写 `_postprocess_image_embeds_evs`：先调父类得到拼好位置通道的 embedding，再**只用特征部分**（不含位置通道）计算选择，按行保留 | RATE PR（它把位置通道也算进了相似度，我们不这样做） |
| 4 | 初始占位位置 | 重写 `_get_mrope_input_positions`：图像 token 数少于完整网格时，用截断网格占位 | RATE PR |
| 5 | 位置重算 | 重写 `recompute_mrope_positions`，调用插件里带修复的副本：找不到 vision_start 时，退回到用第一个媒体 token 定位 | RATE PR 对 evs.py 的修复 |
| 6 | 注册 | 通过 `vllm.general_plugins` 入口，用 `ModelRegistry.register_model` 把 `Qwen3VLForConditionalGeneration`、`Qwen3_5ForConditionalGeneration` 换成插件里的子类；子类重新用 `MULTIMODAL_REGISTRY.register_processor` 绑定插件的处理器 | vLLM 插件文档 |

## 3. 方法

| 方法 | `VTS_METHOD` | 保留数 | 选择规则 | M-RoPE 位置 |
|---|---|---|---|---|
| 不压缩，但走裁剪路径 | `keep_all` | N | 全部保留 | 原始位置（用于检验"机制本身不改变输出"） |
| 随机 | `random` | max(1, int(N × keep)) | 用固定种子（`VTS_SEED`）对每张图做随机排列，保留前 k 个，保持原来的空间顺序 | 原始位置 |
| 2×2 池化 | `pool2x2` | ⌈h/2⌉ × ⌈w/2⌉（约 25%） | 合并后的网格上每 2×2 求平均（边缘不足 2 时按实际大小平均） | 取块左上角那个 token 的原始位置 |
| DivPrune | `divprune` | max(1, int(N × keep)) | 照官方实现：距离 = 1 − 余弦相似度；第一个选"离最近邻最远"的 token；之后每一步选"离已选集合最小距离最大"的；保持原空间顺序 | 原始位置 |
| RATE | `rate` | max(1, int(N × keep)) | 移植 PR 的 `compute_retention_mask_rate`（Apache-2.0，注明出处），输入只用特征部分 | 原始位置 |
| 降分辨率（B1） | 不需要插件 | — | 服务参数 `--mm-processor-kwargs '{"size": {"longest_edge": …}}'` | — |

配置全部通过环境变量传：`VTS_METHOD`、`VTS_KEEP`（保留比例）、`VTS_SEED`。vLLM 的 API server 和 EngineCore 两个进程都会继承环境变量，处理器（在 API server 里）和模型（在 EngineCore 里）看到的是同一套配置。

## 4. 正确性测试（W2 验收）

| # | 测试 | 怎么判断通过 | 负对照 |
|---|---|---|---|
| T1 | `keep_all` 和原版 vLLM 逐 token 一致 | 同一批图文请求，greedy 解码 64 个 token，两边输出完全相同 | `random`、keep 0.5 时输出必须不同 |
| T2 | 占位 token 数等于实际 embedding 数 | 插件内部做断言，测试里再统计 prompt_tokens：减少的数量等于 N − k | 故意让 `keep_count` 算错时，必须报错 |
| T3 | 打开或关闭 chunked prefill、prefix caching，结果一致 | 每步 token 预算设成 256，强制把一张图切成多段 prefill；输出和不切时一致；两次相同请求的输出一致 | — |
| T4 | 选择算子的单元测试 | DivPrune 和官方实现逐项相同；RATE 和 PR 版本相同；所有方法的保留数都等于 `keep_count`；随机方法用同一种子时结果确定 | 换种子后随机结果改变 |
| T5 | 和 HF 参考实现交叉检验 | 把同一个掩码用到 HF transformers 上（重写 `get_image_features` 和位置），前 32 个 token 一致 | W0 已知：即使不压缩，两边在 5 个样本里也有 1 个不一致，要先测出这个基线差异 |

T1–T4 是 W2 的硬性验收；T5 尽量做，做不完就在 W3 开始前补上。

## 5. 风险

| 风险 | 应对 |
|---|---|
| 依赖 vLLM 内部函数，升级后会失效 | 锁定 vllm==0.32.0；插件启动时检查版本，版本不对就直接报错 |
| `ModelRegistry.register_model` 覆盖内置架构时有警告，或者注册方式发生变化 | T1 断言插件的子类确实被加载了（插件加载时打一行特定日志） |
| 打开裁剪机制后 `mrope_positions_factor` 变成 4，可能带来少量额外开销 | `keep_all` 和原版的速度对比会在 W4 里单独报告 |
| 插件需要装进 `envs/vts` 才能被发现 | 每次部署时 `pip install --no-deps` 安装当前 release 目录里的插件，并把版本和 commit 写进日志 |
