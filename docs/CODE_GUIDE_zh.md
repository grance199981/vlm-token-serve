# 代码导读：一个图像请求在插件里怎么走

本文跟着一个带图请求，从 vLLM 进程启动一路走到 decode，逐段说明插件改了什么、为什么这样改。行号对应当前版本的 `src/vts_plugin/`。

## 0. 总体结构

vLLM 0.32 的在线服务分两个进程。

- **API server 进程**：跑多模态预处理器。它决定 prompt 里每张图占多少个占位 token，调度器和 KV cache 只认这个数。
- **EngineCore 进程**：跑模型，包括 ViT、语言模型、调度和 KV cache 管理。

裁剪要在两个进程里同时生效：API server 把占位符改短，EngineCore 把 ViT 输出挑成同样多的行。两边任何一处数量不一致，请求都会直接报错。所以插件把全部配置放在环境变量里（`config.py`），两个进程读到的配置完全相同。

| 文件 | 职责 | 依赖 |
|---|---|---|
| `config.py` | 解析 `VTS_*` 环境变量，`keep_count` 算每张图保留多少 token | 无 |
| `select.py` | 各挑选方法，输入一张图的特征，输出保留的行 | 只有 torch |
| `divprune_triton.py` | DivPrune 的单 kernel 实现 | torch、triton |
| `mrope.py` | 裁剪后的 M-RoPE 位置 | numpy、torch |
| `resize.py` | 降分辨率基线 | PIL |
| `vitbatch.py` | 按 profile 大小把 ViT 输入分组 | torch |
| `models.py` | 把以上各部分接进 vLLM 的 Qwen3-VL / Qwen3.5 模型类 | vLLM |
| `vittrace.py`、`steptrace.py` | 可选的 ViT 调用计时和引擎每步计时 | vLLM |

挑选算法和 vLLM 解耦：`select.py`、`mrope.py`、`vitbatch.py` 都可以脱离 vLLM 在 CPU 上做单元测试。

## 1. 入口点注册：`__init__.register`

`pyproject.toml` 里的 `[project.entry-points."vllm.general_plugins"]` 让 vLLM 在每个进程启动时调用 `register()`（[`__init__.py:21`](../src/vts_plugin/__init__.py)）。它做三件事。

1. 按环境变量决定是否装上两种 trace。这一步不依赖 `VTS_METHOD`，所以不压缩的运行（B0）也能计时。
2. 打开 vLLM 的多模态裁剪开关（[`__init__.py:43-48`](../src/vts_plugin/__init__.py)）。vLLM 0.32 为视频的 EVS 裁剪准备了一条路径：模型在视觉特征后面附上每个 token 的位置，V2 model runner 拿到后重算 M-RoPE。这条路径只在 `is_multimodal_pruning_enabled()` 为真时启用，插件把它改成"只要在裁剪就返回真"，从而把图像也接到这条路径上。
3. 用 `ModelRegistry.register_model` 把两个架构名替换成插件的子类（[`__init__.py:51-56`](../src/vts_plugin/__init__.py)）。不设 `VTS_METHOD` 时直接返回，vLLM 与原版完全一致。

## 2. 预处理器：改写占位符数量

`VTSQwen3VLProcessor`（[`models.py:41`](../src/vts_plugin/models.py)）运行在 API server 进程。

- `_get_prompt_updates`（[`models.py:53`](../src/vts_plugin/models.py)）把每张图的占位 token 数从 `gh × gw` 改成 `keep_count(...)`（[`config.py:101`](../src/vts_plugin/config.py)）。从这一刻起，调度器、前缀缓存和 KV cache 看到的都是裁剪后的长度，显存和计算的节省都从这里来。
- `keep_count` 必须在两个进程里给出同一个数：`pool2x2` 按网格向上取整，其余方法是 `int(N × keep)`，至少保留 1 个。

## 3. 降分辨率在预处理阶段生效

`downsample` 不走裁剪路径，`cfg.prunes` 为假。`_get_hf_mm_inputs`（[`models.py:44`](../src/vts_plugin/models.py)）在 HF 处理器之前把每张图的两条边各缩小到 √keep（[`resize.py:14`](../src/vts_plugin/resize.py)），token 数大约变成 keep 倍。它和裁剪最大的区别是 ViT 的输入也变小了。W4 的 prefill 受限负载下，降分辨率的吞吐增益最大，原因就在这里。

## 4. ViT 按 profile 大小分组：`vitbatch.run_grouped`

这一段解决一个只在裁剪时才出现的显存问题（已报告为 [vllm#63187](https://github.com/vllm-project/vllm/issues/63187)）。

- vLLM 启动时用一张最大尺寸的图 profile ViT 的峰值显存，然后按这个峰值预留空间。
- 调度器扣编码器预算时，按的是**占位 token 数**。裁剪后占位符变短，同样的预算一步能放进约 1/keep 倍的图。这些图在 ViT 里仍然是完整的，激活显存超出 profile，结果就是 OOM。
- 插件在 `_process_image_input`（[`models.py:88`](../src/vts_plugin/models.py)）里把一批图按 patch 总数分成若干组（[`vitbatch.py:21`](../src/vts_plugin/vitbatch.py)），每组不超过 profile 用的那张最大图（[`vitbatch.py:37`](../src/vts_plugin/vitbatch.py)），然后逐组调用 ViT（[`vitbatch.py:53`](../src/vts_plugin/vitbatch.py)）。这样峰值回到 profile 水平，不需要改调度器。
- 服务端覆盖了像素上限时，profile 的大小无法从配置文件推出，cap 取 0，也就是每张图单独调用一次 ViT，这是保守做法。

## 5. 挑选：`_postprocess_image_embeds_evs` 与 `select.py`

ViT 和 merger 的输出进入 `_postprocess_image_embeds_evs`（[`models.py:105`](../src/vts_plugin/models.py)）。父类已经在每个 token 后面附上 5 个位置通道 `[t, h, w, llm_grid_w, ·]`，插件把第 5 个通道填成 `llm_grid_h`（[`models.py:117`](../src/vts_plugin/models.py)），然后对每张图调用一次 `select`（[`models.py:119`](../src/vts_plugin/models.py)）。

`select`（[`select.py:237`](../src/vts_plugin/select.py)）按方法名在分发表 `_METHODS`（[`select.py:224`](../src/vts_plugin/select.py)）里找到对应函数，参数全部按关键字传递。所有方法遵守同一个约定：返回恰好 k 行，并按原来的光栅顺序排好。行顺序不变，后面的位置重算才能直接使用。

| 方法 | 位置 | 要点 |
|---|---|---|
| `random` | [`select.py:40`](../src/vts_plugin/select.py) | 种子由 (seed, N) 决定，同一张图在任何批次里都选同一组 |
| `pool2x2` | [`select.py:48`](../src/vts_plugin/select.py) | 2×2 块取平均，用 fp32 累加，位置取块的左上角 |
| `divprune` | [`select.py:69`](../src/vts_plugin/select.py) | 贪心 max-min 多样性。每步做一次 (N, d) @ (d, 1)，增量更新最小距离，不构造 N×N 矩阵 |
| `rate` / `rate_pr` | [`select.py:123`](../src/vts_plugin/select.py) | 随机投影后排序分桶，每桶取一个锚点，按与锚点的平均相似度打分，低分（更独特）的先保留 |
| `rate_orig` | [`select.py:184`](../src/vts_plugin/select.py) | PR 作者修复后的版本，逐行照抄。它和 `_rate_scores` 有重复，这是有意保留的：与 PR 完全一致才能复现 |

`score_dim`（[`models.py:112`](../src/vts_plugin/models.py)）只取主 merger 的输出列来打分。Qwen3-VL 每行还拼着 3 层 DeepStack 特征（共 4 × 2560 列），全部拿来打分要额外占用 fp32 显存，大 DocVQA 图上会 OOM。挑选只决定保留哪些行，保留下来的行仍然是完整的。

## 6. M-RoPE 位置：`mrope.py`

Qwen3-VL / Qwen3.5 用三维 M-RoPE，图像 token 的位置是 (t, h, w)。裁剪之后，有两处位置需要处理。

1. **预处理时的占位位置**：`_get_mrope_input_positions`（[`models.py:145`](../src/vts_plugin/models.py)）调用 `initial_positions`（[`mrope.py:26`](../src/vts_plugin/mrope.py)）。图后面的文本从 `base + max(gh, gw)` 开始，和不裁剪时完全一样（`VTS_TAIL=original`）。这样文本的位置在预处理时就已确定，前缀缓存命中、跳过编码器时，也不会留下旧的位置。
2. **编码器运行后的真实位置**：V2 runner 调用 `recompute_mrope_positions`（[`models.py:158`](../src/vts_plugin/models.py)），插件把每个保留 token 写回它在原网格上的 (t, h, w)（[`mrope.py:48`](../src/vts_plugin/mrope.py)）。和 vLLM 自带的 EVS 重算相比，这里按"第一个媒体 token 的全局下标"来定位所属的图，所以 prefill 分块的边界正好落在 `<vision_start>` 和第一个图像 token 之间时，也能正确处理。

测试里有两个故意制造的位置错误作负对照（`VTS_DEBUG_POS_SHIFT`、`VTS_DEBUG_POS_TRANSPOSE`），用来确认位置真的影响输出，测试不是空转。

## 7. Triton DivPrune kernel

PyTorch 版 DivPrune 的每一步都是一次 GEMV 加一次 argmax，前后依赖，一张 784 token、保留 50% 的图要启动约 1600 个小 kernel，耗时约 25 ms，瓶颈在 kernel 启动开销。`divprune_triton.py` 的做法：

1. 一次 cuBLAS GEMM 算出相似度矩阵 S = x xᵀ（[`divprune_triton.py:42`](../src/vts_plugin/divprune_triton.py)）。
2. 第一个点的选法与 PyTorch 版相同（[`divprune_triton.py:47`](../src/vts_plugin/divprune_triton.py)），全程在设备上完成，不做主机同步。
3. 一个 Triton program 跑完剩下的 k − 1 步（[`divprune_triton.py:28`](../src/vts_plugin/divprune_triton.py)）。最小距离向量常驻寄存器，每步读 S 的一行，取 argmax，并列时取最左边的下标。

N 超过 4096 时 S 太大，回退到 PyTorch 版（[`divprune_triton.py:76`](../src/vts_plugin/divprune_triton.py)），回退次数记在 `DIVPRUNE_TRITON_STATS` 里。S 由 GEMM 一次算出，PyTorch 版则是逐行 GEMV，两者末位可能不同，近似并列时的选择也可能不同。所以测试检查的是贪心算法的正确性证书，不要求逐位相同。结果：每张图 25.66 → 0.419 ms，并发 64 的吞吐与随机挑选持平（[W5](results/W5_RESULTS.md)）。

## 8. 两个 trace 开关

- `VTS_VIT_TRACE=<dir>`（[`vittrace.py`](../src/vts_plugin/vittrace.py)）：包住 ViT 的 forward，前后各同步一次 CUDA，每次调用记录一行耗时、图片数和 patch 数。同步会拖慢吞吐，所以只用来测时间占比和批次组成。
- `VTS_STEP_TRACE=<dir>`（[`steptrace.py`](../src/vts_plugin/steptrace.py)）：包住 `EngineCore.step` 和调度器，每步记录总耗时、`schedule()` 耗时、GC 耗时、排队数和调度 token 数，不增加 GPU 同步。

## 常见追问

**为什么复用 EVS 路径，不自己改 attention？**
EVS 路径已经处理好了"特征后附位置、runner 重算 M-RoPE"这一整套流程，插件只需要换掉挑选规则和位置计算。不依赖注意力分数，FlashAttention 和 CUDA graph 都不受影响。vLLM 维护者也表示过不希望模型返回注意力分数这类额外输出。

**裁剪为什么在 prefill 受限时赢不了降分辨率？**
裁剪要先把整张图跑完 ViT 才能挑选，省下的只是语言模型那部分计算。降分辨率连 ViT 的输入一起缩小。W4 ① 的单请求分阶段计时里，ViT 占 TTFT 的约四分之一。

**为什么全注意力的 KV 收益比混合注意力大？**
KV 占用只来自全注意力层。Qwen3-VL-4B 每层都是全注意力，每个 token 占 144 KiB KV。Qwen3.5-4B 只有 8 层是全注意力，每个 token 32 KiB，其余层的状态大小固定，不随 token 数增长。同样砍掉一半视觉 token，Qwen3.5 能省下的显存比例小得多（[W4 §3](results/W4_RESULTS.md)）。

**`rate` 和 `rate_pr` 有什么区别？**
PR 原文按升序排序，分桶的分母因此恒为负，所有分数都相同，结果对任何输入都只保留前缀。`rate` 改成降序，分桶逻辑才按设计生效。这个问题已在 PR 下[指出](https://github.com/vllm-project/vllm/pull/60963#issuecomment-6321576342)。PR 后来更新到 `45333ba`，那一版就是 `rate_orig`：仍然升序，但改了分母的判断条件，每个桶取分数最低的 token 作锚点。

**怎么保证重构没有改变已发布结果对应的行为？**
`tests/test_selection_hashes.py` 把每种方法在固定输入上的挑选结果做哈希，与产出已发布结果的代码版本比对，并带一个负对照（把排序方向反过来，哈希必须改变）。
