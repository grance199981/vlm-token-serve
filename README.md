# vlm-token-serve

**English** | [中文](#中文)

An out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugin that compresses the visual tokens of image inputs, plus a benchmark CLI that measures where the savings show up in serving. Two models with different attention structures are compared under the same protocol:

- **Qwen3-VL-4B-Instruct**: full attention in every layer.
- **Qwen3.5-4B**: hybrid attention, 8 of 32 layers use full attention and the rest use linear attention.

The plugin needs no attention scores, so it runs with vLLM's fused attention kernels and CUDA graphs. It is configured entirely through environment variables. When `VTS_METHOD` is unset, vLLM behaves exactly as stock.

## Results

All numbers come from `docs/results/`. Accuracy is compared only against each model's own uncompressed run (B0), with paired bootstrap 95% CIs. Serving was measured on one NVIDIA L40 48 GB with vLLM 0.32.0. Under sustained load the L40 hits its 300 W power cap, and bf16 SM clocks drop from about 2250 to 1550 MHz ([FOLLOWUP3](docs/results/FOLLOWUP3_RESULTS.md)). All throughput numbers are therefore measured at power-capped clocks.

![accuracy vs throughput](docs/figures/pareto.png)

![serving gain by workload](docs/figures/throughput_by_workload.png)

| Finding | Qwen3-VL-4B-Instruct (full) | Qwen3.5-4B (hybrid) | Source |
|---|---|---|---|
| Prefill-bound throughput at keep 50%, concurrency 64. Downsampling skips part of the ViT, token pruning does not. | downsample 2.05–2.08×, random and RATE 1.57–1.63× | downsample 1.94–2.13×, random and RATE 1.49–1.61× | [W4 §5](docs/results/W4_RESULTS.md) |
| KV-bound admitted concurrency at keep 50%, 4 GiB KV, concurrency 128 | 1.744–1.768× | 1.267–1.280× | [W4 §3](docs/results/W4_RESULTS.md) |
| Decode-bound TPOT at keep 50%, concurrency 64 | −25.3% to −27.8% | −9.2% to −14.9% | [W4 §6](docs/results/W4_RESULTS.md) |
| Prefix-cache capacity in multi-turn image chat (S\*, relative to B0) | 1.57–1.59× | 1.17–1.20× | [K2](docs/results/K2_RESULTS.md) |
| Triton DivPrune kernel, one 784-token image at keep 50% | 25.66 → 0.419 ms (61×) | same kernel | [W5 §1](docs/results/W5_RESULTS.md) |
| DivPrune throughput, prefill-bound, concurrency 64 (PyTorch → Triton) | 1.281× → 1.584× | 1.279× → 1.566× | [W5 §1](docs/results/W5_RESULTS.md) |
| FP8 weights alone (OCR Δ) and FP8 + downsample 50%, prefill-bound, concurrency 64 | 1.450× (+0.02) and 3.053× | 1.558× (+0.01) and 3.062× | [W5 §2](docs/results/W5_RESULTS.md) |

Main observations:

1. **On the OCR average, downsampling is on the accuracy–throughput frontier for both models.** At keep 50% it loses 2.2 (Qwen3-VL) and 2.0 (Qwen3.5) points against 5.4 and 2.2 for RATE, at higher throughput. The one setting where pruning is not dominated is TextVQA (small scene text) on Qwen3.5, where RATE 50% changes accuracy by −0.4 [−1.2, +0.5] against −2.6 [−3.9, −1.3] for downsampling ([W4 §7](docs/results/W4_RESULTS.md), [W3](docs/results/W3_RESULTS.md)).
2. **Attention structure decides how much of the KV saving turns into capacity.** Halving the visual tokens gives 1.75× admitted concurrency on full attention and 1.27× on hybrid attention, because most Qwen3.5 layers keep a constant-size recurrent state. A block-ledger model registered before the runs predicts the gain within 0.937–1.064 (measured over predicted) on two new conditions ([FOLLOWUP](docs/results/FOLLOWUP_RESULTS.md)).
3. **On the hybrid model, compression can lower the prefix-cache hit rate.** Hits are aligned to 528-token blocks. After compression the shared prefix of turns 1–3 is shorter than 528 tokens, and with 8 sessions in rotation the hit rate drops from 53% to 29% ([K2](docs/results/K2_RESULTS.md)).
4. **Selection cost matters at high load.** The PyTorch DivPrune loop is launch-bound (about 1600 small kernels per image). One Triton kernel brings it to the throughput of random selection with unchanged accuracy (difference ≤ 0.4 points).

Accuracy of the uncompressed models through the same `vllm serve` path ([W1](docs/results/W1_RESULTS.md)):

| Task (metric) | Qwen3-VL-4B-Instruct | Qwen3.5-4B |
|---|---|---|
| ChartQA test (relaxed acc.) | 84.04 [82.68, 85.48] | 85.28 [84.00, 86.64] |
| TextVQA val (exact match) | 80.50 [79.47, 81.51] | 82.53 [81.62, 83.46] |
| DocVQA val (ANLS) | 94.75 [94.21, 95.27] | 95.56 [95.06, 96.05] |

OCR-average accuracy change against B0 ([W3 §4b.3](docs/results/W3_RESULTS.md)):

| Model | Method | keep 50% | keep 25% | keep 12.5% |
|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | downsample | **−2.2** | **−10.2** | **−22.5** |
| | RATE (`rate`) | −5.4 | −24.4 | −41.1 |
| | DivPrune | −6.3 | −18.5 | −34.8 |
| | random | −8.7 | −26.2 | −44.1 |
| Qwen3.5-4B | downsample | **−2.0** | **−9.0** | **−21.7** |
| | RATE (`rate`) | −2.2 | −11.4 | −31.9 |
| | DivPrune | −4.8 | −15.7 | −31.3 |
| | random | −11.9 | −32.8 | −49.4 |

## Methods

| `VTS_METHOD` | What it does |
|---|---|
| `none` | plugin off (stock vLLM) |
| `downsample` | resize each image by √keep per side before the HF processor |
| `random` | fixed random subset per (seed, token count) |
| `pool2x2` | average 2×2 blocks of the merged token grid |
| `divprune` | greedy max-min diversity, following the [official DivPrune code](https://github.com/vbdi/divprune), `VTS_DIVPRUNE_IMPL=triton` for the fused kernel |
| `rate` | RATE from vLLM PR [#60963](https://github.com/vllm-project/vllm/pull/60963) with a descending projection sort |
| `rate_pr` | the PR as written at `06158cbe`. Its ascending sort always keeps a prefix of the tokens ([reported](https://github.com/vllm-project/vllm/pull/60963#issuecomment-6321576342)). |
| `rate_orig` | the PR after the author's fix (`45333ba`), copied line for line |

Other variables are `VTS_KEEP` (fraction kept, default 1.0), `VTS_SEED` (default 0), `VTS_TAIL`, `VTS_VIT_TRACE=<dir>` (per-call ViT timing) and `VTS_STEP_TRACE=<dir>` (per-step engine timing). See [`src/vts_plugin/config.py`](src/vts_plugin/config.py).

How it plugs into vLLM: the multimodal processor shrinks each image placeholder to the kept token count, so the scheduler and the KV cache only see kept tokens. The model runs the vision tower in profile-sized groups, selects tokens, and re-places them at their original M-RoPE positions through vLLM's multimodal-pruning path. A walkthrough in Chinese is in [`docs/CODE_GUIDE_zh.md`](docs/CODE_GUIDE_zh.md), and the design notes are in [`docs/W2_PLUGIN_DESIGN.md`](docs/W2_PLUGIN_DESIGN.md).

## Quick start

```bash
# in an environment with vllm==0.32.0
pip install --no-deps .
VTS_METHOD=downsample VTS_KEEP=0.5 vllm serve Qwen/Qwen3-VL-4B-Instruct
```

The plugin is loaded through the `vllm.general_plugins` entry point in every vLLM process.

## Benchmark CLI

`vts-bench` reproduces the measurement protocol: one server per (model, method), a discarded warm-up with its own seed, per-point seeds shared across methods for paired comparison, and closed-loop synthetic load through `vllm bench serve`.

```bash
export VTS_ROOT=/path/to/root               # envs/vts, models/<name>, tasks_local/
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx    # a GPU UUID; the CLI refuses a busy GPU
vts-bench serve --preset P --methods none downsample random rate divprune_triton --concurrency 1 8 32 64 --out runs/p
vts-bench serve --preset K1 --kv-bytes 4294967296 --concurrency 128 --out runs/k1
vts-bench summarize serve runs/p/results    # gains vs B0, paired per repetition
vts-bench accuracy --methods none downsample --tasks vts_chartqa vts_textvqa_val --out runs/acc
```

| Preset | Request shape | Bottleneck |
|---|---|---|
| `P` | 4 images 896×896, 64 text tokens in, 32 out | prefill compute, swept over `--max-num-batched-tokens` 2048 and 8192 |
| `D` | 1 image, 512 out | decode |
| `K1` | 2 images, 256 out, same KV byte budget for both models | KV capacity |

`tests/test_bench.py` checks that the CLI builds the same `vllm serve`, `vllm bench serve` and lmms-eval commands as the original runners and gives identical statistics on the same inputs.

## Reproducing the published numbers

The tables above were produced by the scripts in `experiments/`, one directory per milestone. Each script documents its usage in its header. `experiments/common/env.sh` holds the shared environment.

| Results | Scripts |
|---|---|
| [W1](docs/results/W1_RESULTS.md) baselines | `experiments/w1/` (accuracy, serving), `experiments/w0/` (environment smoke tests) |
| [W2](docs/results/W2_RESULTS.md) plugin integration, HF cross-check | `experiments/w2/` |
| [W3](docs/results/W3_RESULTS.md) accuracy sweep | `experiments/w3/`, `experiments/routec/` |
| [W4](docs/results/W4_RESULTS.md) serving (P, D, K1, Pareto) | `experiments/w4/` |
| [W5](docs/results/W5_RESULTS.md) Triton DivPrune, FP8 | `experiments/w5/` |
| [K2](docs/results/K2_RESULTS.md), [FOLLOWUP](docs/results/FOLLOWUP_RESULTS.md), [FOLLOWUP2](docs/results/FOLLOWUP2_RESULTS.md), [FOLLOWUP3](docs/results/FOLLOWUP3_RESULTS.md) | `experiments/w6/` |
| upstream reproducers | `experiments/d/` |
| figures | `experiments/plots/make_figures.py` (from `docs/results/data/*.csv`) |

Hypotheses and pass criteria were written down before each run in [`docs/plans/`](docs/plans/). The result files report each verdict against them, including the hypotheses that were falsified.

`tests/test_selection_hashes.py` locks the token selection of every method to the code version that produced these results.

## Upstream reports

- [vllm-project/vllm#63187](https://github.com/vllm-project/vllm/issues/63187): with multimodal pruning on, the scheduler debits the encoder budget by the pruned token count, so one step can admit several times the profiled ViT input and run out of memory. Reproducer in `experiments/d/encoder_budget_repro.py`. The plugin works around it in `src/vts_plugin/vitbatch.py`.
- [vllm-project/vllm#62937](https://github.com/vllm-project/vllm/issues/62937): with transformers ≥ 5.10, the M-RoPE positions recomputed after Qwen3-VL video pruning are misaligned (a crash with a short text prefix, silently wrong positions with a long one). Reproducible on CPU with `experiments/d/evs_outer_vs_min_repro.py`.
- [Comment on PR #60963](https://github.com/vllm-project/vllm/pull/60963#issuecomment-6321576342): the ascending sort in RATE makes the bucket denominator negative, so the PR keeps a prefix of the tokens for every input.

## Repository layout

```
src/vts_plugin/   the vLLM plugin (selection methods, Triton kernel, M-RoPE recompute, ViT grouping, traces)
src/vts_bench/    the vts-bench CLI
tests/            CPU tests (GPU, Triton and vLLM tests skip when unavailable)
experiments/      the scripts behind the published numbers
docs/results/     result reports (Chinese), with tables and verdicts
docs/plans/       pre-registered plans and criteria (Chinese)
```

## License

Apache-2.0. `rate_pr`, `rate_orig` and `tests/fixtures/rate_pr_45333ba.py` are adapted from vLLM PR #60963 (Apache-2.0, Copyright contributors to the vLLM project). See [NOTICE](NOTICE).

---

## 中文

一个 [vLLM](https://github.com/vllm-project/vllm) 树外插件，对图像输入的视觉 token 做压缩，并附带一个评测命令行，用来测量压缩在 serving 中的收益具体体现在哪里。同一套协议下对比两种注意力结构：

- **Qwen3-VL-4B-Instruct**：每一层都是全注意力。
- **Qwen3.5-4B**：混合注意力，32 层里 8 层是全注意力，其余是线性注意力。

插件不依赖注意力分数，所以可以照常使用 vLLM 的融合注意力 kernel 和 CUDA graph。配置全部通过环境变量完成。不设 `VTS_METHOD` 时，vLLM 的行为与原版完全一致。

### 结果

数字全部来自 `docs/results/`。精度只和各模型自己的不压缩运行（B0）比较，区间是配对 bootstrap 95% CI。serving 在单张 NVIDIA L40 48 GB 上测量，vLLM 0.32.0。持续高负载时 L40 触及 300 W 功耗上限，bf16 的 SM 时钟从约 2250 MHz 降到约 1550 MHz（[FOLLOWUP3](docs/results/FOLLOWUP3_RESULTS.md)），所以所有吞吐数字都是在功耗降频状态下测得的。

| 发现 | Qwen3-VL-4B-Instruct（全注意力） | Qwen3.5-4B（混合注意力） | 出处 |
|---|---|---|---|
| prefill 受限负载的吞吐，保留 50%，并发 64。降分辨率省掉一部分 ViT 计算，裁剪省不掉 | 降分辨率 2.05–2.08×，随机和 RATE 1.57–1.63× | 降分辨率 1.94–2.13×，随机和 RATE 1.49–1.61× | [W4 §5](docs/results/W4_RESULTS.md) |
| KV 受限负载的准入并发，保留 50%，KV 4 GiB，并发 128 | 1.744–1.768× | 1.267–1.280× | [W4 §3](docs/results/W4_RESULTS.md) |
| decode 受限负载的 TPOT，保留 50%，并发 64 | −25.3% 到 −27.8% | −9.2% 到 −14.9% | [W4 §6](docs/results/W4_RESULTS.md) |
| 多轮带图会话的前缀缓存容量（S\*，相对 B0） | 1.57–1.59× | 1.17–1.20× | [K2](docs/results/K2_RESULTS.md) |
| Triton DivPrune kernel，一张 784 token 的图，保留 50% | 25.66 → 0.419 ms（61×） | 同一个 kernel | [W5 §1](docs/results/W5_RESULTS.md) |
| DivPrune 吞吐，prefill 受限，并发 64（PyTorch → Triton） | 1.281× → 1.584× | 1.279× → 1.566× | [W5 §1](docs/results/W5_RESULTS.md) |
| 单用 FP8 权重（OCR Δ），以及 FP8 + 降分辨率 50%，prefill 受限，并发 64 | 1.450×（+0.02），3.053× | 1.558×（+0.01），3.062× | [W5 §2](docs/results/W5_RESULTS.md) |

主要结论：

1. **按 OCR 平均，降分辨率在两个模型上都处于精度–吞吐前沿。** 保留 50% 时掉 2.2 分（Qwen3-VL）和 2.0 分（Qwen3.5），RATE 掉 5.4 分和 2.2 分，而且降分辨率的吞吐更高。裁剪唯一不被支配的位置是 Qwen3.5 的 TextVQA（自然场景小字）：RATE 50% 精度变化 −0.4 [−1.2, +0.5]，降分辨率 −2.6 [−3.9, −1.3]（[W4 §7](docs/results/W4_RESULTS.md)、[W3](docs/results/W3_RESULTS.md)）。
2. **注意力结构决定了 KV 节省能有多少转化成容量。** 视觉 token 减半后，全注意力的准入并发提高到 1.75×，混合注意力只有 1.27×，因为 Qwen3.5 的大多数层保存的是大小固定的循环状态。运行前登记的块账本模型在两个新条件下预测准入增益，实测 ÷ 预测在 0.937–1.064 之间（[FOLLOWUP](docs/results/FOLLOWUP_RESULTS.md)）。
3. **在混合模型上，压缩可能降低前缀缓存命中率。** 命中按 528 token 的块对齐。压缩后第 1–3 轮的共同前缀都不足 528 token，8 个会话轮转时命中率从 53% 降到 29%（[K2](docs/results/K2_RESULTS.md)）。
4. **高负载下，挑选 token 的开销不能忽略。** PyTorch 版 DivPrune 的循环受 kernel 启动开销限制，每张图约 1600 个小 kernel。合并成一个 Triton kernel 后，吞吐与随机挑选持平，精度不变（差 ≤ 0.4 分）。

精度基线和各方法、各保留率的精度变化见上方英文部分的两张表，以及 [W1](docs/results/W1_RESULTS.md)、[W3](docs/results/W3_RESULTS.md)。

### 方法与用法

支持的方法见上方英文部分的 Methods 表，环境变量的完整说明见 [`src/vts_plugin/config.py`](src/vts_plugin/config.py)。插件在 vLLM 中的挂载方式按请求路径逐步讲解，见中文代码导读 [`docs/CODE_GUIDE_zh.md`](docs/CODE_GUIDE_zh.md)。

```bash
# 环境中需要 vllm==0.32.0
pip install --no-deps .
VTS_METHOD=downsample VTS_KEEP=0.5 vllm serve Qwen/Qwen3-VL-4B-Instruct
```

`vts-bench` 复现了测量协议：每个（模型，方法）组合单独起一个服务；正式测量前先做一次热身，热身用单独的种子，结果丢弃；每个测量点的种子在所有方法之间共用，以便配对比较；负载由 `vllm bench serve` 生成，是闭环的合成负载。用法和预设见英文部分的 Benchmark CLI 一节。`tests/test_bench.py` 检查它和原始实验脚本生成的命令一致，并且在相同输入上算出相同的统计量。

### 复现已发布的数字

上面的表格由 `experiments/` 下的脚本产出，每个里程碑一个目录，每个脚本的用法写在文件开头。每次运行前的假设和判据记录在 [`docs/plans/`](docs/plans/)，结果文档按这些判据逐条给出结论，包括被证伪的假设。`tests/test_selection_hashes.py` 把每种方法的挑选结果锁定在产出这些结果时的代码版本上。

### 上游报告

- [vllm-project/vllm#63187](https://github.com/vllm-project/vllm/issues/63187)：开启多模态裁剪后，调度器按裁剪后的 token 数扣减编码器预算，一步可能放进数倍于 profile 的 ViT 输入，导致显存不足。插件在 `src/vts_plugin/vitbatch.py` 中绕开了这个问题。
- [vllm-project/vllm#62937](https://github.com/vllm-project/vllm/issues/62937)：transformers ≥ 5.10 时，Qwen3-VL 视频裁剪后重算的 M-RoPE 位置整体错位（文本前缀短时直接崩溃，前缀长时静默写错位置）。
- [PR #60963 下的评论](https://github.com/vllm-project/vllm/pull/60963#issuecomment-6321576342)：RATE 的升序排序使分桶分母恒为负，导致 PR 对任何输入都只保留前缀。

### 许可证

Apache-2.0，见 [NOTICE](NOTICE)。
