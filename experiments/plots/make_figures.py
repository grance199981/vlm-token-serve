"""README figures from the published tables (docs/results/data/*.csv, transcribed from
docs/results/W4_RESULTS.md; the `source` column names the table).

Usage: python experiments/plots/make_figures.py   (writes docs/figures/*.png)
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA, OUT = REPO / "docs/results/data", REPO / "docs/figures"
MODELS = ["Qwen3-VL-4B-Instruct", "Qwen3.5-4B"]
TITLES = {"Qwen3-VL-4B-Instruct": "Qwen3-VL-4B-Instruct (full attention)",
          "Qwen3.5-4B": "Qwen3.5-4B (hybrid attention)"}
# categorical slots 1-4 of the validated reference palette, fixed order
COLORS = {"downsample": "#2a78d6", "random": "#eb6834", "rate": "#1baf7a", "divprune": "#eda100"}
LABELS = {"downsample": "downsample", "random": "random", "rate": "RATE", "divprune": "DivPrune"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pareto():
    data = rows("pareto.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    for ax, model in zip(axes, MODELS):
        style(ax)
        pts = [r for r in data if r["model"] == model]
        b0 = next(r for r in pts if r["method"] == "none")
        for method in ("downsample", "rate"):
            line = [b0] + sorted((r for r in pts if r["method"] == method), key=lambda r: -float(r["keep"]))
            xs = [float(r["throughput_gain"]) for r in line]
            ys = [float(r["ocr_avg_delta"]) for r in line]
            ax.plot(xs, ys, color=COLORS[method], linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[method], zorder=3)
            for r in line[1:]:
                ax.annotate(f"{int(float(r['keep']) * 100)}%", (float(r["throughput_gain"]), float(r["ocr_avg_delta"])),
                            textcoords="offset points", xytext=(8, -3), fontsize=9, color=INK2)
        ax.annotate("B0", (1.0, 0.0), textcoords="offset points", xytext=(6, 5), fontsize=9, color=INK2)
        ax.set_title(TITLES[model], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("throughput vs B0 (P workload, concurrency 64)", fontsize=9, color=INK2)
    axes[0].set_ylabel("OCR accuracy change vs B0 (points)", fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    fig.suptitle("Accuracy vs throughput, keep 50% / 25% (OCR = mean of TextVQA, ChartQA, DocVQA)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=160, facecolor=SURFACE)


def throughput():
    data = rows("throughput_by_workload.csv")
    workloads = ["P", "D", "K1"]
    wl_names = {"P": "P: prefill-bound\n4 images, 32 out", "D": "D: decode-bound\n1 image, 512 out",
                "K1": "K1: KV-bound\n4 GiB KV, 2 images"}
    methods = list(COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    width = 0.19
    for ax, model in zip(axes, MODELS):
        style(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.grid(False, axis="x")
        for mi, method in enumerate(methods):
            vals = [float(next(r for r in data if r["model"] == model and r["workload"] == w
                               and r["method"] == method)["throughput_gain"]) for w in workloads]
            xs = [wi + (mi - 1.5) * (width + 0.02) for wi in range(len(workloads))]
            ax.bar(xs, [v - 1 for v in vals], bottom=1, width=width, color=COLORS[method],
                   label=LABELS[method], edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.axhline(1.0, color=INK2, linewidth=1, zorder=4)
        ax.set_xticks(range(len(workloads)), [wl_names[w] for w in workloads], fontsize=9, color=INK2)
        ax.set_title(TITLES[model], fontsize=11, color=INK, loc="left")
        ax.set_ylim(0.9, 2.3)
    axes[0].set_ylabel("throughput vs B0 (×), keep 50%", fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK, ncol=2)
    fig.suptitle("Serving gain by workload (L40, power-capped clocks)", fontsize=12, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "throughput_by_workload.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pareto()
    throughput()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
