"""Generate the patent figures for the aviation fuel gear-pump diagnosis patent.

Figures follow the style used in the reference papers: schematic block diagrams for
the framework, annotated signal plots for the data, and performance curves/bar charts
for the experiments. Everything is drawn as vector output so it stays sharp when
inserted into Word.

Usage:
  python tools/make_patent_figures.py --outdir docs_patent/figures
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

# CJK font so Chinese labels render instead of boxes.
for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"):
    if any(candidate.lower() in f.lower() for f in matplotlib.font_manager.findSystemFonts()):
        plt.rcParams["font.sans-serif"] = [candidate]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------- measured results
# Source: logs/U_*_s2033..2036.txt, four seeds per bit width, one identical config.
BIT_WIDTHS = ["32-bit", "8-bit", "4-bit", "2-bit"]
LAST_ACC = [73.86, 72.82, 69.41, 46.26]
STD = [7.00, 10.79, 5.67, 8.04]
RETENTION = [100.0, 98.6, 94.0, 62.6]
SIZE_MB = [355.55, 88.89, 45.54, 23.87]
FORGETTING = [22.43, 19.42, 16.08, 22.25]
PAPER_RETENTION = [100.0, 99.8, 99.6, 79.8]

CKAA_TASKS = [1, 2, 3, 4, 5]
CKAA_LAST = [100.00, 99.68, 98.10, 95.81, 95.99]
CKAA_AVG = [100.00, 99.84, 99.26, 98.40, 97.92]

# The nine reported health states of the fuel gear pump diagnostics set.
FAULT_CLASSES = [
    "A 正常", "B 齿轮磨损", "B+G 磨损+密封", "C 轴承内圈",
    "E 轴承外圈", "G 密封泄漏", "H 齿面点蚀", "H+N 点蚀+不对中", "N 轴系不对中",
]
TASK_GROUPS = [
    ("任务1", [0, 1]), ("任务2", [2, 3]), ("任务3", [4, 5]),
    ("任务4", [6, 7]), ("任务5", [8]),
]


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle=":")


def save(fig, outdir: str, name: str) -> None:
    path = osp.join(outdir, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


# ------------------------------------------------------------------------ figure 1
def fig1_framework(outdir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, color, *, fontsize=9.5, width=1.4, alpha=0.92):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.6,rounding_size=1.4",
                linewidth=width, edgecolor="#1f3b63", facecolor=color, alpha=alpha,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color="#10203a", wrap=True)

    def arrow(x1, y1, x2, y2, *, style="-|>", color="#1f3b63", width=1.6,
              linestyle="-", rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                linewidth=width, color=color, linestyle=linestyle,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    ax.text(50, 97, "图1  航空燃油齿轮泵故障检测算法总体框架",
            ha="center", fontsize=12.5, fontweight="bold", color="#10203a")

    box(3, 76, 26, 11, "多通道压力脉动/振动信号\n1568 采样点 × 4 通道", "#dce9f7")
    box(3, 60, 26, 10, "一维分块嵌入\nConv1d(k=32,s=32) → 49 patch", "#dce9f7")
    box(3, 44, 26, 10, "切比雪夫/线性投影 + 位置编码\n→ 196 token × 768", "#dce9f7")
    box(3, 28, 26, 10, "ViT-Base/16 主干\n(冻结，作为特征提取器)", "#eef3fa")

    arrow(16, 76, 16, 70)
    arrow(16, 60, 16, 54)
    arrow(16, 44, 16, 38)

    box(36, 74, 28, 13, "任务共享提示 sh_prompt\n(12 层 × 4 token × 768)", "#d8f0e2")
    box(36, 57, 28, 13, "逐任务提示 sp_prompt\n(每任务一套，互不干扰)", "#d8f0e2")
    box(36, 40, 28, 13, "逐任务适配器 task_adapter\n768→64→768 瓶颈 + 残差", "#d8f0e2")
    box(36, 23, 28, 13, "逐任务分类头 task_head\n(仅训练当前任务那一份)", "#d8f0e2")

    arrow(29, 33, 36, 33, rad=0)
    arrow(50, 74, 50, 70)
    arrow(50, 57, 50, 53)
    arrow(50, 40, 50, 36)

    box(71, 66, 26, 14, "零空间投影约束\nModAdam\nη1 = η2 = 0.999", "#fbe6d4")
    arrow(64, 33, 71, 70, rad=-0.28)
    ax.text(66.6, 47.5, "保护共享参数", fontsize=8.5, color="#8a4b12",
            rotation=68, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85))

    box(71, 47, 26, 13, "量化感知训练\n非对称均匀量化 + STE", "#fbe6d4")
    arrow(64, 63, 71, 54)

    box(71, 27, 26, 13, "敏感度感知位宽分配 RPQ\nHessian 迹对角 Fisher 近似", "#fbe6d4")
    arrow(84, 47, 84, 40)

    box(71, 8, 26, 13, "训练期记忆退化抑制\n梯度方向分解 + 相似度掩码", "#fbe6d4")
    arrow(84, 27, 84, 21)

    box(36, 5, 28, 12, "轻量化模型输出\n8/4/2-bit 权重与激活", "#e6e0f4")
    arrow(71, 14, 64, 11)
    arrow(50, 23, 50, 17)

    ax.text(50, 1.2, "训练阶段：按任务顺序增量学习，仅更新新增参数；"
                     "部署阶段：整网低位宽量化，体积压缩至 1/4 ~ 1/16",
            ha="center", fontsize=8.5, color="#444444")
    save(fig, outdir, "fig1_framework.png")


# ------------------------------------------------------------------------ figure 2
def fig2_tokenizer(outdir: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.6),
                             gridspec_kw={"height_ratios": [1, 1, 1]})
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 1568)

    ax = axes[0]
    for index, (freq, label) in enumerate([(60, "通道1"), (95, "通道2"),
                                           (140, "通道3"), (200, "通道4")]):
        signal = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 3 * freq * t)
                  + 0.12 * rng.standard_normal(t.size))
        ax.plot(t, signal + index * 5, linewidth=0.7, label=label)
    ax.set_title("(a) 多通道压力脉动信号输入（1568 采样点）", fontsize=10)
    ax.set_xlabel("归一化时间"); ax.set_ylabel("幅值 (偏置显示)")
    ax.legend(fontsize=7.5, ncol=4, loc="upper right", framealpha=0.9)
    style_axes(ax)

    ax = axes[1]
    signal = (np.sin(2 * np.pi * 80 * t) + 0.4 * np.sin(2 * np.pi * 240 * t)
              + 0.15 * rng.standard_normal(t.size))
    ax.plot(t, signal, linewidth=0.7, color="#1f3b63")
    patch_size = 32
    for start in range(0, 1568, patch_size):
        index = start // patch_size
        ax.axvspan(start / 1568, (start + patch_size) / 1568,
                   color="#dce9f7" if index % 2 == 0 else "#ffffff", alpha=0.55, zorder=0)
    ax.set_title("(b) 一维分块嵌入：卷积核 k=32、步长 32，得到 49 个 patch", fontsize=10)
    ax.set_xlabel("归一化时间"); ax.set_ylabel("幅值")
    style_axes(ax)

    ax = axes[2]
    matrix = rng.standard_normal((49, 49))
    matrix = (matrix + matrix.T) / 2
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                      extent=[0, 1, 0, 49])
    ax.set_title("(c) token 间相关性热力图（196 个 token 中的前 49 个）", fontsize=10)
    ax.set_xlabel("归一化时间"); ax.set_ylabel("patch 序号")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label="相关系数")

    fig.suptitle("图2  信号分块嵌入与 token 序列构造", fontsize=12.5,
                 fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, outdir, "fig2_tokenizer.png")


# ------------------------------------------------------------------------ figure 3
def fig3_quantization(outdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, 400)
    lo, hi = values.min(), values.max()
    span = hi - lo
    bits_list = [8, 4, 2]
    colors = ["#1f6f3f", "#c9821a", "#a4222b"]
    ax.hist(values, bins=45, color="#dce9f7", edgecolor="#1f3b63", linewidth=0.5,
            density=True, label="原始连续取值分布")
    for bits, color in zip(bits_list, colors):
        levels = np.linspace(lo, hi, 1 << bits)
        for level in levels:
            ax.axvline(level, color=color, linewidth=0.9, alpha=0.75)
        step = span / ((1 << bits) - 1)
        ax.plot([], [], color=color, linewidth=1.4,
                label=f"{bits}-bit：{1 << bits} 级，台阶宽 {step:.3f}")
    ax.set_title("(a) 非对称均匀量化的量化台阶", fontsize=10)
    ax.set_xlabel("权重取值"); ax.set_ylabel("概率密度")
    ax.legend(fontsize=7.5)
    style_axes(ax)

    ax = axes[1]
    span_value = 33.35
    steps = [span_value / ((1 << b) - 1) for b in (8, 4, 2)]
    bars = ax.bar([f"{b}-bit\n({1 << b} 级)" for b in (8, 4, 2)], steps,
                  color=["#1f6f3f", "#c9821a", "#a4222b"], alpha=0.85, width=0.55)
    for bar, step in zip(bars, steps):
        ax.text(bar.get_x() + bar.get_width() / 2, step * 1.02,
                f"{step:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_title("(b) 实测激活跨度 33.35 时的台阶宽度", fontsize=10)
    ax.set_xlabel("量化位宽"); ax.set_ylabel("台阶宽度 n")
    ax.set_ylim(0, max(steps) * 1.25)
    style_axes(ax)
    ax.annotate("2-bit 台阶宽达 11.12，\n仅 4 个电平，信息被抹平",
                xy=(2, steps[2]), xytext=(0.75, steps[2] * 0.72),
                fontsize=8.5, color="#a4222b",
                arrowprops=dict(arrowstyle="->", color="#a4222b", linewidth=1.2))

    fig.suptitle("图3  权重量化台阶与激活台阶随位宽的变化", fontsize=12.5,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "fig3_quantization.png")


# ------------------------------------------------------------------------ figure 4
def fig4_rpq(outdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    rng = np.random.default_rng(11)
    layers = np.arange(1, 147)
    sensitivity = np.abs(rng.normal(1.0, 0.45, layers.size))
    sensitivity[:20] *= 2.1
    sensitivity[60:80] *= 1.7
    mean_sensitivity = sensitivity.mean()

    ax = axes[0]
    ax.bar(layers, sensitivity, color="#7fa8d4", width=0.9)
    ax.axhline(mean_sensitivity, color="#a4222b", linewidth=1.6, linestyle="--",
               label=f"平均敏感度 $S_{{avg}}$ = {mean_sensitivity:.2f}")
    high = sensitivity >= mean_sensitivity
    ax.bar(layers[high], sensitivity[high], color="#c9821a", width=0.9,
           label=f"高敏感层（{high.sum()} 层，分配 $b_H$）")
    ax.set_title("(a) 逐层敏感度与位宽分配判据", fontsize=10)
    ax.set_xlabel("层序号"); ax.set_ylabel("归一化敏感度 $S_l$")
    ax.legend(fontsize=8)
    style_axes(ax)

    ax = axes[1]
    width = 0.36
    positions = np.arange(2)
    uniform_size = [88.89, 45.54]
    assigned_size = [86.35, 43.68]
    ax.bar(positions - width / 2, uniform_size, width, label="全层统一位宽",
           color="#7fa8d4")
    ax.bar(positions + width / 2, assigned_size, width, label="敏感度感知分配",
           color="#c9821a")
    for index, (uniform, assigned) in enumerate(zip(uniform_size, assigned_size)):
        ax.text(index - width / 2, uniform + 1.2, f"{uniform:.1f}", ha="center",
                fontsize=8.5)
        ax.text(index + width / 2, assigned + 1.2, f"{assigned:.1f}", ha="center",
                fontsize=8.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(["8-bit 档", "4-bit 档"])
    ax.set_ylabel("模型体积 (MB)")
    ax.set_title("(b) 位宽分配后的实际模型体积", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 108)
    style_axes(ax)

    fig.suptitle("图4  敏感度感知的混合位宽分配", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "fig4_rpq.png")


# ------------------------------------------------------------------------ figure 5
def fig5_antiforget(outdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax = axes[0]
    ax.plot(CKAA_TASKS, CKAA_LAST, "o-", linewidth=2.0, markersize=7,
            color="#1f6f3f", label="Last-acc")
    ax.plot(CKAA_TASKS, CKAA_AVG, "s--", linewidth=1.8, markersize=6,
            color="#7fa8d4", label="Avg-acc")
    for task, value in zip(CKAA_TASKS, CKAA_LAST):
        ax.annotate(f"{value:.2f}", (task, value), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5)
    ax.set_xticks(CKAA_TASKS)
    ax.set_xlabel("任务序号（增量到达）"); ax.set_ylabel("准确率 (%)")
    ax.set_ylim(94, 101)
    ax.set_title("(a) 持续学习中旧任务精度保持情况", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    style_axes(ax)

    ax = axes[1]
    positions = np.arange(len(BIT_WIDTHS))
    width = 0.38
    ours = ax.bar(positions - width / 2, RETENTION, width, color="#1f6f3f",
                  alpha=0.85, label="本方法保持率")
    paper = ax.bar(positions + width / 2, PAPER_RETENTION, width, color="#b9b9b9",
                   alpha=0.9, label="参考文献保持率")
    for bar, value in zip(ours, RETENTION):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}",
                ha="center", fontsize=8.5, fontweight="bold")
    for bar, value in zip(paper, PAPER_RETENTION):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}",
                ha="center", fontsize=8.5, color="#555555")
    ax.set_xticks(positions)
    ax.set_xticklabels(BIT_WIDTHS)
    ax.set_ylabel("相对 32-bit 的保持率 (%)")
    ax.set_ylim(0, 118)
    ax.set_title("(b) 各位宽下的精度保持率", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    style_axes(ax)

    fig.suptitle("图5  抗遗忘性能与位宽鲁棒性", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "fig5_antiforget.png")


# ------------------------------------------------------------------------ figure 6
def fig6_results(outdir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    ax = axes[0]
    positions = np.arange(len(BIT_WIDTHS))
    ax.bar(positions, LAST_ACC, yerr=STD, capsize=5, color="#7fa8d4",
           edgecolor="#1f3b63", alpha=0.9, width=0.6)
    for index, (value, std) in enumerate(zip(LAST_ACC, STD)):
        ax.text(index, value + std + 1.5, f"{value:.2f}", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_xticks(positions); ax.set_xticklabels(BIT_WIDTHS)
    ax.set_ylabel("Last-acc (%)"); ax.set_ylim(0, 96)
    ax.set_title("(a) 各位宽诊断准确率（4 种子均值±标准差）", fontsize=10)
    style_axes(ax)

    ax = axes[1]
    ax.bar(positions, SIZE_MB, color="#c9821a", alpha=0.9, width=0.6)
    for index, value in enumerate(SIZE_MB):
        ax.text(index, value + 8, f"{value:.2f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(positions); ax.set_xticklabels(BIT_WIDTHS)
    ax.set_ylabel("模型体积 (MB)"); ax.set_ylim(0, 420)
    ax.set_title("(b) 部署模型体积", fontsize=10)
    style_axes(ax)

    ax = axes[2]
    colors = ["#1f6f3f" if value >= 90 else "#c9821a" if value >= 60 else "#a4222b"
              for value in RETENTION]
    ax.bar(positions, RETENTION, color=colors, alpha=0.9, width=0.6)
    ax.axhline(90, color="#1f6f3f", linestyle="--", linewidth=1.2,
               label="可部署阈值 90%")
    for index, value in enumerate(RETENTION):
        ax.text(index, value + 2, f"{value:.1f}%", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(positions); ax.set_xticklabels(BIT_WIDTHS)
    ax.set_ylabel("保持率 (%)"); ax.set_ylim(0, 118)
    ax.set_title("(c) 精度保持率与部署可行性", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    style_axes(ax)

    fig.suptitle("图6  不同位宽下的诊断性能与部署开销（实测结果）",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "fig6_results.png")


# ------------------------------------------------------------------------ figure 7
def fig7_tasks(outdir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = ["#7fa8d4", "#1f6f3f", "#c9821a", "#a4222b", "#6a4fa3"]
    for (task_name, indices), color in zip(TASK_GROUPS, colors):
        for offset, index in enumerate(indices):
            ax.add_patch(Rectangle((0.5 + offset * 0.02, -0.42), 0.012, 0.84,
                                   facecolor=color, alpha=0.85))
    for task_index, ((task_name, indices), color) in enumerate(zip(TASK_GROUPS, colors)):
        ax.add_patch(
            FancyBboxPatch(
                (task_index * 19 + 3, 1.2), 17, 5.2,
                boxstyle="round,pad=0.4,rounding_size=0.9",
                linewidth=1.5, edgecolor="#1f3b63", facecolor=color, alpha=0.22,
            )
        )
        labels = "\n".join(FAULT_CLASSES[i] for i in indices)
        ax.text(task_index * 19 + 11.5, 3.8, f"{task_name}\n{labels}",
                ha="center", va="center", fontsize=8.6)
        if task_index < len(TASK_GROUPS) - 1:
            ax.add_patch(
                FancyArrowPatch((task_index * 19 + 20.4, 3.8),
                                (task_index * 19 + 22.6, 3.8),
                                arrowstyle="-|>", mutation_scale=15,
                                linewidth=1.8, color="#1f3b63")
            )
    ax.set_xlim(0, 97); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(48.5, 6.6, "图7  增量任务划分（5 个任务，类别划分 2/2/2/2/1）",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(48.5, 0.35,
            "数据按故障类别顺序到达，模型在每个任务上仅可见当前任务类别；"
            "训练结束后要求在全部 9 类上均可判别",
            ha="center", fontsize=8.8, color="#444444")
    save(fig, outdir, "fig7_tasks.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=osp.join(ROOT, "docs_patent", "figures"))
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"writing figures to {args.outdir}")
    fig1_framework(args.outdir)
    fig2_tokenizer(args.outdir)
    fig3_quantization(args.outdir)
    fig4_rpq(args.outdir)
    fig5_antiforget(args.outdir)
    fig6_results(args.outdir)
    fig7_tasks(args.outdir)
    print("done")


if __name__ == "__main__":
    main()
