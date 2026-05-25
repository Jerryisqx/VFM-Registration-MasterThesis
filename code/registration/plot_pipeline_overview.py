"""§6.2 figure: end-to-end pipeline overview.

Five blocks left-to-right:
  (a) Inputs:    pre-op CT volume V + intra-op X-ray pair (I_a*, I_b*) + P_a, P_b
  (b) Segmenter: VFM (SAM2 / SAM3 / 5-shot MedSAM2) -> masks M_a, M_b
  (c) Renderer:  dupla_renderers DRR(V, theta, P) for both planes -> I_hat_a, I_hat_b
  (d) Loss:      masked dual-plane similarity L_reg
  (e) Optimiser: Adam update on theta with feedback loop back to (c)

Output: theta^*
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
OUT = ROOT / "results/figure_pipeline_overview.png"


def box(ax, x, y, w, h, title, lines, face="#f0f4f8", edge="#3370b8"):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.7, edgecolor=edge, facecolor=face)
    ax.add_patch(p)
    # Title near the top of the box
    ax.text(x + w / 2, y + h - 0.25, title,
             ha="center", va="top", fontsize=11, weight="bold", color="#1d3a55")
    # Body lines, evenly spaced below the title
    n = len(lines)
    body_top = y + h - 0.65
    body_bottom = y + 0.30
    span = body_top - body_bottom
    step = span / max(n - 1, 1)
    for i, line in enumerate(lines):
        ty = body_top - i * step
        ax.text(x + w / 2, ty, line,
                 ha="center", va="center", fontsize=9.5, color="#1d3a55")


def arrow(ax, x1, y1, x2, y2, color="#444", lw=1.6, label=None, label_y_off=0.04, style="->"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=18,
                        color=color, lw=lw)
    ax.add_patch(a)
    if label is not None:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + label_y_off
        ax.text(mx, my, label, ha="center", va="bottom",
                 fontsize=9, color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 5.5); ax.axis("off")

    bw, bh = 2.55, 1.85  # block width, height
    gap = 0.35           # horizontal spacing
    y0 = 2.25            # bottom of blocks

    xs = [0.20 + i * (bw + gap) for i in range(5)]

    # (a) Inputs
    box(ax, xs[0], y0, bw, bh,
        "(a) Inputs",
        [r"pre-op CT $V$",
         r"X-rays $I_a^\star, I_b^\star$",
         r"calibration $P_a, P_b$"],
        face="#fff6e8", edge="#cc8a3a")

    # (b) VFM segmenter
    box(ax, xs[1], y0, bw, bh,
        "(b) VFM segmenter",
        [r"SAM2 / SAM3 / MedSAM2",
         r"point or text prompts",
         r"$\rightarrow M_a, M_b$"],
        face="#eef5ee", edge="#5a8a4f")

    # (c) Differentiable renderer
    box(ax, xs[2], y0, bw, bh,
        "(c) Diff.\\ renderer",
        [r"dupla\_renderers",
         r"$\mathrm{DRR}(V, \theta, P)$",
         r"$\rightarrow \hat I_a, \hat I_b$"],
        face="#eef2fb", edge="#3370b8")

    # (d) Loss
    box(ax, xs[3], y0, bw, bh,
        "(d) Loss",
        [r"masked dual-plane",
         r"$\mathcal{L}_\mathrm{reg}(\theta)$",
         r"MSE / NCC / GCC"],
        face="#fbeef2", edge="#b8336a")

    # (e) Optimiser
    box(ax, xs[4], y0, bw, bh,
        "(e) Optimiser",
        [r"Adam, $\eta = 1.0$",
         r"rel.\ loss $<10^{-5}$",
         r"converges $\rightarrow \theta^\star$"],
        face="#f4eefb", edge="#7a4cb8")

    # Forward arrows between blocks
    for i in range(4):
        arrow(ax, xs[i] + bw, y0 + bh / 2, xs[i + 1], y0 + bh / 2)

    # Feedback loop: optimiser back to renderer (drop down, left, up)
    feedback_y = y0 - 0.95
    arrow(ax, xs[4] + bw / 2, y0, xs[4] + bw / 2, feedback_y, lw=1.5, color="#7a4cb8", style="-")
    arrow(ax, xs[4] + bw / 2, feedback_y, xs[2] + bw / 2, feedback_y, lw=1.5, color="#7a4cb8", style="-")
    arrow(ax, xs[2] + bw / 2, feedback_y, xs[2] + bw / 2, y0, lw=1.5, color="#7a4cb8")
    ax.text((xs[2] + xs[4] + bw) / 2, feedback_y - 0.20,
             r"updated pose $\theta$ (one optimisation step)",
             ha="center", va="top", fontsize=10, color="#7a4cb8", style="italic")

    # Final output arrow on the right
    arrow(ax, xs[4] + bw, y0 + bh / 2, xs[4] + bw + 0.55, y0 + bh / 2, lw=1.8)
    ax.text(xs[4] + bw + 0.65, y0 + bh / 2 + 0.10,
             r"$\theta^\star$",
             ha="left", va="bottom", fontsize=14, color="#1d3a55", weight="bold")
    ax.text(xs[4] + bw + 0.65, y0 + bh / 2 - 0.18,
             r"rigid pose",
             ha="left", va="top", fontsize=9, color="#666")

    # Top annotation
    ax.text(7.5, 5.05, "End-to-end registration pipeline",
             ha="center", va="center", fontsize=14, weight="bold", color="#1d3a55")
    ax.text(7.5, 4.65,
             "Solid arrows: forward pass.  Purple loop: gradient-based Adam updates re-render at each step until convergence.",
             ha="center", va="center", fontsize=10, color="#555", style="italic")

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
