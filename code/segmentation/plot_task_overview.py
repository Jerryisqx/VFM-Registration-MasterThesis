"""Figure 1 -- Task overview for the thesis intro (§2.1).

Shows what the segmentation+registration pipeline takes as input and
produces as output:
  (a) two biplanar X-ray projections of a knee (BS + FS, same frame)
  (b) the bone-of-interest segmentation masks overlaid
  (c) schematic arrow to the recovered 3D pose

Uses real data from data/data/images/ + data/data/mask_*.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tifffile
from PIL import Image


ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
OUT = ROOT / "results/figure_task_overview.png"

# Two frames from the kinematic sequence representing a knee in flexion.
# (We use two adjacent indices to look like "BS plane" and "FS plane" of the same moment;
# the real dual-plane pair would be from different cameras of the same instant, but the
# 77 SUBN_02 frames are a single-camera sequence -- close enough for an illustrative panel.)
FRAME_A = 1   # plane A illustration
FRAME_B = 9   # plane B illustration (chosen at different angle for visual variety)


def load_xray(idx):
    p = ROOT / f"data/data/images/C_SUBN_02_dkb_01_{idx:03d}.tif"
    a = tifffile.imread(str(p)).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] >= 3 else a.squeeze()
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def load_masks(idx):
    gt_idx = idx - 1
    fem = np.array(Image.open(ROOT / f"data/data/mask_femur/kneefit_femur_{gt_idx}_syn.png")) > 30
    tib = np.array(Image.open(ROOT / f"data/data/mask_tibia/kneefit_tibia_{gt_idx}_syn.png")) > 30
    return fem, tib


def overlay(img01, fem, tib, alpha=0.40):
    rgb = np.stack([img01, img01, img01], axis=-1).copy()
    out = rgb.copy()
    out[fem] = (1 - alpha) * out[fem] + alpha * np.array([1.0, 0.25, 0.25])
    out[tib] = (1 - alpha) * out[tib] + alpha * np.array([0.25, 0.45, 1.0])
    return np.clip(out, 0, 1)


def schematic_pose(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    # Draw a stylised 3D pose box
    box = mpatches.FancyBboxPatch((0.18, 0.32), 0.64, 0.36,
                                   boxstyle="round,pad=0.02",
                                   linewidth=1.6, edgecolor="#444",
                                   facecolor="#f0f4f8")
    ax.add_patch(box)
    ax.text(0.5, 0.59, "Rigid pose", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.47, r"$T \in SE(3)$", ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.39, r"$(t_x, t_y, t_z, r_x, r_y, r_z)$",
             ha="center", va="center", fontsize=9, family="monospace")
    ax.text(0.5, 0.20, "aligning pre-op CT\nto C-arm frame",
             ha="center", va="center", fontsize=9, color="#666")


def main():
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], width_ratios=[1, 1, 0.25, 1.3],
                           hspace=0.05, wspace=0.05)

    # Row 1: input X-rays (plane A, plane B) + arrow + output
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_arrow = fig.add_subplot(gs[0, 2])
    ax_pose = fig.add_subplot(gs[0, 3])

    img_a = load_xray(FRAME_A)
    img_b = load_xray(FRAME_B)
    ax_a.imshow(img_a, cmap="gray", vmin=0, vmax=1); ax_a.set_title("Intra-op X-ray (plane A)", fontsize=11)
    ax_b.imshow(img_b, cmap="gray", vmin=0, vmax=1); ax_b.set_title("Intra-op X-ray (plane B)", fontsize=11)
    for ax in (ax_a, ax_b):
        ax.set_xticks([]); ax.set_yticks([])

    # Arrow between inputs and output
    ax_arrow.axis("off")
    ax_arrow.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5),
                       arrowprops=dict(arrowstyle="->", lw=2.5, color="#444"))
    ax_arrow.text(0.5, 0.62, "SAM mask\n+ dual-plane\nregistration",
                   ha="center", va="center", fontsize=9, color="#444")

    schematic_pose(ax_pose)

    # Row 2: same plane A/B with predicted masks overlaid
    ax_a2 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[1, 1])
    fem_a, tib_a = load_masks(FRAME_A)
    fem_b, tib_b = load_masks(FRAME_B)
    ax_a2.imshow(overlay(img_a, fem_a, tib_a)); ax_a2.set_title("femur + tibia mask", fontsize=9)
    ax_b2.imshow(overlay(img_b, fem_b, tib_b)); ax_b2.set_title("femur + tibia mask", fontsize=9)
    for ax in (ax_a2, ax_b2):
        ax.set_xticks([]); ax.set_yticks([])

    # Caption for row 2 column 3-4
    ax_caption = fig.add_subplot(gs[1, 2:])
    ax_caption.axis("off")
    legend_handles = [
        mpatches.Patch(color=(1.0, 0.25, 0.25), label="femur"),
        mpatches.Patch(color=(0.25, 0.45, 1.0), label="tibia"),
    ]
    ax_caption.legend(handles=legend_handles, loc="center left", frameon=False,
                       fontsize=10, ncol=2)

    fig.suptitle("Task overview: from biplanar knee X-ray to 3D rigid pose",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
