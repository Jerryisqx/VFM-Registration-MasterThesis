"""Render a labelled mockup of the segmentation GUI used in the thesis.

The real GUI (code/gui/03_tool/test_sam2.py) is an interactive OpenCV
window that requires a display. This script renders an equivalent
annotated diagram of the GUI layout using matplotlib so the thesis
can include a figure without needing a live screenshot.

Layout follows the actual GUI:
  - Left panel: image canvas with predicted mask overlay + sample
    positive/negative point prompts.
  - Right panel: schematic representation of the control panel
    (active class, prompt mode, threshold/smoothing/dilation sliders,
    enhancement toggles, status bar, shortcuts).
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image


ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
OUT = ROOT / "results/figure_gui_mockup.png"


def load_xray(idx=29):
    p = ROOT / f"data/data/images/C_SUBN_02_dkb_01_{idx:03d}.tif"
    a = tifffile.imread(str(p)).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] >= 3 else a.squeeze()
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def load_masks(idx=29):
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


def draw_left_canvas(ax, img_rgb):
    ax.imshow(img_rgb)
    H, W = img_rgb.shape[:2]
    # Sample prompt points
    points = [
        (W * 0.45, H * 0.35, "+", "lime"),   # positive for femur
        (W * 0.55, H * 0.30, "+", "lime"),
        (W * 0.50, H * 0.65, "+", "lime"),   # positive for tibia (placed lower)
        (W * 0.75, H * 0.55, "-", "red"),    # negative
    ]
    for x, y, sign, color in points:
        ax.plot(x, y, "o", markersize=10, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.text(x + W * 0.02, y - H * 0.012, sign, color=color, fontsize=14, weight="bold")

    ax.set_title("Image canvas (left)\n2-finger zoom, drag to pan, left/right click = positive/negative prompt",
                  fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def draw_right_panel(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    # Panel background
    ax.add_patch(mpatches.Rectangle((0.0, 0.0), 1.0, 1.0,
                                     facecolor="#f4f4f6", edgecolor="#888", linewidth=1))

    # Top: status bar
    ax.add_patch(mpatches.Rectangle((0.03, 0.92), 0.94, 0.06,
                                     facecolor="white", edgecolor="#bbb"))
    ax.text(0.50, 0.95, "[image 029 / 077]   C_SUBN_02_dkb_01_029.tif",
             ha="center", va="center", fontsize=8, family="monospace")

    # Class toggle
    ax.text(0.04, 0.86, "Active class:", fontsize=9, weight="bold")
    ax.add_patch(mpatches.Rectangle((0.04, 0.80), 0.30, 0.045,
                                     facecolor="#ff8888", edgecolor="#cc4444"))
    ax.text(0.19, 0.823, "1: femur", ha="center", va="center", fontsize=9)
    ax.add_patch(mpatches.Rectangle((0.36, 0.80), 0.30, 0.045,
                                     facecolor="white", edgecolor="#bbb"))
    ax.text(0.51, 0.823, "2: tibia", ha="center", va="center", fontsize=9, color="#555")

    # Prompt mode
    ax.text(0.04, 0.755, "Prompt mode:", fontsize=9, weight="bold")
    ax.add_patch(mpatches.Rectangle((0.04, 0.70), 0.30, 0.04,
                                     facecolor="#88c8ff", edgecolor="#3370b8"))
    ax.text(0.19, 0.72, "SAM points", ha="center", va="center", fontsize=8)
    ax.add_patch(mpatches.Rectangle((0.36, 0.70), 0.30, 0.04,
                                     facecolor="white", edgecolor="#bbb"))
    ax.text(0.51, 0.72, "Polygon (M)", ha="center", va="center", fontsize=8, color="#555")

    # Sliders
    sliders = [
        ("Threshold [tau]",   0.62, 0.42),
        ("Smoothing kernel",  0.55, 0.30),
        ("Mask dilation",     0.48, 0.10),
    ]
    for label, y, val_frac in sliders:
        ax.text(0.04, y + 0.018, label, fontsize=9)
        ax.add_patch(mpatches.Rectangle((0.04, y), 0.92, 0.015,
                                         facecolor="#dddddd", edgecolor="#aaaaaa"))
        ax.add_patch(mpatches.Rectangle((0.04, y), 0.92 * val_frac, 0.015,
                                         facecolor="#3370b8", edgecolor="#3370b8"))
        ax.plot(0.04 + 0.92 * val_frac, y + 0.0075, "o", markersize=7,
                 markerfacecolor="#3370b8", markeredgecolor="white")

    # Preprocessing buttons row
    ax.text(0.04, 0.40, "Preprocessing:", fontsize=9, weight="bold")
    for i, name in enumerate(["raw", "denoise", "CLAHE", "blur", "all"]):
        x0 = 0.04 + i * 0.19
        color = "#88c8ff" if i == 2 else "white"
        edge = "#3370b8" if i == 2 else "#bbb"
        ax.add_patch(mpatches.Rectangle((x0, 0.34), 0.17, 0.04,
                                         facecolor=color, edgecolor=edge))
        ax.text(x0 + 0.085, 0.36, name, ha="center", va="center", fontsize=8)

    # Anti-class-bleed toggle
    ax.add_patch(mpatches.Rectangle((0.04, 0.245), 0.45, 0.04,
                                     facecolor="#88c8ff", edgecolor="#3370b8"))
    ax.text(0.265, 0.265, "anti-bleed (T)  [ON]", ha="center", va="center", fontsize=8)
    ax.add_patch(mpatches.Rectangle((0.51, 0.245), 0.45, 0.04,
                                     facecolor="white", edgecolor="#bbb"))
    ax.text(0.735, 0.265, "auto-neg (G)  [OFF]", ha="center", va="center", fontsize=8, color="#555")

    # Action buttons
    actions = ["Save (S)", "Undo (Z)", "Reset (R)", "Clear (C)", "Eraser (B)"]
    for i, name in enumerate(actions):
        x0 = 0.04 + i * 0.19
        ax.add_patch(mpatches.Rectangle((x0, 0.18), 0.17, 0.04,
                                         facecolor="white", edgecolor="#bbb"))
        ax.text(x0 + 0.085, 0.20, name, ha="center", va="center", fontsize=8)

    # Shortcut help block
    ax.add_patch(mpatches.Rectangle((0.03, 0.01), 0.94, 0.08,
                                     facecolor="#fffaf0", edgecolor="#bbb"))
    ax.text(0.50, 0.05,
             "A/D prev/next  |  1/2 class  |  M mode  |  Q/E smooth  |  [ / ] threshold",
             ha="center", va="center", fontsize=8, family="monospace")

    ax.set_title("Control panel (right)\nclass toggle | prompt mode | sliders | preprocessing | actions",
                  fontsize=10)


def main():
    img = load_xray(29)
    fem, tib = load_masks(29)
    img_rgb = overlay(img, fem, tib)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                              gridspec_kw=dict(width_ratios=[1.5, 1.0], wspace=0.05))
    draw_left_canvas(axes[0], img_rgb)
    draw_right_panel(axes[1])
    fig.suptitle("Segmentation GUI mockup -- annotated layout of the OpenCV interactive tool\n"
                  "(red dot = positive prompt, blue dot = negative prompt; class-, prompt-, "
                  "and post-processing controls on the right)",
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
