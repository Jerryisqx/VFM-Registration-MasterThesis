"""§8.1 qualitative figure: cross-patient mask predictions for 2 hold-out
patients across the 5-method benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

PROJECT_ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
TEST_DIR = PROJECT_ROOT / "data/data/test"
GUI_SAMPLES = PROJECT_ROOT / "code/gui/04_samples"
PRED_DIR = PROJECT_ROOT / "results/E2_crosspatient"
OUT = PROJECT_ROOT / "results/E2_crosspatient/qualitative_grid.png"

CASES = ["C_SUBN_11_gt_f_02_029", "C_SUBO_01_gt_f_05_042"]
METHODS = [
    ("sam2_basep_zeroshot", "SAM2-Base+ ZS"),
    ("sam3_zeroshot",       "SAM3 ZS"),
    ("medsam2_tiny_5shot",  "MedSAM2 5-shot"),
    ("unet_5shot",          "U-Net 5-shot"),
]


def load_input_normalised(p):
    a = tifffile.imread(str(p)).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] >= 3 else a.squeeze()
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo), 0, 1) if hi > lo else a


def overlay(img01, fem_bin, tib_bin):
    rgb = np.stack([img01, img01, img01], axis=-1)
    fem_overlay = rgb.copy()
    fem_overlay[fem_bin] = [1.0, 0.2, 0.2]
    tib_overlay = rgb.copy()
    tib_overlay[tib_bin] = [0.2, 0.4, 1.0]
    out = 0.55 * rgb + 0.45 * (0.5 * fem_overlay + 0.5 * tib_overlay)
    out[fem_bin & tib_bin] = [0.9, 0.0, 0.9]
    return np.clip(out, 0, 1)


def load_reference(stem):
    lm = np.array(Image.open(GUI_SAMPLES / f"{stem}_label_map.png"))
    return lm == 1, lm == 2


def load_pred_pair(stem, method_key):
    fp = PRED_DIR / f"{stem}_{method_key}_femur_pred.png"
    tp = PRED_DIR / f"{stem}_{method_key}_tibia_pred.png"
    fem = (np.array(Image.open(fp)) > 127) if fp.exists() else None
    tib = (np.array(Image.open(tp)) > 127) if tp.exists() else None
    return fem, tib


def main():
    n_rows = len(CASES)
    n_cols = 2 + len(METHODS)  # input + reference + methods

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for r, stem in enumerate(CASES):
        img01 = load_input_normalised(TEST_DIR / f"{stem}.tif")
        ref_fem, ref_tib = load_reference(stem)

        axes[r, 0].imshow(img01, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("input" if r == 0 else "", fontsize=10)
        axes[r, 0].set_ylabel(stem.replace("_gt", "")[:14], fontsize=9)

        axes[r, 1].imshow(overlay(img01, ref_fem, ref_tib))
        axes[r, 1].set_title("reference" if r == 0 else "", fontsize=10)

        for k, (mk, ml) in enumerate(METHODS):
            ax = axes[r, 2 + k]
            fem, tib = load_pred_pair(stem, mk)
            if fem is None or tib is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            else:
                if fem.shape != img01.shape:
                    fem = np.array(Image.fromarray((fem * 255).astype(np.uint8)).resize(
                        (img01.shape[1], img01.shape[0]), Image.NEAREST)) > 127
                    tib = np.array(Image.fromarray((tib * 255).astype(np.uint8)).resize(
                        (img01.shape[1], img01.shape[0]), Image.NEAREST)) > 127
                ax.imshow(overlay(img01, fem, tib))
            ax.set_title(ml if r == 0 else "", fontsize=10)

        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Cross-patient bone segmentation: 4 method comparison\n(red=femur, blue=tibia, magenta=overlap)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
