"""§8.7 figure: Dice vs interactive click count for SAM2-Base+ ZS vs
MedSAM2-Tiny 5-shot."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
DATA = ROOT / "results/E3_gui_protocol/dice_per_image_per_click.csv"
OUT = ROOT / "results/E3_gui_protocol/dice_vs_clicks.png"

METHODS = [
    ("sam2_basep_zeroshot", "SAM2-Base+ ZS",      "#cc4444"),
    ("sam3_zeroshot",       "SAM3 ZS",            "#22aa66"),
    ("medsam2_tiny_5shot",  "MedSAM2 5-shot",     "#3370b8"),
]
N_CLICKS = 5


def aggregate(rows, method, klass):
    sel = [r for r in rows if r["method"] == method]
    arr = np.array([[float(r[f"{klass}_dice_click{k+1}"]) for k in range(N_CLICKS)] for r in sel])
    mean = arr.mean(axis=0)
    fail_rate = (arr < 0.5).mean(axis=0)
    sat_rate = (arr >= 0.8).mean(axis=0)
    return mean, fail_rate, sat_rate, arr


def main():
    rows = list(csv.DictReader(open(DATA)))
    xs = np.arange(1, N_CLICKS + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # Top row: mean Dice ± IQR
    for col, klass in enumerate(["femur", "tibia"]):
        ax = axes[0, col]
        for mkey, mlabel, color in METHODS:
            mean, fail, sat, arr = aggregate(rows, mkey, klass)
            p25 = np.percentile(arr, 25, axis=0)
            p75 = np.percentile(arr, 75, axis=0)
            ax.fill_between(xs, p25, p75, alpha=0.2, color=color)
            ax.plot(xs, mean, "o-", color=color, linewidth=2, label=mlabel)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(xs)
        ax.set_xlabel("Number of point prompts")
        ax.set_ylabel(f"{klass.capitalize()} Dice")
        ax.set_title(f"{klass.capitalize()} - mean Dice vs click count (band = IQR)")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(loc="lower right")

    # Bottom row: failure rate (Dice < 0.5)
    for col, klass in enumerate(["femur", "tibia"]):
        ax = axes[1, col]
        for mkey, mlabel, color in METHODS:
            mean, fail, sat, arr = aggregate(rows, mkey, klass)
            ax.plot(xs, fail * 100, "o-", color=color, linewidth=2, label=mlabel)
        ax.set_ylim(-2, 60)
        ax.set_xticks(xs)
        ax.set_xlabel("Number of point prompts")
        ax.set_ylabel("Failure rate Dice < 0.5 (%)")
        ax.set_title(f"{klass.capitalize()} - failure rate vs click count")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(loc="upper right")

    fig.suptitle(
        "E3: Dice and failure rate vs interactive prompt count (n=77 in-distribution frames)\n"
        "Optimal-clinician protocol: click 1 at centroid, then click at largest error region",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
