"""Compose the 3 real GUI screenshots into a single 3-panel figure for §6.7.

Steps captured:
  (a) Initial state -- image loaded, no prompts.
  (b) After 1 positive click on femur -- SAM3 produces the femur mask
      (score 0.765 visible in status bar).
  (c) After switching to tibia and adding 2 positive clicks -- both
      femur and tibia masks are present.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
SRC = ROOT / "results/gui_screenshots"
OUT = ROOT / "results/figure_gui_workflow.png"

PANELS = [
    (SRC / "gui_step1_initial.png",
     "(a) Image loaded, no prompts yet."),
    (SRC / "gui_step2_femur.png",
     "(b) One positive click on the femur (green dot, centre-left of image) "
     "yields the SAM3 femur mask (purple, score 0.77)."),
    (SRC / "gui_step3_both.png",
     "(c) Switching to the tibia class and placing two positive clicks "
     "yields the second mask (orange); both bones are now segmented."),
]


def main():
    images = [Image.open(p) for p, _ in PANELS]
    fig, axes = plt.subplots(len(images), 1, figsize=(11, 4.6 * len(images)))
    for ax, img, (_, cap) in zip(axes, images, PANELS):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(cap, fontsize=10, loc="left", pad=6)

    fig.suptitle("GUI workflow: progressive interactive segmentation of femur and tibia",
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
