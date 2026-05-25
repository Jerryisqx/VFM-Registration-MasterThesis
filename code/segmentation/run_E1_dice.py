"""
E1 -- Compute Dice / IoU of the deployed 5-shot MedSAM2 fine-tune
predictions against the DRR-derived ground-truth bone silhouettes.

This uses already-computed predictions in `results/seg_finetune_clinical/
mask_{femur,tibia}_pred/` (one binary PNG per clinical X-ray) and the
soft GT silhouettes in `data/data/mask_{femur,tibia}/` (binarised at
intensity > 30).

Image-to-mask correspondence is by frame index:
    C_SUBN_02_dkb_01_001.tif  <->  kneefit_*_0_syn.png
    C_SUBN_02_dkb_01_002.tif  <->  kneefit_*_1_syn.png
    ...
    C_SUBN_02_dkb_01_077.tif  <->  kneefit_*_76_syn.png

Outputs:
    results/E1_segmentation/dice_per_image.csv
    results/E1_segmentation/summary.txt
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
GT_FEM_DIR = PROJECT_ROOT / "data/data/mask_femur"
GT_TIB_DIR = PROJECT_ROOT / "data/data/mask_tibia"
PRED_FEM_DIR = PROJECT_ROOT / "results/seg_finetune_clinical/mask_femur_pred"
PRED_TIB_DIR = PROJECT_ROOT / "results/seg_finetune_clinical/mask_tibia_pred"
OUT_DIR = PROJECT_ROOT / "results/E1_segmentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_THRESH = 30  # binarisation cutoff for the soft DRR density GT masks


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    return float(2.0 * inter / total) if total > 0 else 0.0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    pred_files = sorted(PRED_FEM_DIR.glob("C_SUBN_02_dkb_01_*.png"))
    if not pred_files:
        raise SystemExit("No prediction files found")

    rows = []
    for pred_f in pred_files:
        # Extract index from file name e.g. C_SUBN_02_dkb_01_001 -> 1 -> GT idx 0
        m = re.search(r"_(\d+)\.png$", pred_f.name)
        if not m:
            continue
        clinical_idx = int(m.group(1))  # 001-077
        gt_idx = clinical_idx - 1       # 0-76

        gt_fem_f = GT_FEM_DIR / f"kneefit_femur_{gt_idx}_syn.png"
        gt_tib_f = GT_TIB_DIR / f"kneefit_tibia_{gt_idx}_syn.png"
        pred_fem_f = pred_f
        pred_tib_f = PRED_TIB_DIR / pred_f.name

        if not (gt_fem_f.exists() and gt_tib_f.exists() and pred_tib_f.exists()):
            print(f"Missing pair for idx={clinical_idx}, skipping")
            continue

        gt_fem = np.array(Image.open(gt_fem_f)) > GT_THRESH
        gt_tib = np.array(Image.open(gt_tib_f)) > GT_THRESH
        pred_fem = np.array(Image.open(pred_fem_f)) > 0
        pred_tib = np.array(Image.open(pred_tib_f)) > 0

        d_fem = dice(pred_fem, gt_fem)
        d_tib = dice(pred_tib, gt_tib)
        i_fem = iou(pred_fem, gt_fem)
        i_tib = iou(pred_tib, gt_tib)

        rows.append({
            "clinical_idx": clinical_idx,
            "image_name": pred_f.stem + ".tif",
            "femur_dice": d_fem,
            "femur_iou": i_fem,
            "femur_gt_area": int(gt_fem.sum()),
            "femur_pred_area": int(pred_fem.sum()),
            "tibia_dice": d_tib,
            "tibia_iou": i_tib,
            "tibia_gt_area": int(gt_tib.sum()),
            "tibia_pred_area": int(pred_tib.sum()),
        })

    if not rows:
        raise SystemExit("No rows produced")

    csv_path = OUT_DIR / "dice_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    fem_d = np.array([r["femur_dice"] for r in rows])
    tib_d = np.array([r["tibia_dice"] for r in rows])
    fem_i = np.array([r["femur_iou"] for r in rows])
    tib_i = np.array([r["tibia_iou"] for r in rows])

    def stats(arr):
        return (float(arr.mean()), float(np.median(arr)),
                float(arr.min()), float(arr.max()), float(arr.std()))

    print()
    print("=== E1 Summary: 5-shot MedSAM2 vs DRR GT, n=%d ===" % len(rows))
    print(f"{'metric':<14} {'mean':>7} {'median':>7} {'min':>7} {'max':>7} {'std':>7}")
    for name, arr in [("femur Dice", fem_d), ("femur IoU", fem_i),
                       ("tibia Dice", tib_d), ("tibia IoU", tib_i)]:
        m, med, lo, hi, std = stats(arr)
        print(f"{name:<14} {m:>7.3f} {med:>7.3f} {lo:>7.3f} {hi:>7.3f} {std:>7.3f}")
    print()
    print(f"femur Dice >= 0.9: {(fem_d >= 0.9).sum()} / {len(fem_d)}")
    print(f"femur Dice >= 0.8: {(fem_d >= 0.8).sum()} / {len(fem_d)}")
    print(f"femur Dice <  0.5: {(fem_d <  0.5).sum()} / {len(fem_d)}")
    print(f"tibia Dice >= 0.9: {(tib_d >= 0.9).sum()} / {len(tib_d)}")
    print(f"tibia Dice >= 0.8: {(tib_d >= 0.8).sum()} / {len(tib_d)}")
    print(f"tibia Dice <  0.5: {(tib_d <  0.5).sum()} / {len(tib_d)}")

    summary = OUT_DIR / "summary.txt"
    with open(summary, "w") as f:
        f.write(f"E1 -- 5-shot MedSAM2 vs DRR GT, n={len(rows)} clinical X-rays\n")
        f.write(f"GT binarisation threshold: > {GT_THRESH} (0-255 soft DRR density)\n\n")
        f.write(f"{'metric':<14} {'mean':>7} {'median':>7} {'min':>7} {'max':>7} {'std':>7}\n")
        for name, arr in [("femur Dice", fem_d), ("femur IoU", fem_i),
                           ("tibia Dice", tib_d), ("tibia IoU", tib_i)]:
            m, med, lo, hi, std = stats(arr)
            f.write(f"{name:<14} {m:>7.3f} {med:>7.3f} {lo:>7.3f} {hi:>7.3f} {std:>7.3f}\n")
        f.write("\n")
        f.write(f"femur Dice >= 0.9: {(fem_d >= 0.9).sum()} / {len(fem_d)}\n")
        f.write(f"femur Dice >= 0.8: {(fem_d >= 0.8).sum()} / {len(fem_d)}\n")
        f.write(f"femur Dice <  0.5: {(fem_d <  0.5).sum()} / {len(fem_d)}\n")
        f.write(f"tibia Dice >= 0.9: {(tib_d >= 0.9).sum()} / {len(tib_d)}\n")
        f.write(f"tibia Dice >= 0.8: {(tib_d >= 0.8).sum()} / {len(tib_d)}\n")
        f.write(f"tibia Dice <  0.5: {(tib_d <  0.5).sum()} / {len(tib_d)}\n")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
