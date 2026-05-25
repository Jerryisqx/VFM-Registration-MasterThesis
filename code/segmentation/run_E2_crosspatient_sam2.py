"""
E2 cross-patient eval (SAM2 family) -- runs SAM2-Base+ zero-shot and
MedSAM2-Tiny 5-shot fine-tune on the 2 cross-patient X-rays for which
clinician-validated reference masks exist
(\`code/gui/04_samples/\*_label_map.png\` -- produced by the SAM3
GUI workflow with manual point prompts; treat as expert-validated
reference, not pixel-perfect GT).

Same prompt strategy as E1: centroid of reference mask, single
positive point per bone.

Run:
    PYTHONPATH=/Users/jerrychen/Desktop/MasterThesis/code/sam2_lib \\
        /Users/jerrychen/opt/anaconda3/envs/sam2/bin/python \\
        code/segmentation/run_E2_crosspatient_sam2.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

from hydra import initialize_config_module
try:
    initialize_config_module("sam2", version_base="1.2")
except Exception:
    pass

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


PROJECT_ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
TEST_DIR = PROJECT_ROOT / "data/data/test"
GUI_SAMPLES = PROJECT_ROOT / "code/gui/04_samples"
OUT_DIR = PROJECT_ROOT / "results/E2_crosspatient"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Patients with reference masks
TEST_CASES = [
    "C_SUBN_11_gt_f_02_029",
    "C_SUBO_01_gt_f_05_042",
]

METHODS = [
    {
        "name": "sam2_basep_zeroshot",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "ckpt": str(PROJECT_ROOT / "checkpoints/sam2_official/sam2.1_b.pt"),
    },
    {
        "name": "medsam2_tiny_5shot",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "ckpt": str(PROJECT_ROOT / "checkpoints/medsam2_finetune/checkpoint_robust_v1.pt"),
    },
]


def load_image_rgb(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] >= 3 else arr.squeeze()
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
    arr = arr.astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def dice(p, g):
    i = (p & g).sum(); t = p.sum() + g.sum()
    return float(2.0 * i / t) if t > 0 else 0.0


def iou(p, g):
    i = (p & g).sum(); u = (p | g).sum()
    return float(i / u) if u > 0 else 0.0


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    rows = []
    for m in METHODS:
        print(f"\n=== {m['name']} ===")
        t0 = time.time()
        model = build_sam2(m["config"], m["ckpt"], device=device)
        predictor = SAM2ImagePredictor(model)
        print(f"  loaded in {time.time()-t0:.1f}s")
        for stem in TEST_CASES:
            img_p = TEST_DIR / f"{stem}.tif"
            lm_p = GUI_SAMPLES / f"{stem}_label_map.png"
            if not (img_p.exists() and lm_p.exists()):
                continue
            lm = np.array(Image.open(lm_p))
            gt_fem = lm == 1
            gt_tib = lm == 2

            rgb = load_image_rgb(img_p)
            predictor.set_image(rgb)
            row = {"image": stem, "method": m["name"]}
            for cls, gt in [("femur", gt_fem), ("tibia", gt_tib)]:
                c = mask_centroid(gt)
                if c is None:
                    row[f"{cls}_dice"] = 0.0; row[f"{cls}_iou"] = 0.0
                    continue
                cx, cy = c
                masks, _, _ = predictor.predict(
                    point_coords=np.array([[cx, cy]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=False,
                )
                pred = masks[0].astype(bool)
                row[f"{cls}_dice"] = dice(pred, gt)
                row[f"{cls}_iou"] = iou(pred, gt)
                row[f"{cls}_pred_px"] = int(pred.sum())
                row[f"{cls}_gt_px"] = int(gt.sum())
                # save predicted mask for qualitative figure
                pred_path = OUT_DIR / f"{stem}_{m['name']}_{cls}_pred.png"
                Image.fromarray((pred.astype(np.uint8) * 255)).save(pred_path)
            print(f"  {stem}: femur Dice={row.get('femur_dice', 0):.3f} tibia Dice={row.get('tibia_dice', 0):.3f}")
            rows.append(row)

    csv_path = OUT_DIR / "crosspatient_sam2family.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
