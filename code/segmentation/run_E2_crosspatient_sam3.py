"""E2 cross-patient eval (SAM3 zero-shot)."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image

from sam3 import build_sam3_image_model

PROJECT_ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
SAM3_TOOL = PROJECT_ROOT / "code/gui/03_tool"
TEST_DIR = PROJECT_ROOT / "data/data/test"
GUI_SAMPLES = PROJECT_ROOT / "code/gui/04_samples"
OUT_DIR = PROJECT_ROOT / "results/E2_crosspatient"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = str(SAM3_TOOL / "weights" / "sam3.pt")
BPE = str(SAM3_TOOL / "sam3_code" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")

TEST_CASES = ["C_SUBN_11_gt_f_02_029", "C_SUBO_01_gt_f_05_042"]


def load_rgb(p):
    a = tifffile.imread(str(p)).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] >= 3 else a.squeeze()
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    if hi > lo:
        a = np.clip((a - lo) / (hi - lo), 0, 1) * 255
    a = a.astype(np.uint8)
    return np.stack([a, a, a], axis=-1)


def centroid(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0: return None
    return int(xs.mean()), int(ys.mean())


def dice(p, g):
    i = (p & g).sum(); t = p.sum() + g.sum()
    return float(2.0 * i / t) if t > 0 else 0.0


def iou(p, g):
    i = (p & g).sum(); u = (p | g).sum()
    return float(i / u) if u > 0 else 0.0


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    model = build_sam3_image_model(
        checkpoint_path=CKPT, bpe_path=BPE, device=device,
        eval_mode=True, load_from_HF=False,
        enable_segmentation=True, enable_inst_interactivity=True,
    )
    predictor = model.inst_interactive_predictor
    if getattr(predictor.model, "backbone", None) is None:
        predictor.model.backbone = model.backbone
    print(f"SAM3 loaded in {time.time()-t0:.1f}s")

    rows = []
    for stem in TEST_CASES:
        img_p = TEST_DIR / f"{stem}.tif"
        lm_p = GUI_SAMPLES / f"{stem}_label_map.png"
        if not (img_p.exists() and lm_p.exists()): continue
        lm = np.array(Image.open(lm_p))
        gt_fem = lm == 1
        gt_tib = lm == 2

        rgb = load_rgb(img_p)
        predictor.set_image(rgb)
        row = {"image": stem, "method": "sam3_zeroshot"}
        for cls, gt in [("femur", gt_fem), ("tibia", gt_tib)]:
            c = centroid(gt)
            if c is None:
                row[f"{cls}_dice"] = 0.0; row[f"{cls}_iou"] = 0.0
                continue
            cx, cy = c
            masks, _, _ = predictor.predict(
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=False, return_logits=False,
            )
            pred = masks[0].astype(bool)
            row[f"{cls}_dice"] = dice(pred, gt)
            row[f"{cls}_iou"] = iou(pred, gt)
            row[f"{cls}_pred_px"] = int(pred.sum())
            row[f"{cls}_gt_px"] = int(gt.sum())
            Image.fromarray(pred.astype(np.uint8) * 255).save(
                OUT_DIR / f"{stem}_sam3_zeroshot_{cls}_pred.png")
        print(f"  {stem}: femur Dice={row.get('femur_dice', 0):.3f} tibia Dice={row.get('tibia_dice', 0):.3f}")
        rows.append(row)

    csv_path = OUT_DIR / "crosspatient_sam3_zeroshot.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
