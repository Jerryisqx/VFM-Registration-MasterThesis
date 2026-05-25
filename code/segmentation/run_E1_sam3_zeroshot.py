"""
E1 cross-method (SAM3 zero-shot) -- runs SAM3 on the same 77 clinical
X-rays with the centroid-of-GT-mask point prompt strategy, computing
Dice/IoU against the DRR-derived GT silhouettes.

Run with the `sam3_delivery` conda env:

    PYTHONPATH=/Users/jerrychen/Desktop/MasterThesis/code/gui/03_tool/sam3_code \\
        /Users/jerrychen/opt/anaconda3/envs/sam3_delivery/bin/python \\
        code/segmentation/run_E1_sam3_zeroshot.py

Outputs:
    results/E1_segmentation/dice_per_image_sam3_zeroshot.csv
    results/E1_segmentation/summary_sam3_zeroshot.txt
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image

from sam3 import build_sam3_image_model


PROJECT_ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
SAM3_TOOL_ROOT = PROJECT_ROOT / "code/gui/03_tool"
IMAGE_DIR = PROJECT_ROOT / "data/data/images"
GT_FEM_DIR = PROJECT_ROOT / "data/data/mask_femur"
GT_TIB_DIR = PROJECT_ROOT / "data/data/mask_tibia"
OUT_DIR = PROJECT_ROOT / "results/E1_segmentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT = str(SAM3_TOOL_ROOT / "weights" / "sam3.pt")
BPE = str(SAM3_TOOL_ROOT / "sam3_code" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")

GT_THRESH = 30


def load_image_rgb(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] >= 3 else arr.squeeze()
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
    arr = arr.astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
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
    print(f"Loading SAM3 from {CKPT}")
    t0 = time.time()
    model = build_sam3_image_model(
        checkpoint_path=CKPT,
        bpe_path=BPE,
        device=device,
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=True,
    )
    predictor = model.inst_interactive_predictor
    if getattr(predictor.model, "backbone", None) is None:
        predictor.model.backbone = model.backbone
    print(f"SAM3 loaded in {time.time()-t0:.1f}s")

    rows = []
    image_paths = sorted(IMAGE_DIR.glob("C_SUBN_02_dkb_01_*.tif"))
    for img_path in image_paths:
        m = re.search(r"_(\d+)\.tif$", img_path.name)
        idx = int(m.group(1))
        gt_idx = idx - 1
        gt_fem_p = GT_FEM_DIR / f"kneefit_femur_{gt_idx}_syn.png"
        gt_tib_p = GT_TIB_DIR / f"kneefit_tibia_{gt_idx}_syn.png"
        if not (gt_fem_p.exists() and gt_tib_p.exists()):
            continue
        gt_fem = np.array(Image.open(gt_fem_p)) > GT_THRESH
        gt_tib = np.array(Image.open(gt_tib_p)) > GT_THRESH

        rgb = load_image_rgb(img_path)
        predictor.set_image(rgb)

        out = {"clinical_idx": idx, "image_name": img_path.name, "method": "sam3_zeroshot"}
        for cls_name, gt in [("femur", gt_fem), ("tibia", gt_tib)]:
            c = mask_centroid(gt)
            if c is None:
                out[f"{cls_name}_dice"] = 0.0
                out[f"{cls_name}_iou"] = 0.0
                continue
            cx, cy = c
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=False,
                return_logits=False,
            )
            pred = masks[0].astype(bool)
            out[f"{cls_name}_dice"] = dice(pred, gt)
            out[f"{cls_name}_iou"] = iou(pred, gt)
            out[f"{cls_name}_pred_px"] = int(pred.sum())
            out[f"{cls_name}_gt_px"] = int(gt.sum())
        rows.append(out)
        if len(rows) % 10 == 0:
            print(f"  ... {len(rows)}/{len(image_paths)} done, femur Dice={out.get('femur_dice', 0):.3f} tibia Dice={out.get('tibia_dice', 0):.3f}")

    csv_path = OUT_DIR / "dice_per_image_sam3_zeroshot.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    fem_d = np.array([r["femur_dice"] for r in rows])
    tib_d = np.array([r["tibia_dice"] for r in rows])
    print()
    print(f"=== SAM3 zero-shot, n={len(rows)} ===")
    print(f"femur Dice mean={fem_d.mean():.3f} median={np.median(fem_d):.3f} min={fem_d.min():.3f} max={fem_d.max():.3f} | ≥0.8: {(fem_d>=0.8).sum()}/{len(fem_d)} | <0.5: {(fem_d<0.5).sum()}/{len(fem_d)}")
    print(f"tibia Dice mean={tib_d.mean():.3f} median={np.median(tib_d):.3f} min={tib_d.min():.3f} max={tib_d.max():.3f} | ≥0.8: {(tib_d>=0.8).sum()}/{len(tib_d)} | <0.5: {(tib_d<0.5).sum()}/{len(tib_d)}")

    with open(OUT_DIR / "summary_sam3_zeroshot.txt", "w") as f:
        f.write("E1 SAM3 zero-shot on 77 clinical X-rays\n")
        f.write(f"GT thresh > {GT_THRESH}, prompt = GT centroid (single positive point)\n\n")
        f.write(f"femur Dice mean={fem_d.mean():.3f} median={np.median(fem_d):.3f} min={fem_d.min():.3f} max={fem_d.max():.3f}\n")
        f.write(f"  ≥0.8: {(fem_d>=0.8).sum()}/{len(fem_d)} | <0.5: {(fem_d<0.5).sum()}/{len(fem_d)}\n")
        f.write(f"tibia Dice mean={tib_d.mean():.3f} median={np.median(tib_d):.3f} min={tib_d.min():.3f} max={tib_d.max():.3f}\n")
        f.write(f"  ≥0.8: {(tib_d>=0.8).sum()}/{len(tib_d)} | <0.5: {(tib_d<0.5).sum()}/{len(tib_d)}\n")


if __name__ == "__main__":
    main()
