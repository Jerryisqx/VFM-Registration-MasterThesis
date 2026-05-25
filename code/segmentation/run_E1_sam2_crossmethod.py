"""
E1 cross-method (SAM2 family) -- runs SAM2 Base+ zero-shot and
MedSAM2-Tiny 5-shot fine-tune on the same 77 clinical X-rays using
identical point prompts (centroid of GT mask per bone), then computes
Dice and IoU against the DRR-derived GT silhouettes.

Run with the `sam2` conda env, PYTHONPATH including the local
sam2 fork:

    PYTHONPATH=/Users/jerrychen/Desktop/MasterThesis/code/sam2_lib \\
        /Users/jerrychen/opt/anaconda3/envs/sam2/bin/python \\
        code/segmentation/run_E1_sam2_crossmethod.py

Outputs:
    results/E1_segmentation/dice_per_image_<method>.csv
    results/E1_segmentation/summary_crossmethod.txt
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

# Hydra config init (required before build_sam2)
from hydra import initialize_config_module
try:
    initialize_config_module("sam2", version_base="1.2")
except Exception:
    pass

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


PROJECT_ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
IMAGE_DIR = PROJECT_ROOT / "data/data/images"
GT_FEM_DIR = PROJECT_ROOT / "data/data/mask_femur"
GT_TIB_DIR = PROJECT_ROOT / "data/data/mask_tibia"
OUT_DIR = PROJECT_ROOT / "results/E1_segmentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_THRESH = 30

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
    """Load uint16 TIFF, percentile-stretch to uint8, convert to RGB (H,W,3)."""
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] >= 3 else arr.squeeze()
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
    arr = arr.astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return rgb


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    """Return (x, y) pixel coordinates of GT mask centroid, or None if empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    return float(2.0 * inter / total) if total > 0 else 0.0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0


def run_method(method: dict, device: torch.device) -> list[dict]:
    print(f"\n=== {method['name']} ===")
    print(f"  config: {method['config']}")
    print(f"  ckpt:   {method['ckpt']}")
    t0 = time.time()
    model = build_sam2(method["config"], method["ckpt"], device=device)
    predictor = SAM2ImagePredictor(model)
    print(f"  model loaded in {time.time()-t0:.1f}s on {device}")

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

        out = {"clinical_idx": idx, "image_name": img_path.name, "method": method["name"]}
        for cls_name, gt in [("femur", gt_fem), ("tibia", gt_tib)]:
            c = mask_centroid(gt)
            if c is None:
                out[f"{cls_name}_dice"] = 0.0
                out[f"{cls_name}_iou"]  = 0.0
                continue
            cx, cy = c
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=False,
            )
            pred = masks[0].astype(bool)
            out[f"{cls_name}_dice"]    = dice(pred, gt)
            out[f"{cls_name}_iou"]     = iou(pred, gt)
            out[f"{cls_name}_pred_px"] = int(pred.sum())
            out[f"{cls_name}_gt_px"]   = int(gt.sum())
        rows.append(out)
        if len(rows) % 10 == 0:
            print(f"  ... {len(rows)}/{len(image_paths)} done, last femur Dice={out.get('femur_dice', 0):.3f} tibia Dice={out.get('tibia_dice', 0):.3f}")

    elapsed = time.time() - t0
    print(f"  finished {len(rows)} images in {elapsed:.0f}s")
    return rows


def summarise(rows: list[dict], name: str) -> str:
    fem_d = np.array([r["femur_dice"] for r in rows])
    tib_d = np.array([r["tibia_dice"] for r in rows])
    fem_i = np.array([r["femur_iou"] for r in rows])
    tib_i = np.array([r["tibia_iou"] for r in rows])

    def s(a):
        return f"mean={a.mean():.3f} median={np.median(a):.3f} min={a.min():.3f} max={a.max():.3f} std={a.std():.3f}"

    return (
        f"{name}: n={len(rows)}\n"
        f"  femur Dice  {s(fem_d)}  | ≥0.8: {(fem_d>=0.8).sum()}/{len(fem_d)}  | <0.5: {(fem_d<0.5).sum()}/{len(fem_d)}\n"
        f"  tibia Dice  {s(tib_d)}  | ≥0.8: {(tib_d>=0.8).sum()}/{len(tib_d)}  | <0.5: {(tib_d<0.5).sum()}/{len(tib_d)}\n"
        f"  femur IoU   {s(fem_i)}\n"
        f"  tibia IoU   {s(tib_i)}\n"
    )


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    summary_lines = []
    for method in METHODS:
        rows = run_method(method, device)
        csv_path = OUT_DIR / f"dice_per_image_{method['name']}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  -> wrote {csv_path}")
        s = summarise(rows, method["name"])
        print()
        print(s)
        summary_lines.append(s)

    out_summary = OUT_DIR / "summary_crossmethod_sam2.txt"
    with open(out_summary, "w") as f:
        f.write("E1 cross-method (SAM2 family) on 77 clinical X-rays\n")
        f.write(f"GT binarisation > {GT_THRESH}, prompt = centroid of GT mask, multimask_output=False\n\n")
        for s in summary_lines:
            f.write(s + "\n")
    print(f"Wrote summary {out_summary}")


if __name__ == "__main__":
    main()
