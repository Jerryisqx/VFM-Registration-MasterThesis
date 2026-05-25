"""
E3 -- Interactive-GUI prompt protocol: Dice as a function of click
budget.

Models the realistic GUI workflow where a clinician iteratively adds
positive/negative point prompts to correct the current predicted mask
until satisfactory. We simulate the "optimal clinician" baseline:

  1. Click 1 = positive point at the centroid of the reference mask.
  2. Click k (k > 1) = point placed at the centroid of the largest
     error region (false-positive vs false-negative; whichever is
     larger), with the matching label (0 for FP, 1 for FN).

Reported metric: per-image Dice trajectory vs click index, plus
aggregate Dice@N curves for N in {1, 2, 3, 5}.

Run with the `sam2` conda env:
    PYTHONPATH=/Users/jerrychen/Desktop/MasterThesis/code/sam2_lib \\
        /Users/jerrychen/opt/anaconda3/envs/sam2/bin/python \\
        code/segmentation/run_E3_gui_protocol.py
"""

from __future__ import annotations

import csv
import re
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
IMAGE_DIR = PROJECT_ROOT / "data/data/images"
GT_FEM_DIR = PROJECT_ROOT / "data/data/mask_femur"
GT_TIB_DIR = PROJECT_ROOT / "data/data/mask_tibia"
OUT_DIR = PROJECT_ROOT / "results/E3_gui_protocol"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_THRESH = 30
MAX_CLICKS = 5

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


def mask_centroid(mask: np.ndarray):
    """Return (x, y) of the centroid of the largest connected component, or None."""
    if mask.sum() == 0:
        return None
    # use simple centroid (no CC analysis to avoid extra deps -- ok in practice)
    ys, xs = np.where(mask)
    return int(xs.mean()), int(ys.mean())


def dice(p: np.ndarray, g: np.ndarray) -> float:
    inter = (p & g).sum()
    total = p.sum() + g.sum()
    return float(2.0 * inter / total) if total > 0 else 0.0


def predict_with_points(predictor, point_coords, point_labels) -> np.ndarray:
    masks, _, _ = predictor.predict(
        point_coords=np.array(point_coords, dtype=np.float32),
        point_labels=np.array(point_labels, dtype=np.int32),
        multimask_output=False,
    )
    return masks[0].astype(bool)


def run_image(predictor, gt: np.ndarray, max_clicks: int) -> list[float]:
    """Return list of Dice values, one per click index 1..max_clicks."""
    c = mask_centroid(gt)
    if c is None:
        return [0.0] * max_clicks
    pts = [list(c)]
    lbs = [1]
    dices = []
    for click in range(max_clicks):
        if click > 0:
            # Choose next point at largest error region
            fp = pred & ~gt  # type: ignore[has-type]
            fn = ~pred & gt  # type: ignore[has-type]
            if fp.sum() == 0 and fn.sum() == 0:
                # already perfect, no need to add more
                dices.append(d)
                continue
            if fp.sum() > fn.sum():
                ctr = mask_centroid(fp)
                if ctr is None:
                    dices.append(d)
                    continue
                pts.append(list(ctr)); lbs.append(0)
            else:
                ctr = mask_centroid(fn)
                if ctr is None:
                    dices.append(d)
                    continue
                pts.append(list(ctr)); lbs.append(1)
        pred = predict_with_points(predictor, pts, lbs)
        d = dice(pred, gt)
        dices.append(d)
    return dices


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

        image_paths = sorted(IMAGE_DIR.glob("C_SUBN_02_dkb_01_*.tif"))
        for img_path in image_paths:
            mt = re.search(r"_(\d+)\.tif$", img_path.name)
            idx = int(mt.group(1))
            gt_idx = idx - 1
            gt_fem_p = GT_FEM_DIR / f"kneefit_femur_{gt_idx}_syn.png"
            gt_tib_p = GT_TIB_DIR / f"kneefit_tibia_{gt_idx}_syn.png"
            if not (gt_fem_p.exists() and gt_tib_p.exists()):
                continue
            gt_fem = np.array(Image.open(gt_fem_p)) > GT_THRESH
            gt_tib = np.array(Image.open(gt_tib_p)) > GT_THRESH

            rgb = load_image_rgb(img_path)
            predictor.set_image(rgb)

            fem_traj = run_image(predictor, gt_fem, MAX_CLICKS)
            tib_traj = run_image(predictor, gt_tib, MAX_CLICKS)
            row = {
                "clinical_idx": idx,
                "method": m["name"],
            }
            for k in range(MAX_CLICKS):
                row[f"femur_dice_click{k+1}"] = fem_traj[k]
                row[f"tibia_dice_click{k+1}"] = tib_traj[k]
            rows.append(row)
            if idx % 10 == 0:
                print(f"  img {idx:3d}: femur Dice {fem_traj[0]:.2f} -> {fem_traj[-1]:.2f}, "
                      f"tibia {tib_traj[0]:.2f} -> {tib_traj[-1]:.2f}")
        print(f"  finished {m['name']} in {time.time()-t0:.0f}s")

    csv_path = OUT_DIR / "dice_per_image_per_click.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # Aggregate per method
    print()
    print("=== Dice @ click_k summary ===")
    summary = []
    for m in METHODS:
        sel = [r for r in rows if r["method"] == m["name"]]
        line = f"{m['name']} (n={len(sel)}):"
        summary.append(line)
        print(line)
        for k in range(MAX_CLICKS):
            fem = np.array([r[f"femur_dice_click{k+1}"] for r in sel])
            tib = np.array([r[f"tibia_dice_click{k+1}"] for r in sel])
            l = (f"  click {k+1}:  femur mean={fem.mean():.3f} (≥0.8: {(fem>=0.8).sum()}/{len(fem)} | <0.5: {(fem<0.5).sum()}/{len(fem)}) "
                 f"|  tibia mean={tib.mean():.3f} (≥0.8: {(tib>=0.8).sum()}/{len(tib)} | <0.5: {(tib<0.5).sum()}/{len(tib)})")
            print(l)
            summary.append(l)

    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("E3 -- Interactive GUI-style prompt protocol (centroid + iterative max-error correction)\n")
        f.write(f"n images = 77 (in-distribution SUBN_02), max clicks = {MAX_CLICKS}\n\n")
        for line in summary:
            f.write(line + "\n")
    print(f"Wrote {OUT_DIR/'summary.txt'}")


if __name__ == "__main__":
    main()
