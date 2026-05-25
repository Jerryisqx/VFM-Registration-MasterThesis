"""E3 SAM3 -- Interactive-GUI prompt protocol for SAM3 zero-shot.
Same protocol as run_E3_gui_protocol.py but loads SAM3 in the
sam3_delivery env.

Run:
    PYTHONPATH=/Users/jerrychen/Desktop/MasterThesis/code/gui/03_tool/sam3_code \\
        /Users/jerrychen/opt/anaconda3/envs/sam3_delivery/bin/python \\
        code/segmentation/run_E3_gui_protocol_sam3.py
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

from sam3 import build_sam3_image_model

PROJECT_ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
SAM3_TOOL = PROJECT_ROOT / "code/gui/03_tool"
IMAGE_DIR = PROJECT_ROOT / "data/data/images"
GT_FEM_DIR = PROJECT_ROOT / "data/data/mask_femur"
GT_TIB_DIR = PROJECT_ROOT / "data/data/mask_tibia"
OUT_DIR = PROJECT_ROOT / "results/E3_gui_protocol"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = str(SAM3_TOOL / "weights" / "sam3.pt")
BPE = str(SAM3_TOOL / "sam3_code" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")

GT_THRESH = 30
MAX_CLICKS = 5
METHOD_NAME = "sam3_zeroshot"


def load_rgb(p):
    a = tifffile.imread(str(p)).astype(np.float32)
    if a.ndim == 3: a = a[..., 0] if a.shape[-1] >= 3 else a.squeeze()
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    if hi > lo: a = np.clip((a-lo)/(hi-lo),0,1)*255
    return np.stack([a.astype(np.uint8)]*3, -1)


def centroid(m):
    if m.sum() == 0: return None
    ys, xs = np.where(m)
    return int(xs.mean()), int(ys.mean())


def dice(p, g):
    i = (p & g).sum(); t = p.sum() + g.sum()
    return float(2.0 * i / t) if t > 0 else 0.0


def predict(predictor, pts, lbs):
    masks, _, _ = predictor.predict(
        point_coords=np.array(pts, np.float32),
        point_labels=np.array(lbs, np.int32),
        multimask_output=False, return_logits=False)
    return masks[0].astype(bool)


def run_image(predictor, gt, max_clicks):
    c = centroid(gt)
    if c is None: return [0.0] * max_clicks
    pts = [list(c)]; lbs = [1]
    dices = []
    pred = None
    for k in range(max_clicks):
        if k > 0:
            fp = pred & ~gt
            fn = ~pred & gt
            if fp.sum() == 0 and fn.sum() == 0:
                dices.append(d); continue
            if fp.sum() > fn.sum():
                ctr = centroid(fp)
                if ctr is None: dices.append(d); continue
                pts.append(list(ctr)); lbs.append(0)
            else:
                ctr = centroid(fn)
                if ctr is None: dices.append(d); continue
                pts.append(list(ctr)); lbs.append(1)
        pred = predict(predictor, pts, lbs)
        d = dice(pred, gt)
        dices.append(d)
    return dices


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    t0 = time.time()
    model = build_sam3_image_model(
        checkpoint_path=CKPT, bpe_path=BPE, device=device,
        eval_mode=True, load_from_HF=False,
        enable_segmentation=True, enable_inst_interactivity=True)
    predictor = model.inst_interactive_predictor
    if getattr(predictor.model, "backbone", None) is None:
        predictor.model.backbone = model.backbone
    print(f"SAM3 loaded in {time.time()-t0:.1f}s")

    rows = []
    image_paths = sorted(IMAGE_DIR.glob("C_SUBN_02_dkb_01_*.tif"))
    for img_path in image_paths:
        mt = re.search(r"_(\d+)\.tif$", img_path.name)
        idx = int(mt.group(1)); gt_idx = idx - 1
        gt_fem_p = GT_FEM_DIR / f"kneefit_femur_{gt_idx}_syn.png"
        gt_tib_p = GT_TIB_DIR / f"kneefit_tibia_{gt_idx}_syn.png"
        if not (gt_fem_p.exists() and gt_tib_p.exists()): continue
        gt_fem = np.array(Image.open(gt_fem_p)) > GT_THRESH
        gt_tib = np.array(Image.open(gt_tib_p)) > GT_THRESH

        predictor.set_image(load_rgb(img_path))
        fem_traj = run_image(predictor, gt_fem, MAX_CLICKS)
        tib_traj = run_image(predictor, gt_tib, MAX_CLICKS)
        row = {"clinical_idx": idx, "method": METHOD_NAME}
        for k in range(MAX_CLICKS):
            row[f"femur_dice_click{k+1}"] = fem_traj[k]
            row[f"tibia_dice_click{k+1}"] = tib_traj[k]
        rows.append(row)
        if idx % 10 == 0:
            print(f"  img {idx:3d}: femur {fem_traj[0]:.2f}->{fem_traj[-1]:.2f}, "
                  f"tibia {tib_traj[0]:.2f}->{tib_traj[-1]:.2f}")
    print(f"\nFinished SAM3 in {time.time()-t0:.0f}s")

    # Save to its own file then merge
    csv_path = OUT_DIR / "dice_per_image_per_click_sam3.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {csv_path}")

    print()
    print("=== SAM3 Dice @ click_k ===")
    summary_lines = [f"sam3_zeroshot (n={len(rows)}):"]
    for k in range(MAX_CLICKS):
        fem = np.array([r[f"femur_dice_click{k+1}"] for r in rows])
        tib = np.array([r[f"tibia_dice_click{k+1}"] for r in rows])
        l = (f"  click {k+1}:  femur mean={fem.mean():.3f} (≥0.8: {(fem>=0.8).sum()}/{len(fem)} | <0.5: {(fem<0.5).sum()}/{len(fem)}) "
             f"|  tibia mean={tib.mean():.3f} (≥0.8: {(tib>=0.8).sum()}/{len(tib)} | <0.5: {(tib<0.5).sum()}/{len(tib)})")
        print(l); summary_lines.append(l)
    with open(OUT_DIR / "summary_sam3.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
