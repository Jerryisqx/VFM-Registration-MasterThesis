"""
E2 cross-patient eval (U-Net 5-shot) -- retrains the same U-Net
configuration as run_E1_unet_baseline.py --shots 5 (deterministic
seed=42, same train pairs) and evaluates on the 2 cross-patient
images that have clinician-validated reference masks.

Run with the `thesis_reg` env:
    /Users/jerrychen/opt/anaconda3/envs/thesis_reg/bin/python \\
        code/segmentation/run_E2_crosspatient_unet.py
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

# Reuse training scaffolding
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_E1_unet_baseline import (
    list_pairs, KneeSeg, dice_loss, dice_score_np, iou_np,
    GT_THRESH, IMG_SIZE, SEED,
)


PROJECT_ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
TEST_DIR = PROJECT_ROOT / "data/data/test"
GUI_SAMPLES = PROJECT_ROOT / "code/gui/04_samples"
OUT_DIR = PROJECT_ROOT / "results/E2_crosspatient"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = ["C_SUBN_11_gt_f_02_029", "C_SUBO_01_gt_f_05_042"]


def load_image_uint8(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] >= 3 else arr.squeeze()
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
    return arr.astype(np.uint8)


def prep_input(img_uint8: np.ndarray, size: int, device) -> torch.Tensor:
    import cv2
    img = cv2.resize(img_uint8, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pairs = list_pairs()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]
    train_pairs = pairs[:5]
    print(f"Re-training U-Net on 5 SUBN_02 frames: {[p[0] for p in train_pairs]}")

    train_ds = KneeSeg(train_pairs, augment=True)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    bce = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for ep in range(200):
        model.train()
        for imgs, masks, _ in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = bce(out, masks) + dice_loss(out, masks)
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}/200, elapsed {time.time()-t0:.0f}s")
    print(f"Training done in {time.time()-t0:.0f}s")

    # Evaluate on cross-patient
    print("\n=== U-Net 5-shot cross-patient eval ===")
    model.eval()
    rows = []
    with torch.no_grad():
        for stem in TEST_CASES:
            img_p = TEST_DIR / f"{stem}.tif"
            lm_p = GUI_SAMPLES / f"{stem}_label_map.png"
            if not (img_p.exists() and lm_p.exists()):
                continue
            img_u8 = load_image_uint8(img_p)
            lm = np.array(Image.open(lm_p))
            gt_fem_full = (lm == 1)
            gt_tib_full = (lm == 2)

            # Resize GT to model resolution for fair comparison, then compute Dice
            import cv2
            gt_fem_lo = cv2.resize(gt_fem_full.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
            gt_tib_lo = cv2.resize(gt_tib_full.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)

            x = prep_input(img_u8, IMG_SIZE, device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            pred_fem = probs[0] > 0.5
            pred_tib = probs[1] > 0.5

            row = {
                "image": stem,
                "method": "unet_5shot",
                "femur_dice": dice_score_np(pred_fem, gt_fem_lo),
                "femur_iou":  iou_np(pred_fem, gt_fem_lo),
                "tibia_dice": dice_score_np(pred_tib, gt_tib_lo),
                "tibia_iou":  iou_np(pred_tib, gt_tib_lo),
                "femur_pred_px": int(pred_fem.sum()),
                "femur_gt_px":   int(gt_fem_lo.sum()),
                "tibia_pred_px": int(pred_tib.sum()),
                "tibia_gt_px":   int(gt_tib_lo.sum()),
            }
            print(f"  {stem}: femur Dice={row['femur_dice']:.3f} tibia Dice={row['tibia_dice']:.3f}")
            rows.append(row)

            # Save predicted masks resized back for visualization
            for cls, pred in [("femur", pred_fem), ("tibia", pred_tib)]:
                pred_full = cv2.resize(pred.astype(np.uint8) * 255,
                                        (gt_fem_full.shape[1], gt_fem_full.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                Image.fromarray(pred_full).save(OUT_DIR / f"{stem}_unet_5shot_{cls}_pred.png")

    csv_path = OUT_DIR / "crosspatient_unet5shot.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
