"""
E1 cross-method (U-Net baseline) -- trains a standard U-Net on the
same 77 clinical biplanar X-rays for two regimes and evaluates Dice
against the DRR-derived GT silhouettes.

  - 5-shot:    train on 5 random frames, test on the remaining 72
  - fulldata:  train on 67 random frames, test on the remaining 10

Architecture: U-Net with a frozen ImageNet-pretrained ResNet-34 encoder
(segmentation-models-pytorch), 2 output channels (femur, tibia).
Loss = BCE + Dice. Augmentation = flip + rotate.

Run with the `thesis_reg` env (segmentation-models-pytorch installed):

    /Users/jerrychen/opt/anaconda3/envs/thesis_reg/bin/python \\
        code/segmentation/run_E1_unet_baseline.py [--shots 5|full]

Outputs:
    results/E1_segmentation/dice_per_image_unet_<regime>.csv
    results/E1_segmentation/summary_unet_<regime>.txt
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp
import albumentations as A


PROJECT_ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
IMAGE_DIR = PROJECT_ROOT / "data/data/images"
GT_FEM_DIR = PROJECT_ROOT / "data/data/mask_femur"
GT_TIB_DIR = PROJECT_ROOT / "data/data/mask_tibia"
OUT_DIR = PROJECT_ROOT / "results/E1_segmentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_THRESH = 30
IMG_SIZE = 512
SEED = 42


def list_pairs():
    """Return list of (image_path, gt_femur_path, gt_tibia_path) by frame index."""
    paths = []
    for p in sorted(IMAGE_DIR.glob("C_SUBN_02_dkb_01_*.tif")):
        m = re.search(r"_(\d+)\.tif$", p.name)
        idx = int(m.group(1))
        gt_idx = idx - 1
        gf = GT_FEM_DIR / f"kneefit_femur_{gt_idx}_syn.png"
        gt = GT_TIB_DIR / f"kneefit_tibia_{gt_idx}_syn.png"
        if gf.exists() and gt.exists():
            paths.append((idx, p, gf, gt))
    return paths


def load_image_uint8(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] >= 3 else arr.squeeze()
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
    return arr.astype(np.uint8)


class KneeSeg(Dataset):
    def __init__(self, pairs, size=IMG_SIZE, augment=False):
        self.pairs = pairs
        self.size = size
        if augment:
            self.tx = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=25, p=0.7, border_mode=0),
                A.RandomBrightnessContrast(p=0.3),
            ])
        else:
            self.tx = A.Compose([A.Resize(size, size)])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        idx, img_p, gf_p, gt_p = self.pairs[i]
        img = load_image_uint8(img_p)
        gf = (np.array(Image.open(gf_p)) > GT_THRESH).astype(np.float32)
        gt = (np.array(Image.open(gt_p)) > GT_THRESH).astype(np.float32)
        a = self.tx(image=img, masks=[gf, gt])
        img = a["image"]
        gf, gt = a["masks"]
        img_t = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
        # ImageNet normalisation for resnet encoder
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        mask_t = torch.from_numpy(np.stack([gf, gt], axis=0)).float()
        return img_t, mask_t, idx


def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    inter = (pred * target).sum(dim=(2, 3))
    total = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (total + eps)).mean()


def dice_score_np(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    return float(2.0 * inter / total) if total > 0 else 0.0


def iou_np(p, g):
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="5", help="'5' for 5-shot, 'full' for fulldata")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    pairs = list_pairs()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]

    if args.shots == "5":
        train_pairs = pairs[:5]
        test_pairs = pairs[5:]
        epochs = args.epochs or 200   # 5 imgs → high epoch count
        regime = "5shot"
    elif args.shots == "full":
        train_pairs = pairs[:67]
        test_pairs = pairs[67:]
        epochs = args.epochs or 50
        regime = "fulldata"
    else:
        raise ValueError("--shots must be 5 or full")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  |  regime: {regime}  |  train n={len(train_pairs)}  test n={len(test_pairs)}  epochs={epochs}")
    print(f"Test image indices: {[p[0] for p in test_pairs[:10]]}...")

    train_ds = KneeSeg(train_pairs, augment=True)
    test_ds = KneeSeg(test_pairs, augment=False)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()

    print("Training...")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for imgs, masks, _ in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = bce(out, masks) + dice_loss(out, masks)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {ep+1}/{epochs}  loss={ep_loss/len(train_dl):.4f}  elapsed={time.time()-t0:.0f}s")
    print(f"Training done in {time.time()-t0:.0f}s")

    print("Evaluating...")
    model.eval()
    rows = []
    with torch.no_grad():
        for img_t, mask_t, idx_t in test_dl:
            img_t = img_t.to(device)
            logits = model(img_t)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # (2, H, W)
            mask_np = mask_t.numpy()[0]  # (2, H, W)
            pred_fem = probs[0] > 0.5
            pred_tib = probs[1] > 0.5
            gt_fem = mask_np[0] > 0.5
            gt_tib = mask_np[1] > 0.5
            idx = int(idx_t.item())
            rows.append({
                "clinical_idx": idx,
                "method": f"unet_{regime}",
                "femur_dice": dice_score_np(pred_fem, gt_fem),
                "femur_iou": iou_np(pred_fem, gt_fem),
                "tibia_dice": dice_score_np(pred_tib, gt_tib),
                "tibia_iou": iou_np(pred_tib, gt_tib),
                "femur_pred_px": int(pred_fem.sum()),
                "femur_gt_px": int(gt_fem.sum()),
                "tibia_pred_px": int(pred_tib.sum()),
                "tibia_gt_px": int(gt_tib.sum()),
            })

    csv_path = OUT_DIR / f"dice_per_image_unet_{regime}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    fem = np.array([r["femur_dice"] for r in rows])
    tib = np.array([r["tibia_dice"] for r in rows])
    print()
    print(f"=== U-Net {regime}, n={len(rows)} ===")
    print(f"femur Dice mean={fem.mean():.3f} median={np.median(fem):.3f} min={fem.min():.3f} max={fem.max():.3f} | ≥0.8: {(fem>=0.8).sum()}/{len(fem)} | <0.5: {(fem<0.5).sum()}/{len(fem)}")
    print(f"tibia Dice mean={tib.mean():.3f} median={np.median(tib):.3f} min={tib.min():.3f} max={tib.max():.3f} | ≥0.8: {(tib>=0.8).sum()}/{len(tib)} | <0.5: {(tib<0.5).sum()}/{len(tib)}")

    with open(OUT_DIR / f"summary_unet_{regime}.txt", "w") as f:
        f.write(f"E1 U-Net {regime}, train n={len(train_pairs)}, test n={len(test_pairs)}, epochs={epochs}\n")
        f.write(f"Encoder: resnet34 (ImageNet pretrained). Loss = BCE + Dice. Aug: flip + rotate.\n\n")
        f.write(f"femur Dice mean={fem.mean():.3f} median={np.median(fem):.3f} min={fem.min():.3f} max={fem.max():.3f}\n")
        f.write(f"  ≥0.8: {(fem>=0.8).sum()}/{len(fem)} | <0.5: {(fem<0.5).sum()}/{len(fem)}\n")
        f.write(f"tibia Dice mean={tib.mean():.3f} median={np.median(tib):.3f} min={tib.min():.3f} max={tib.max():.3f}\n")
        f.write(f"  ≥0.8: {(tib>=0.8).sum()}/{len(tib)} | <0.5: {(tib<0.5).sum()}/{len(tib)}\n")


if __name__ == "__main__":
    main()
