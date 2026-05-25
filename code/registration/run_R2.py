"""
R2 -- Masked vs unmasked similarity for dual-plane registration.

Procedure:
  1. Same cube + true pose as R1, dual-plane setup.
  2. Build target images by *adding structured background noise* outside
     the cube silhouette to both BS and FS targets, simulating the kind
     of cluttered background a real X-ray would have.
  3. The "ground-truth mask" for each plane is the cube silhouette
     itself (rendered separately at the true pose).
  4. For N=10 random initial poses (perturbation +-5 mm / +-5 deg):
       (a) unmasked dual-plane register on noisy targets
       (b) masked dual-plane register on noisy targets (loss restricted
           to the cube silhouette region)
  5. Compute mTRE on cube corners.

Hypothesis: unmasked registration is biased by background noise, while
the masked variant matches the dual-plane R1 result.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from dupla_renderers.pytorch3d import CubeSTL, Scene, STLRenderer
from register import (
    DualPlaneRegister,
    default_dual_plane_cameras,
    pose6_to_tmat,
)
from evaluation import (
    cube_corner_fiducials,
    mtre,
    rotation_error_deg,
    success_rate,
    translation_error,
)


PROJECT_ROOT = THIS_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "registration_R2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRUE_POSE = torch.tensor([0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
CUBE_HALF_SIDE = 40.0
N_INITS = 10
MAX_PERTURB_TRANS_MM = 5.0
MAX_PERTURB_ROT_DEG = 5.0
MAX_ITER = 120
LR = 1.0
IMG_SIZE = 128
LOSS_NAME = "mse"
NOISE_AMPLITUDE = 0.6     # mean of structured background noise outside silhouette
NOISE_FREQ = 8            # frequency of the sinusoidal background pattern (cycles per image)
SEED = 42


class _CubeRegister(DualPlaneRegister):
    def __init__(self, device, img_size):
        self.device = torch.device(device)
        self.modality = "STL"
        self.img_size = img_size
        self.anatomy = CubeSTL(width=80, height=80, depth=80, center_pose=(0, 0, 0)).to(self.device)
        self.renderer = STLRenderer(device=self.device)
        self.cam_bs, self.cam_fs = default_dual_plane_cameras()
        self.cam_bs = self.cam_bs.to(self.device)
        self.cam_fs = self.cam_fs.to(self.device)
        self.scene = Scene()
        self.scene.add_anatomies(self.anatomy)
        self.scene.add_cameras(self.cam_bs)
        self.scene.add_cameras(self.cam_fs)
        self.renderer.bind_scene(self.scene)


def make_structured_noise(img_size: int, freq: float, amplitude: float, seed: int) -> torch.Tensor:
    """Periodic + Gaussian background noise pattern in [0, 1]."""
    rng = np.random.default_rng(seed)
    xs = np.arange(img_size) / img_size * 2 * np.pi * freq
    ys = np.arange(img_size) / img_size * 2 * np.pi * freq
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    pattern = 0.5 + 0.25 * np.sin(XX) * np.cos(YY)        # smooth oscillation in [0.25, 0.75]
    pattern += rng.normal(0.0, 0.05, pattern.shape)        # high-frequency Gaussian
    pattern = np.clip(pattern, 0.0, 1.0).astype(np.float32) * amplitude
    return torch.tensor(pattern, dtype=torch.float32)


def main():
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    fiducials = cube_corner_fiducials(CUBE_HALF_SIDE)

    print(f"=== R2: Masked vs Unmasked Similarity (dual-plane) ===")
    print(f"True pose: {TRUE_POSE.numpy()}")
    print(f"N inits: {N_INITS}, perturbation: +-{MAX_PERTURB_TRANS_MM} mm / +-{MAX_PERTURB_ROT_DEG} deg")
    print(f"Background noise: amplitude {NOISE_AMPLITUDE}, freq {NOISE_FREQ}")

    reg = _CubeRegister(device, IMG_SIZE)

    # Clean targets at true pose
    clean_bs = reg.render(TRUE_POSE, "bs").detach()
    clean_fs = reg.render(TRUE_POSE, "fs").detach()

    # Ground-truth silhouette masks (binary)
    mask_bs = (clean_bs > 0.1).float()
    mask_fs = (clean_fs > 0.1).float()

    # Background noise patterns added OUTSIDE the silhouette
    noise_bs = make_structured_noise(IMG_SIZE, NOISE_FREQ, NOISE_AMPLITUDE, SEED)
    noise_fs = make_structured_noise(IMG_SIZE, NOISE_FREQ, NOISE_AMPLITUDE, SEED + 1)
    bg_bs = noise_bs * (1 - mask_bs)
    bg_fs = noise_fs * (1 - mask_fs)
    target_bs = torch.clamp(clean_bs + bg_bs, 0.0, 1.0)
    target_fs = torch.clamp(clean_fs + bg_fs, 0.0, 1.0)

    print(f"target_bs background fraction with noise: "
          f"{((target_bs > 0.1) & (mask_bs < 0.5)).float().mean().item():.3f}")
    print(f"target_fs background fraction with noise: "
          f"{((target_fs > 0.1) & (mask_fs < 0.5)).float().mean().item():.3f}")

    # Save targets for reference
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax_row, name, clean, target, mask in [
        (axes[0], "BS", clean_bs, target_bs, mask_bs),
        (axes[1], "FS", clean_fs, target_fs, mask_fs),
    ]:
        ax_row[0].imshow(clean.numpy(), cmap="gray", vmin=0, vmax=1); ax_row[0].set_title(f"{name} clean")
        ax_row[1].imshow(target.numpy(), cmap="gray", vmin=0, vmax=1); ax_row[1].set_title(f"{name} target+bg noise")
        ax_row[2].imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1); ax_row[2].set_title(f"{name} GT mask")
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "targets_and_masks.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'targets_and_masks.png'}")

    rows = []
    for i in range(N_INITS):
        dt = rng.uniform(-MAX_PERTURB_TRANS_MM, MAX_PERTURB_TRANS_MM, size=3).astype(np.float32)
        dr = rng.uniform(-MAX_PERTURB_ROT_DEG, MAX_PERTURB_ROT_DEG, size=3).astype(np.float32)
        init = TRUE_POSE.numpy().copy()
        init[:3] += dt
        init[3:] += dr
        init_t = torch.tensor(init, dtype=torch.float32)

        for mode in ["unmasked", "masked"]:
            t0 = time.time()
            kwargs = dict(
                target_bs=target_bs,
                target_fs=target_fs,
                init_pose=init_t.clone(),
                max_iter=MAX_ITER,
                lr=LR,
                loss=LOSS_NAME,
            )
            if mode == "masked":
                kwargs["mask_bs"] = mask_bs
                kwargs["mask_fs"] = mask_fs
            result = reg.register(**kwargs)
            mtre_final = mtre(torch.tensor(result.final_pose), TRUE_POSE, fiducials)
            t_err = translation_error(result.final_pose, TRUE_POSE)
            r_err = rotation_error_deg(result.final_pose, TRUE_POSE)
            wall = time.time() - t0
            row = {
                "init_id": i,
                "mode": mode,
                "mtre_mm": mtre_final,
                "t_err_mm": t_err,
                "r_err_deg": r_err,
                "iterations": result.iterations,
                "converged": result.converged,
                "elapsed_s": wall,
            }
            rows.append(row)
            print(f"init {i:2d} {mode:8s}: mTRE={mtre_final:5.2f} mm, t={t_err:5.2f}, r={r_err:5.2f}, "
                  f"iters={result.iterations}, {wall:4.1f}s")

    # Save CSV
    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Summary
    print()
    print("=== Summary ===")
    summary_lines = []
    for mode in ["unmasked", "masked"]:
        sel = [r["mtre_mm"] for r in rows if r["mode"] == mode]
        line = (f"{mode:8s}: mean={np.mean(sel):5.2f} median={np.median(sel):5.2f} "
                f"max={np.max(sel):5.2f} | success @2mm={success_rate(sel, 2.0):.0%}, "
                f"@5mm={success_rate(sel, 5.0):.0%}")
        print(line)
        summary_lines.append(line)

    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        f.write("R2 -- Masked vs Unmasked dual-plane registration with structured background noise\n")
        f.write(f"True pose: {TRUE_POSE.numpy().tolist()}, N inits: {N_INITS}\n")
        f.write(f"Noise: amplitude {NOISE_AMPLITUDE}, freq {NOISE_FREQ}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"Saved: {OUTPUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
