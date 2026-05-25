"""
R2 v2 -- Masked vs unmasked similarity, with a STRUCTURAL distractor.

Difference from R2 v1: instead of additive background noise (which the
unmasked optimiser handled fine because the bone silhouette was still
the brightest signal), we add a second cube as a distractor anatomy,
rendered at a known wrong location. Both cubes appear in the BS and
FS targets.

Procedure:
  1. Target = (cube at TRUE_POSE) ∪ (distractor cube at DISTRACTOR_POSE),
     in both BS and FS, clipped to [0, 1].
  2. Mask  = silhouette of the TRUE cube only (no distractor pixels).
  3. For N=10 random initial poses (perturbation +-5 mm/+-5 deg around TRUE):
       (a) unmasked dual-plane register on target -- optimiser confused
           by distractor and may lock onto it.
       (b) masked dual-plane register on target -- loss restricted to
           true-cube region; distractor pixels carry zero weight.
  4. Compute mTRE on cube corners.

Hypothesis: masked >> unmasked when the distractor is structurally
similar to the target (mimics multiple anatomies / implants in field).
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "registration_R2_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRUE_POSE = torch.tensor([0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
DISTRACTOR_POSE = torch.tensor([60.0, 0.0, 250.0, 0.0, 0.0, 30.0])  # offset +x, closer, rotated 30 deg about Z
CUBE_HALF_SIDE = 40.0
N_INITS = 10
MAX_PERTURB_TRANS_MM = 5.0
MAX_PERTURB_ROT_DEG = 5.0
MAX_ITER = 120
LR = 1.0
IMG_SIZE = 128
LOSS_NAME = "mse"
SEED = 42


class _CubeRegister(DualPlaneRegister):
    """Register a single cube anatomy (the target). Distractor lives in target image only."""

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


def render_one(reg: _CubeRegister, pose: torch.Tensor, plane: str) -> torch.Tensor:
    return reg.render(pose, plane=plane).detach()


def main():
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    fiducials = cube_corner_fiducials(CUBE_HALF_SIDE)

    print(f"=== R2 v2: Masked vs Unmasked Similarity (with structural distractor) ===")
    print(f"True pose:       {TRUE_POSE.numpy()}")
    print(f"Distractor pose: {DISTRACTOR_POSE.numpy()}")
    print(f"N inits: {N_INITS}, perturbation: +-{MAX_PERTURB_TRANS_MM} mm / +-{MAX_PERTURB_ROT_DEG} deg")
    print()

    reg = _CubeRegister(device, IMG_SIZE)

    # Render true cube at true pose -> "clean" image and silhouette mask
    target_clean_bs = render_one(reg, TRUE_POSE, "bs")
    target_clean_fs = render_one(reg, TRUE_POSE, "fs")
    mask_bs = (target_clean_bs > 0.1).float()
    mask_fs = (target_clean_fs > 0.1).float()

    # Render distractor cube at distractor pose
    distractor_bs = render_one(reg, DISTRACTOR_POSE, "bs")
    distractor_fs = render_one(reg, DISTRACTOR_POSE, "fs")

    # Combine: target image = max(target, distractor) clipped to [0,1]
    target_bs = torch.clamp(target_clean_bs + distractor_bs, 0.0, 1.0)
    target_fs = torch.clamp(target_clean_fs + distractor_fs, 0.0, 1.0)

    print(f"target_bs fg pixels: {int((target_bs > 0.1).sum().item())} "
          f"(true cube {int(mask_bs.sum().item())}, distractor {int((distractor_bs > 0.1).sum().item())})")
    print(f"target_fs fg pixels: {int((target_fs > 0.1).sum().item())} "
          f"(true cube {int(mask_fs.sum().item())}, distractor {int((distractor_fs > 0.1).sum().item())})")
    overlap_bs = int(((mask_bs > 0.5) & (distractor_bs > 0.1)).sum().item())
    overlap_fs = int(((mask_fs > 0.5) & (distractor_fs > 0.1)).sum().item())
    print(f"target/distractor overlap pixels: BS={overlap_bs}, FS={overlap_fs}")

    # Save targets for inspection
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax_row, name, clean, target, mask, distractor in [
        (axes[0], "BS", target_clean_bs, target_bs, mask_bs, distractor_bs),
        (axes[1], "FS", target_clean_fs, target_fs, mask_fs, distractor_fs),
    ]:
        ax_row[0].imshow(clean.numpy(), cmap="gray", vmin=0, vmax=1); ax_row[0].set_title(f"{name} target only")
        ax_row[1].imshow(distractor.numpy(), cmap="gray", vmin=0, vmax=1); ax_row[1].set_title(f"{name} distractor only")
        ax_row[2].imshow(target.numpy(), cmap="gray", vmin=0, vmax=1); ax_row[2].set_title(f"{name} target+distractor")
        ax_row[3].imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1); ax_row[3].set_title(f"{name} mask (target only)")
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scene_setup.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'scene_setup.png'}")
    print()

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

    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

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
        f.write("R2 v2 -- Masked vs unmasked dual-plane registration with structural distractor\n")
        f.write(f"True pose: {TRUE_POSE.numpy().tolist()}, Distractor: {DISTRACTOR_POSE.numpy().tolist()}\n")
        f.write(f"N inits: {N_INITS}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"Saved: {OUTPUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
