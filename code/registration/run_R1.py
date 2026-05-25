"""
R1 -- Single-plane vs dual-plane registration accuracy on synthetic cube data.

Procedure:
  1. Place an 80x80x80 mm cube STL at a known true pose visible to both BS and FS.
  2. Render the dual-plane target images (the registration "observations").
  3. For N random initial poses (perturbation: +-5 mm translation, +-5 deg rotation),
     run two registrations:
       (a) single-plane (BS only)
       (b) dual-plane   (BS + FS)
  4. Compute mTRE on cube-corner fiducials.
  5. Save results.csv + summary stats.

Hypothesis: dual-plane should resolve depth ambiguity that the single-plane
optimisation gets stuck on (small loss but large z error).
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "registration_R1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment configuration -------------------------------------------------
TRUE_POSE = torch.tensor([0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
CUBE_HALF_SIDE = 40.0  # 80mm cube
N_INITS = 10
MAX_PERTURB_TRANS_MM = 5.0
MAX_PERTURB_ROT_DEG = 5.0
MAX_ITER = 120
LR = 1.0
IMG_SIZE = 128
LOSS_NAME = "mse"
SEED = 42


def render_target_pair(device: torch.device):
    """Build target images at TRUE_POSE for both BS and FS planes."""
    cube = CubeSTL(width=80, height=80, depth=80, center_pose=(0, 0, 0)).to(device)
    bs, fs = default_dual_plane_cameras()
    bs = bs.to(device); fs = fs.to(device)

    renderer = STLRenderer(device=device)
    scene = Scene()
    scene.add_anatomies(cube)
    scene.add_cameras(bs)
    scene.add_cameras(fs)
    renderer.bind_scene(scene)

    cube.set_model_matrix(pose6_to_tmat(TRUE_POSE)[None], is_yours=True)
    img_bs = renderer.render(cam_index=0, width_pixels_num=IMG_SIZE, height_pixels_num=IMG_SIZE)[0].detach()
    img_fs = renderer.render(cam_index=1, width_pixels_num=IMG_SIZE, height_pixels_num=IMG_SIZE)[0].detach()
    return img_bs, img_fs


class _CubeRegister(DualPlaneRegister):
    """Override to use a CubeSTL anatomy instead of loading from disk."""

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


def main():
    device = torch.device("cpu")  # pytorch3d on Apple Silicon is fastest on CPU here
    rng = np.random.default_rng(SEED)
    fiducials = cube_corner_fiducials(CUBE_HALF_SIDE)

    print(f"=== R1: Single vs Dual-Plane Registration ===")
    print(f"True pose: {TRUE_POSE.numpy()}")
    print(f"N inits: {N_INITS}, perturbation: +-{MAX_PERTURB_TRANS_MM} mm / +-{MAX_PERTURB_ROT_DEG} deg")
    print(f"Loss: {LOSS_NAME}, LR: {LR}, max_iter: {MAX_ITER}, img_size: {IMG_SIZE}")
    print()

    # Render targets once
    target_bs, target_fs = render_target_pair(device)
    print(f"Target BS: shape={tuple(target_bs.shape)}, fg={(target_bs > 0.1).sum().item()}")
    print(f"Target FS: shape={tuple(target_fs.shape)}, fg={(target_fs > 0.1).sum().item()}")

    # Re-create register so anatomy + scene state is fresh
    reg = _CubeRegister(device=device, img_size=IMG_SIZE)

    rows = []
    for i in range(N_INITS):
        # Sample perturbation
        dt = rng.uniform(-MAX_PERTURB_TRANS_MM, MAX_PERTURB_TRANS_MM, size=3).astype(np.float32)
        dr = rng.uniform(-MAX_PERTURB_ROT_DEG, MAX_PERTURB_ROT_DEG, size=3).astype(np.float32)
        init_pose = TRUE_POSE.numpy().copy()
        init_pose[:3] += dt
        init_pose[3:] += dr
        init_t = torch.tensor(init_pose, dtype=torch.float32)
        init_mtre = mtre(init_t, TRUE_POSE, fiducials)

        for mode, target_fs_arg in [("single", None), ("dual", target_fs)]:
            t0 = time.time()
            result = reg.register(
                target_bs=target_bs,
                target_fs=target_fs_arg,
                init_pose=init_t.clone(),
                max_iter=MAX_ITER,
                lr=LR,
                loss=LOSS_NAME,
                conv_tol=1e-5,
                conv_window=10,
                verbose=False,
            )
            final_t = torch.tensor(result.final_pose, dtype=torch.float32)
            mtre_final = mtre(final_t, TRUE_POSE, fiducials)
            t_err = translation_error(final_t, TRUE_POSE)
            r_err = rotation_error_deg(final_t, TRUE_POSE)
            row = {
                "init_id": i,
                "mode": mode,
                "init_dt": dt.tolist(),
                "init_dr": dr.tolist(),
                "init_mtre_mm": init_mtre,
                "final_mtre_mm": mtre_final,
                "t_err_mm": t_err,
                "r_err_deg": r_err,
                "final_loss": result.final_loss,
                "iterations": result.iterations,
                "converged": result.converged,
                "elapsed_s": result.elapsed_seconds,
            }
            rows.append(row)
            wall = time.time() - t0
            print(f"init {i:2d} {mode:6s}: init_mTRE={init_mtre:5.2f} -> "
                  f"final_mTRE={mtre_final:5.2f} mm, t_err={t_err:5.2f} mm, r_err={r_err:5.2f} deg, "
                  f"iters={result.iterations:3d}, conv={result.converged}, {wall:5.1f}s")

    # Save CSV
    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results: {csv_path}")

    # Summary stats
    print()
    print("=== Summary ===")
    for mode in ["single", "dual"]:
        sel = [r for r in rows if r["mode"] == mode]
        mtres = [r["final_mtre_mm"] for r in sel]
        t_errs = [r["t_err_mm"] for r in sel]
        r_errs = [r["r_err_deg"] for r in sel]
        print(f"{mode:6s}: mTRE mean={np.mean(mtres):5.2f} median={np.median(mtres):5.2f} "
              f"max={np.max(mtres):5.2f}, t_err mean={np.mean(t_errs):5.2f}, r_err mean={np.mean(r_errs):5.2f}")
        print(f"        success @2mm: {success_rate(mtres, 2.0):.0%}, @5mm: {success_rate(mtres, 5.0):.0%}")

    # Save summary
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"R1 -- Single vs Dual-Plane Registration on synthetic cube\n")
        f.write(f"True pose: {TRUE_POSE.numpy().tolist()}\n")
        f.write(f"N inits: {N_INITS}, perturbation: +-{MAX_PERTURB_TRANS_MM} mm / +-{MAX_PERTURB_ROT_DEG} deg\n\n")
        for mode in ["single", "dual"]:
            sel = [r for r in rows if r["mode"] == mode]
            mtres = [r["final_mtre_mm"] for r in sel]
            t_errs = [r["t_err_mm"] for r in sel]
            r_errs = [r["r_err_deg"] for r in sel]
            f.write(f"{mode}: mTRE mean={np.mean(mtres):.3f} median={np.median(mtres):.3f} max={np.max(mtres):.3f} mm\n")
            f.write(f"  t_err mean={np.mean(t_errs):.3f} mm, r_err mean={np.mean(r_errs):.3f} deg\n")
            f.write(f"  success @2mm: {success_rate(mtres, 2.0):.0%}, @5mm: {success_rate(mtres, 5.0):.0%}\n\n")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
