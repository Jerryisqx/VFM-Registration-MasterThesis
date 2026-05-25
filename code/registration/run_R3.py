"""
R3 -- Initialisation perturbation robustness on synthetic cube data.

Procedure:
  1. Same cube + true pose as R1.
  2. Sweep perturbation magnitude delta in {2, 5, 10, 20} (paired mm and deg).
  3. For each delta, sample N=20 random initial poses uniformly within +-delta.
  4. Run dual-plane registration only.
  5. Report success rate (mTRE < 2 mm) vs delta.
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "registration_R3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRUE_POSE = torch.tensor([0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
CUBE_HALF_SIDE = 40.0
N_PER_DELTA = 10
DELTAS = [2.0, 5.0, 10.0, 20.0]
MAX_ITER = 120
LR = 1.0
IMG_SIZE = 128
LOSS_NAME = "mse"
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


def main():
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    fiducials = cube_corner_fiducials(CUBE_HALF_SIDE)

    print(f"=== R3: Initialisation Robustness (dual-plane) ===")
    print(f"True pose: {TRUE_POSE.numpy()}")
    print(f"Deltas: {DELTAS}, N per delta: {N_PER_DELTA}, max_iter: {MAX_ITER}")
    print()

    reg = _CubeRegister(device, IMG_SIZE)
    target_bs = reg.render(TRUE_POSE, "bs").detach()
    target_fs = reg.render(TRUE_POSE, "fs").detach()

    rows = []
    for delta in DELTAS:
        delta_results = []
        for i in range(N_PER_DELTA):
            dt = rng.uniform(-delta, delta, size=3).astype(np.float32)
            dr = rng.uniform(-delta, delta, size=3).astype(np.float32)
            init = TRUE_POSE.numpy().copy()
            init[:3] += dt
            init[3:] += dr
            init_t = torch.tensor(init, dtype=torch.float32)

            t0 = time.time()
            result = reg.register(
                target_bs=target_bs,
                target_fs=target_fs,
                init_pose=init_t.clone(),
                max_iter=MAX_ITER,
                lr=LR,
                loss=LOSS_NAME,
            )
            mtre_final = mtre(torch.tensor(result.final_pose), TRUE_POSE, fiducials)
            t_err = translation_error(result.final_pose, TRUE_POSE)
            r_err = rotation_error_deg(result.final_pose, TRUE_POSE)
            wall = time.time() - t0

            row = {
                "delta": delta,
                "init_id": i,
                "mtre_mm": mtre_final,
                "t_err_mm": t_err,
                "r_err_deg": r_err,
                "iterations": result.iterations,
                "converged": result.converged,
                "elapsed_s": wall,
            }
            rows.append(row)
            delta_results.append(mtre_final)
            print(f"  delta={delta:5.1f} init={i:2d}: mTRE={mtre_final:5.2f} mm "
                  f"({wall:4.1f}s, iters={result.iterations})")
        succ_2 = success_rate(delta_results, 2.0)
        succ_5 = success_rate(delta_results, 5.0)
        print(f"  --> delta={delta:5.1f}: mean mTRE={np.mean(delta_results):5.2f} mm, "
              f"success @2mm={succ_2:.0%}, @5mm={succ_5:.0%}")

    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"R3 -- Initialisation perturbation robustness (dual-plane)\n")
        f.write(f"True pose: {TRUE_POSE.numpy().tolist()}, N per delta: {N_PER_DELTA}\n\n")
        f.write(f"{'delta(mm,deg)':>14} {'mean_mTRE_mm':>14} {'median_mTRE_mm':>16} "
                f"{'success@2mm':>13} {'success@5mm':>13}\n")
        for delta in DELTAS:
            sel = [r["mtre_mm"] for r in rows if r["delta"] == delta]
            f.write(f"{delta:>14.1f} {np.mean(sel):>14.3f} {np.median(sel):>16.3f} "
                    f"{success_rate(sel, 2.0):>13.0%} {success_rate(sel, 5.0):>13.0%}\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
