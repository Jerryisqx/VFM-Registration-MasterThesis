"""
Pose-comparison metrics for the dual-plane registration experiments.

The key metric is mean target registration error (mTRE), computed as the
mean Euclidean distance between a fixed set of fiducial points after
applying a predicted vs ground-truth rigid pose.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from register import pose6_to_tmat


# --- Fiducial generators ---------------------------------------------------

def cube_corner_fiducials(half_size: float = 50.0) -> torch.Tensor:
    """8 corners of a cube centered at origin with half-side `half_size` mm."""
    s = float(half_size)
    return torch.tensor([
        [+s, +s, +s], [+s, +s, -s], [+s, -s, +s], [+s, -s, -s],
        [-s, +s, +s], [-s, +s, -s], [-s, -s, +s], [-s, -s, -s],
    ], dtype=torch.float32)


def random_sphere_fiducials(n: int = 20, radius: float = 50.0,
                            rng: np.random.Generator | None = None) -> torch.Tensor:
    """N points sampled uniformly on a sphere of given radius."""
    rng = rng or np.random.default_rng(0)
    pts = rng.standard_normal((n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * radius
    return torch.tensor(pts, dtype=torch.float32)


# --- Pose transformations --------------------------------------------------

def transform_points(points: torch.Tensor, pose6: torch.Tensor) -> torch.Tensor:
    """Apply a (6,) pose [tx, ty, tz, rx, ry, rz deg] to (N, 3) points.

    Returns (N, 3) transformed points.
    """
    tmat = pose6_to_tmat(pose6.to(points.dtype))  # (4, 4)
    R = tmat[:3, :3]
    t = tmat[:3, 3]
    return points @ R.T + t


# --- Metrics ---------------------------------------------------------------

def mtre(pose_pred: torch.Tensor | np.ndarray,
         pose_gt: torch.Tensor | np.ndarray,
         fiducials: torch.Tensor) -> float:
    """Mean target registration error in millimetres.

    mTRE = mean over fiducials of || T_pred(p) - T_gt(p) ||_2.
    Both poses are 6-vectors [tx, ty, tz, rx, ry, rz].
    """
    pose_pred = torch.as_tensor(pose_pred, dtype=torch.float32)
    pose_gt = torch.as_tensor(pose_gt, dtype=torch.float32)
    p_pred = transform_points(fiducials, pose_pred)
    p_gt = transform_points(fiducials, pose_gt)
    dist = torch.linalg.norm(p_pred - p_gt, dim=1)
    return float(dist.mean().item())


def translation_error(pose_pred: torch.Tensor | np.ndarray,
                      pose_gt: torch.Tensor | np.ndarray) -> float:
    """Magnitude of translation error vector in mm."""
    pose_pred = np.asarray(pose_pred, dtype=np.float32)
    pose_gt = np.asarray(pose_gt, dtype=np.float32)
    return float(np.linalg.norm(pose_pred[:3] - pose_gt[:3]))


def rotation_error_deg(pose_pred: torch.Tensor | np.ndarray,
                       pose_gt: torch.Tensor | np.ndarray) -> float:
    """Angular error between predicted and GT rotation, in degrees."""
    from scipy.spatial.transform import Rotation
    pose_pred = np.asarray(pose_pred, dtype=np.float32)
    pose_gt = np.asarray(pose_gt, dtype=np.float32)
    R_pred = Rotation.from_euler("XYZ", pose_pred[3:], degrees=True).as_matrix()
    R_gt = Rotation.from_euler("XYZ", pose_gt[3:], degrees=True).as_matrix()
    R_diff = R_pred @ R_gt.T
    cos_theta = (np.trace(R_diff) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def success_rate(mtres: Sequence[float], threshold_mm: float) -> float:
    """Fraction of mTRE values below threshold."""
    arr = np.asarray(mtres, dtype=np.float32)
    return float((arr < threshold_mm).mean())


# --- Self-test -------------------------------------------------------------

def _self_test():
    fids = cube_corner_fiducials(50.0)
    gt = torch.tensor([0.0, 0.0, -300.0, 0.0, 0.0, 0.0])

    # Identity prediction → mTRE 0
    pred_perfect = gt.clone()
    e = mtre(pred_perfect, gt, fids)
    print(f"perfect prediction: mTRE = {e:.4f} mm (expect 0)")
    assert e < 1e-4

    # Pure 5 mm translation in x → all fiducials shift 5 mm → mTRE = 5
    pred_5mm = gt.clone()
    pred_5mm[0] += 5.0
    e = mtre(pred_5mm, gt, fids)
    print(f"+5 mm in x:         mTRE = {e:.4f} mm (expect 5.0)")
    assert abs(e - 5.0) < 1e-3

    # 10 deg about z → corner 50 mm out shifts by ~50*sin(10) = 8.68 mm
    pred_rot = gt.clone()
    pred_rot[5] += 10.0  # rotation about Z (last in XYZ Euler) -- but our conv is XYZ
    e = mtre(pred_rot, gt, fids)
    print(f"+10 deg in last:    mTRE = {e:.4f} mm (expect ~8.7)")

    # Trans-only error
    print(f"translation_error (5mm in x): {translation_error(pred_5mm, gt):.4f} mm")
    print(f"rotation_error    (10 deg):  {rotation_error_deg(pred_rot, gt):.4f} deg")

    # Success rate
    sample = [0.5, 1.2, 2.3, 4.9, 6.1, 8.0, 12.0]
    print(f"success @ 2 mm: {success_rate(sample, 2.0):.2f}, "
          f"@ 5 mm: {success_rate(sample, 5.0):.2f}")


if __name__ == "__main__":
    _self_test()
