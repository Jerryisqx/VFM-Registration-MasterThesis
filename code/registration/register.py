"""
Dual-plane 2D/3D rigid registration on top of dupla_renderers.

Extends the single-step PoC `Pytorch3dPoseRefiner` (in
`dupla_renderers/pytorch3d/test_cases/test_differentiability.py`) into a
full optimisation loop with:

  - dual-plane (BS + FS) loss combination
  - configurable loss (MSE, NCC) and per-plane weighting
  - optional mask-weighted similarity (for SAM-driven masked variant)
  - Adam optimiser with relative-loss convergence criterion
  - clean result object (final pose, loss history, convergence flag)

This is the core thesis contribution -- the renderer infrastructure is
provided by Veriserum; the registration *algorithm* is implemented here.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from dupla_renderers.pytorch3d import (
    AnatomyCT,
    AnatomySTL,
    Camera,
    CTRenderer,
    Scene,
    STLRenderer,
)
from pytorch3d.transforms import euler_angles_to_matrix


# --- Pose helpers ---------------------------------------------------------

def pose6_to_tmat(pose6: torch.Tensor, conv: Tuple[str, str, str] = ("X", "Y", "Z")) -> torch.Tensor:
    """Convert a (6,) pose vector [tx, ty, tz, rx, ry, rz] (degrees) to (4, 4) tmat.

    Differentiable -- gradients flow through both translation and rotation parts.
    """
    assert pose6.shape == (6,), f"expected (6,), got {tuple(pose6.shape)}"
    pad_t = nn.ZeroPad2d((3, 0, 0, 1))
    pad_r = nn.ZeroPad2d((0, 1, 0, 1))
    rot = euler_angles_to_matrix(torch.deg2rad(pose6[3:]), conv)  # (3, 3)
    tmat = pad_t(pose6[0:3, None]) + pad_r(rot)  # (4, 4)
    tmat = tmat.clone()
    tmat[3, 3] = 1.0
    return tmat


# --- Renderer setup -------------------------------------------------------

def make_camera(name: str,
                center: Tuple[float, float, float],
                normal: Tuple[float, float, float],
                vertical: Tuple[float, float, float],
                screen_size_mm: float,
                principal_h: float,
                principal_v: float,
                focal_length: float) -> Camera:
    return Camera(
        name,
        screen_center_poses=center,
        screen_normals=normal,
        screen_verticals=vertical,
        screen_sizes_h=screen_size_mm,
        screen_sizes_v=screen_size_mm,
        principal_points_h=principal_h,
        principal_points_v=principal_v,
        focal_lengths=focal_length,
    )


# Defaults match the kneefit/example_04 calibration roughly:
#   screen_size = 1000 px * 0.2876 mm/px = 287.6 mm physical intensifier
#   focal_length = 972 mm (kneefit) / 1720 mm (rendering script default)
DEFAULT_SCREEN_SIZE_MM = 287.6
DEFAULT_FOCAL_LENGTH_MM = 972.0


def default_dual_plane_cameras(screen_size_mm: float = DEFAULT_SCREEN_SIZE_MM,
                               focal_length: float = DEFAULT_FOCAL_LENGTH_MM
                               ) -> Tuple[Camera, Camera]:
    """Standard BS + FS pair matching the dupla_renderers / kneefit convention.

    BS: origin, looking +Z (normal +Z, source at z=+focal_length).
    FS: at (333, 0, 233), normal rotated 70 deg about -Y from BS.
    """
    from scipy.spatial.transform import Rotation
    fs_normal = Rotation.from_rotvec(np.deg2rad(70) * np.array([0, -1, 0])).as_matrix() @ np.array([0.0, 0.0, 1.0])
    bs = make_camera("camera_bs", (0, 0, 0), (0, 0, 1), (0, 1, 0),
                     screen_size_mm, 0.0, 0.0, focal_length)
    fs = make_camera("camera_fs", (333.0, 0, 233), tuple(fs_normal), (0, 1, 0),
                     screen_size_mm, 0.0, 0.0, focal_length)
    return bs, fs


# --- Loss functions -------------------------------------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.to(pred.dtype)
    if m.sum() < 1:
        return F.mse_loss(pred, target)
    return ((pred - target) ** 2 * m).sum() / m.sum()


def ncc_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Negative normalized cross-correlation. Lower = better match.

    Implemented inline to avoid the torch_similarity dependency.
    """
    if mask is not None:
        m = mask.to(pred.dtype)
        n = m.sum().clamp(min=1)
        p_mean = (pred * m).sum() / n
        t_mean = (target * m).sum() / n
        p_d = (pred - p_mean) * m
        t_d = (target - t_mean) * m
    else:
        p_mean = pred.mean()
        t_mean = target.mean()
        p_d = pred - p_mean
        t_d = target - t_mean
    num = (p_d * t_d).sum()
    den = torch.sqrt((p_d ** 2).sum() * (t_d ** 2).sum() + 1e-8)
    return -num / den


LOSSES = {"mse": mse_loss, "ncc": ncc_loss}


# --- Result dataclass -----------------------------------------------------

@dataclass
class RegisterResult:
    final_pose: np.ndarray            # shape (6,)
    loss_history: List[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    final_loss: float = float("inf")
    elapsed_seconds: float = 0.0
    init_pose: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "final_pose": self.final_pose.tolist(),
            "init_pose": self.init_pose.tolist() if self.init_pose is not None else None,
            "loss_history": self.loss_history,
            "converged": self.converged,
            "iterations": self.iterations,
            "final_loss": self.final_loss,
            "elapsed_seconds": self.elapsed_seconds,
        }


# --- Main registration class ---------------------------------------------

class DualPlaneRegister:
    """Differentiable dual-plane 2D/3D rigid registration."""

    def __init__(self,
                 anatomy_path: str,
                 modality: str = "STL",
                 device: str = "cpu",
                 img_size: int = 256,
                 screen_size_mm: float = DEFAULT_SCREEN_SIZE_MM,
                 focal_length: float = DEFAULT_FOCAL_LENGTH_MM,
                 cameras: Optional[Tuple[Camera, Camera]] = None):
        self.device = torch.device(device)
        self.modality = modality.upper()
        self.img_size = img_size

        # Load anatomy
        if self.modality == "STL":
            self.anatomy = AnatomySTL.load_data(anatomy_path).to(self.device)
            self.renderer = STLRenderer(device=self.device)
        elif self.modality == "CT":
            self.anatomy = AnatomyCT.load_data(anatomy_path).to(self.device)
            self.renderer = CTRenderer(device=self.device)
        else:
            raise ValueError(f"modality must be STL or CT, got {self.modality}")

        # Cameras
        if cameras is None:
            self.cam_bs, self.cam_fs = default_dual_plane_cameras(screen_size_mm, focal_length)
        else:
            self.cam_bs, self.cam_fs = cameras
        self.cam_bs = self.cam_bs.to(self.device)
        self.cam_fs = self.cam_fs.to(self.device)

        # Bind a scene with both cameras
        self.scene = Scene()
        self.scene.add_anatomies(self.anatomy)
        self.scene.add_cameras(self.cam_bs)
        self.scene.add_cameras(self.cam_fs)
        self.renderer.bind_scene(self.scene)

    def _set_pose(self, pose6: torch.Tensor) -> None:
        tmat = pose6_to_tmat(pose6).to(self.device)
        self.anatomy.set_model_matrix(tmat[None], is_yours=True)

    def render(self, pose6: torch.Tensor, plane: str = "bs") -> torch.Tensor:
        self._set_pose(pose6)
        cam_index = 0 if plane.lower() == "bs" else 1
        kw = dict(cam_index=cam_index,
                  width_pixels_num=self.img_size,
                  height_pixels_num=self.img_size)
        if self.modality == "CT":
            kw["binary"] = False
        return self.renderer.render(**kw)[0]  # (H, W)

    def render_dual(self, pose6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render BOTH planes for the same pose. Returns (img_bs, img_fs)."""
        self._set_pose(pose6)
        kw = dict(width_pixels_num=self.img_size, height_pixels_num=self.img_size)
        if self.modality == "CT":
            kw["binary"] = False
        img_bs = self.renderer.render(cam_index=0, **kw)[0]
        img_fs = self.renderer.render(cam_index=1, **kw)[0]
        return img_bs, img_fs

    def register(self,
                 target_bs: torch.Tensor,
                 target_fs: Optional[torch.Tensor],
                 init_pose: torch.Tensor,
                 mask_bs: Optional[torch.Tensor] = None,
                 mask_fs: Optional[torch.Tensor] = None,
                 alpha: float = 0.5,
                 loss: str = "mse",
                 max_iter: int = 200,
                 lr: float = 1.0,
                 conv_tol: float = 1e-5,
                 conv_window: int = 10,
                 verbose: bool = False) -> RegisterResult:
        """Run gradient-based registration.

        Parameters
        ----------
        target_bs : (H, W) target image for BS plane (required)
        target_fs : (H, W) target image for FS plane (None to disable dual-plane)
        init_pose : (6,) initial pose vector [tx, ty, tz, rx, ry, rz] (degrees)
        mask_bs, mask_fs : optional (H, W) binary masks to weight per-plane loss
        alpha : weight on BS loss; FS loss gets (1 - alpha). Ignored when target_fs=None.
        loss : 'mse' or 'ncc'
        max_iter : maximum optimisation steps
        lr : Adam learning rate (translation; rotation uses lr * 0.1 internally)
        conv_tol : if last `conv_window` losses change by less than this fraction, declare converged
        conv_window : number of consecutive small-change steps required
        """
        loss_fn = LOSSES[loss.lower()]
        target_bs = target_bs.detach().to(self.device)
        if target_fs is not None:
            target_fs = target_fs.detach().to(self.device)
        if mask_bs is not None:
            mask_bs = mask_bs.detach().to(self.device).float()
        if mask_fs is not None:
            mask_fs = mask_fs.detach().to(self.device).float()

        init_np = init_pose.detach().cpu().numpy().copy()

        # Split learnable params for differentiated learning rates
        trans = nn.Parameter(init_pose[0:3].clone().to(self.device).float(), requires_grad=True)
        rot = nn.Parameter(init_pose[3:6].clone().to(self.device).float(), requires_grad=True)
        optim = torch.optim.Adam([
            {"params": [trans], "lr": lr},
            {"params": [rot],   "lr": lr * 0.1},
        ])

        loss_hist: List[float] = []
        converged = False
        t0 = time.time()

        for step in range(max_iter):
            optim.zero_grad()
            pose6 = torch.cat([trans, rot], dim=0)
            if target_fs is not None:
                img_bs, img_fs = self.render_dual(pose6)
                l_bs = loss_fn(img_bs, target_bs, mask_bs)
                l_fs = loss_fn(img_fs, target_fs, mask_fs)
                total = alpha * l_bs + (1 - alpha) * l_fs
            else:
                img_bs = self.render(pose6, plane="bs")
                total = loss_fn(img_bs, target_bs, mask_bs)
            total.backward()
            optim.step()

            loss_hist.append(float(total.item()))
            if verbose and step % 10 == 0:
                print(f"  step {step:3d}: loss={total.item():.5f}  pose={pose6.detach().cpu().numpy()}")

            # Convergence: relative change over window < tol
            if len(loss_hist) >= conv_window + 1:
                recent = loss_hist[-(conv_window + 1):]
                rel = [abs(recent[i + 1] - recent[i]) / (abs(recent[i]) + 1e-8) for i in range(conv_window)]
                if max(rel) < conv_tol:
                    converged = True
                    break

        elapsed = time.time() - t0
        final = torch.cat([trans.detach(), rot.detach()], dim=0).cpu().numpy()
        return RegisterResult(
            final_pose=final,
            init_pose=init_np,
            loss_history=loss_hist,
            converged=converged,
            iterations=len(loss_hist),
            final_loss=loss_hist[-1] if loss_hist else float("inf"),
            elapsed_seconds=elapsed,
        )


# --- Self-test (smoke) ----------------------------------------------------

def _self_test():
    """Synthetic smoke test: register a femur from a known perturbed pose."""
    test_files = THIS_DIR / "dupla_renderers" / "test_files"
    reg = DualPlaneRegister(
        anatomy_path=str(test_files / "SUBN_02_Femur_RE_Surface.stl"),
        modality="STL",
        img_size=128,  # small for speed
    )

    true_pose = torch.tensor([0.0, 0.0, -300.0, 0.0, 0.0, 0.0])
    target = reg.render(true_pose, plane="bs").detach()

    init_pose = true_pose.clone() + torch.tensor([5.0, -3.0, 7.0, 2.0, -1.5, 1.0])
    print(f"True pose:  {true_pose.numpy()}")
    print(f"Init pose:  {init_pose.numpy()}")
    print(f"Init delta: {(init_pose - true_pose).numpy()}")

    result = reg.register(
        target_bs=target,
        target_fs=None,            # single-plane for self-test (smoke)
        init_pose=init_pose,
        max_iter=80,
        lr=1.0,
        loss="mse",
        verbose=True,
    )
    print()
    print(f"Final pose:  {result.final_pose}")
    print(f"Final delta: {result.final_pose - true_pose.numpy()}")
    print(f"Iterations: {result.iterations}, converged={result.converged}, loss={result.final_loss:.5f}")
    print(f"Elapsed:    {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    _self_test()
