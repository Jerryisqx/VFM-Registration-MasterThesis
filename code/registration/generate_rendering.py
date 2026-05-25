import os
import numpy as np
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from renderers import simple_renderer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tmat(translation=(0, 0, -900), rotation_deg=(0, 0, 0)):
    """
    Build a 4x4 transformation matrix from translation and Euler angles.

    Parameters
    ----------
    translation : tuple
        (tx, ty, tz)
    rotation_deg : tuple
        Euler angles in degrees, interpreted with sequence 'ZXY'.

    Returns
    -------
    torch.Tensor
        Shape: (1, 4, 4)
    """
    r = R.from_euler('ZXY', rotation_deg, degrees=True)
    rot_mat = r.as_matrix()

    tmat = np.eye(4, dtype=np.float32)
    tmat[:3, :3] = rot_mat
    tmat[:3, 3] = translation

    return torch.tensor(tmat, dtype=torch.float32).unsqueeze(0).to(device)


def show_image(image, title="Image", cmap="gray", block=True):
    """
    Keep current visualization behavior for generate_rendering().
    """
    plt.figure(figsize=(5, 5))
    if cmap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.01)


def _render_with_renderer(renderer, tmat, renderer_type, img_width, img_height):
    """
    Render one image using an already-created renderer.
    """
    if renderer_type.upper() == 'STL':
        return renderer.generate_virtual_xray_STL(tmat, tmat, img_width, img_height).squeeze()
    else:
        return renderer.generate_virtual_xray_CT(tmat, tmat, img_width, img_height).squeeze()


def generate_rendering(anatomy_path,
                       translation=(0, 0, -500),
                       rotation_deg=(0, 0, 0),
                       renderer_type='STL',
                       img_width=512,
                       img_height=512,
                       save_dir=None,
                       save_prefix="render",
                       visualize=False,
                       show_mode='both',
                       block=True):
    """
    Generate rendering images from BS and/or FS cameras on demand.

    Parameters
    ----------
    anatomy_path : str
        Path to STL/CT file.
    translation : tuple
        Translation vector.
    rotation_deg : tuple
        Euler angles in degrees, interpreted in 'ZXY' order.
    renderer_type : str
        'STL' or 'CT'
    img_width : int
        Render width.
    img_height : int
        Render height.
    save_dir : str or None
        Directory to save images.
    save_prefix : str
        Prefix for saved file names.
    visualize : bool
        Whether to display rendered images.
    show_mode : str
        'both', 'bs', or 'fs'
    block : bool
        Whether matplotlib display should block.

    Returns
    -------
    dict
        {"bs": img_bs_or_None, "fs": img_fs_or_None}
    """
    show_mode = show_mode.lower()
    if show_mode not in ['both', 'bs', 'fs']:
        raise ValueError("show_mode must be one of: 'both', 'bs', 'fs'")

    tmat = build_tmat(translation=translation, rotation_deg=rotation_deg)

    img_bs = None
    img_fs = None

    if show_mode in ['both', 'bs']:
        renderer_bs = simple_renderer(
            femur_path=anatomy_path,
            tibia_path=None,
            renderer_type=renderer_type,
            plane='bs',
            cal_pixel_size=0.225,
            cal_focal_length=1720.0,
            screen_size=1600,
        )
        img_bs = _render_with_renderer(
            renderer_bs, tmat, renderer_type, img_width, img_height)

    if show_mode in ['both', 'fs']:
        renderer_fs = simple_renderer(
            femur_path=anatomy_path,
            tibia_path=None,
            renderer_type=renderer_type,
            plane='fs',
            cal_pixel_size=0.225,
            cal_focal_length=1720.0,
            screen_size=1600,
        )
        img_fs = _render_with_renderer(
            renderer_fs, tmat, renderer_type, img_width, img_height)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if img_bs is not None:
            bs_path = os.path.join(save_dir, f"{save_prefix}_bs.png")
            cv2.imwrite(bs_path, (img_bs * 255).astype(np.uint8))
            print(f"Saved BS rendering to: {bs_path}")

        if img_fs is not None:
            fs_path = os.path.join(save_dir, f"{save_prefix}_fs.png")
            cv2.imwrite(fs_path, (img_fs * 255).astype(np.uint8))
            print(f"Saved FS rendering to: {fs_path}")

    # Keep your current visualization behavior here
    if visualize:
        if img_bs is not None:
            show_image(img_bs, title=f"{save_prefix} - BS", block=block)
        if img_fs is not None:
            show_image(img_fs, title=f"{save_prefix} - FS", block=block)

    return {
        "bs": img_bs,
        "fs": img_fs
    }


if __name__ == "__main__":
    # Example 1: generate rendering from one or both cameras
    generate_rendering(
        anatomy_path=r"D:\Martin_rematch\DPZM_05_Femur.stl", # replace with your STL/CT path
        translation=(0, 0, 400), # roughly here, but you can adjust to see different views
        rotation_deg=(0, 0, 0),   # interpreted as (Z, X, Y) in Euler angle
        renderer_type='STL',
        img_width=512,
        img_height=512,
        save_dir="render_examples",
        save_prefix="sample_pose",
        visualize=True,
        show_mode='both',   # 'both', 'bs', or 'fs'
        block=True,
    )
