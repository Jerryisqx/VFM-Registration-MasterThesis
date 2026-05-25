#!/usr/bin/env python3
"""
SAM2 zero-shot segmentation on single-plane and dual-plane fluoroscopic images.

Uses your cloned SAM2 repo (facebookresearch/sam2) and the sam2 conda environment.
Runs in "Segment Everything" mode via SAM2AutomaticMaskGenerator (no prompts).
Supports .tif / .tiff (including 16-bit grayscale); converts to RGB for the model.

Usage:
  conda activate sam2
  python sam2_zero_shot_segment.py
  # Process only specific image(s):
  python sam2_zero_shot_segment.py -i walking_96_2.tiff
  python sam2_zero_shot_segment.py --image /path/to/a.tif /path/to/b.tiff

References:
  [1] SAM, DINOv2, Total Segmentator (zero-shot segmentation)
  [2] UniversalSeg, MultiVerseSeg (few-shot)
  SAM 2: https://github.com/facebookresearch/sam2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_tiff_as_rgb(path: str | Path) -> np.ndarray:
    """Load a TIFF (single-plane or dual-plane) as RGB for SAM2."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    arr = np.array(img)

    # Multi-page TIFF (e.g. dual-plane): use first frame or stack
    if arr.ndim == 3 and arr.shape[0] not in (3, 4):
        # Assume (frames, H, W) -> use first frame
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        # (C, H, W) -> (H, W, C)
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]

    if arr.ndim == 2:
        # Grayscale (possibly 16-bit)
        if arr.dtype == np.uint16:
            arr = (arr / (arr.max() or 1) * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 array, got shape {arr.shape}")
    return arr


def run_sam2_repo(
    image_paths: list[Path],
    config_file: str,
    checkpoint_path: Path,
    output_dir: Path,
    device: str,
    points_per_side: int | None,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    box_nms_thresh: float,
) -> None:
    """Run SAM2 zero-shot (Segment Everything) using sam2 package (conda env)."""
    # If we're running from MasterThesis (parent of sam2 repo), "import sam2" would load
    # the repo root (MasterThesis/sam2) and trigger SAM2's "running from parent dir" check.
    # Insert repo root at front of path so "import sam2" resolves to repo_root/sam2 (the real package).
    _script_dir = Path(__file__).resolve().parent
    _repo_root = _script_dir / "sam2"
    if _repo_root.is_dir():
        import sys
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))

    try:
        import sam2
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as e:
        raise RuntimeError(
            "Failed to import sam2. Activate the sam2 conda env and install the repo "
            "(e.g. cd sam2 && pip install -e .)."
        ) from e

    # Resolve checkpoint: relative paths are relative to repo root (parent of sam2 package dir)
    repo_root = Path(sam2.__path__[0]).resolve().parent
    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = repo_root / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Download from https://github.com/facebookresearch/sam2#model-checkpoints "
            "and place in e.g. sam2/checkpoints/"
        )

    print(f"Building SAM2 model on {device} (this may take a moment)...")
    model = build_sam2(config_file=config_file, ckpt_path=str(ckpt), device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        output_mode="binary_mask",
    )
    print("Model ready.\n")

    for image_path in image_paths:
            name = image_path.stem
            rgb = load_tiff_as_rgb(image_path)
            # Save the pre-segmentation input PNG
            out_input = output_dir / f"{name}_input.png"
            Image.fromarray(rgb).save(out_input)
            print(f"  Saved input: {out_input}")
            # generate() expects HWC uint8
            anns = mask_generator.generate(rgb)

            # anns: list of dicts with "segmentation" (H,W bool), "bbox" (xywh), etc.
            masks_list = [ann["segmentation"] for ann in anns]
            boxes_list = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                boxes_list.append([x, y, x + w, y + h])  # xywh -> xyxy

            masks = np.stack(masks_list, axis=0).astype(np.uint8) if masks_list else np.array([])
            boxes = np.array(boxes_list, dtype=np.float64) if boxes_list else np.array([])

            # Overlay
            out_vis = output_dir / f"{name}_sam2_overlay.png"
            if masks.size > 0:
                try:
                    # Overlay: distinct color per mask, blended with image
                    im = rgb.copy().astype(np.float32)
                    n_masks = len(masks)
                    np.random.seed(42)
                    colors_arr = np.random.randint(50, 255, (max(n_masks, 1), 3), dtype=np.uint8)
                    for i in range(n_masks):
                        where = masks[i] > 0
                        color = colors_arr[i % len(colors_arr)].astype(np.float32)
                        im[where] = im[where] * 0.5 + color * 0.5
                    Image.fromarray(im.clip(0, 255).astype(np.uint8)).save(out_vis)
                except Exception:
                    Image.fromarray(rgb).save(out_vis)
            else:
                Image.fromarray(rgb).save(out_vis)

            out_npz = output_dir / f"{name}_sam2_masks.npz"
            np.savez_compressed(out_npz, masks=masks, boxes=boxes)
            print(f"  Saved: {out_vis}, {out_npz}")


# SAM2.1 model variant -> (config_name, checkpoint_path relative to repo root)
SAM2_1_MODELS = {
    "t": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    "s": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
    "b": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
    "l": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM2 zero-shot segmentation (Segment Everything) using your cloned SAM2 repo and sam2 conda env."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="b",
        choices=("t", "s", "b", "l"),
        help="SAM2.1 model: t=tiny, s=small, b=base-plus, l=large (loads corresponding config and checkpoint)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config (optional; default from --model)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint path (optional; default from --model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device to run on: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        nargs="+",
        default=None,
        metavar="PATH",
        help="Image path(s) to segment; if given, only these images are processed (ignore --data-dir / single/dual/extra).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data" / "samples",
        help="Directory for default images when --image is not used",
    )
    parser.add_argument(
        "--single-plane",
        type=str,
        default="C_SUBN_02_dkb_01_009.tif",
        help="Single-plane image filename (used only when --image is not set)",
    )
    parser.add_argument(
        "--dual-plane",
        type=str,
        default="bs_000009.tif",
        help="Dual-plane image filename (used only when --image is not set)",
    )
    parser.add_argument(
        "--extra",
        type=str,
        nargs="*",
        default=[],
        help="Extra image filenames when --image is not set (e.g. walking_96_2.tiff)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <project>/results/seg_sam2_demo)",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="Points per side for automatic mask grid (default: 32)",
    )
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=64,
        help="Points per batch (default: 64)",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.8,
        help="Predicted IoU threshold (default: 0.8)",
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.95,
        help="Stability score threshold (default: 0.95)",
    )
    parser.add_argument(
        "--box-nms-thresh",
        type=float,
        default=0.7,
        help="Box NMS IoU threshold (default: 0.7)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = args.output_dir or (project_root / "results" / "seg_sam2_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        # Direct image path(s): only process these
        image_paths = [Path(p).resolve() for p in args.image]
        missing = [p for p in image_paths if not p.exists()]
        if missing:
            print("Not found:", missing)
            return
        for p in image_paths:
            print(f"Image: {p.name}")
    else:
        # Default: single-plane + dual-plane + extra
        single_path = data_dir / args.single_plane
        dual_path = data_dir / args.dual_plane
        extra_paths = [data_dir / f for f in args.extra]
        image_paths = []
        if single_path.exists():
            image_paths.append(single_path)
            print(f"Single-plane: {single_path.name}")
        else:
            print(f"Skip (not found): {single_path}")
        if dual_path.exists():
            image_paths.append(dual_path)
            print(f"Dual-plane:   {dual_path.name}")
        else:
            print(f"Skip (not found): {dual_path}")
        for p in extra_paths:
            if p.exists():
                image_paths.append(p)
                print(f"Extra:        {p.name}")
            else:
                print(f"Skip (not found): {p}")
        if not image_paths:
            print("No images found. Use --image PATH or check --data-dir and filenames.")
            return

    config_file, checkpoint_path = SAM2_1_MODELS[args.model]
    if args.config is not None:
        config_file = args.config
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = Path(checkpoint_path)

    print(f"Model:        SAM2.1 {args.model} (tiny/small/base-plus/large)")
    print(f"Config:       {config_file}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Output dir:   {output_dir}\n")

    run_sam2_repo(
        image_paths=image_paths,
        config_file=config_file,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=args.device,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
    )
    print("Done.")


if __name__ == "__main__":
    main()
