#!/usr/bin/env python3
"""
SAM3 zero-shot segmentation on single-plane and dual-plane fluoroscopic images.

Uses Meta SAM 3 (Segment Anything with Concepts) for text-prompted segmentation.
Supports .tif / .tiff (including 16-bit grayscale); converts to RGB for the model.

References:
  [1] SAM, DINOv2, Total Segmentator (zero-shot segmentation)
  [2] UniversalSeg, MultiVerseSeg (few-shot)
  SAM 3: https://github.com/facebookresearch/sam3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_tiff_as_rgb(path: str | Path) -> np.ndarray:
    """Load a TIFF (single-plane or dual-plane) as RGB for SAM3."""
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


def run_ultralytics_sam3(
    image_paths: list[Path],
    text_prompts: list[str],
    model_path: str | Path,
    output_dir: Path,
    conf: float,
    half: bool,
) -> None:
    """Run SAM3 zero-shot segmentation via Ultralytics."""
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError:
        raise ImportError(
            "Ultralytics is required. Install with: pip install -U ultralytics"
        )

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=str(model_path),
        half=half,
        save=True,
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    for image_path in image_paths:
        name = image_path.stem
        rgb = load_tiff_as_rgb(image_path)
        # Ultralytics predictor expects file path or numpy BGR (OpenCV order)
        # Save a temporary RGB then load; or pass numpy in HWC RGB
        pil_image = Image.fromarray(rgb)
        temp_path = output_dir / f"{name}_input.png"
        pil_image.save(temp_path)
        predictor.set_image(str(temp_path))

        results = predictor(text=text_prompts)
        # results: can be list of Results (one per prompt) or single Results
        res_list = results if isinstance(results, (list, tuple)) else [results]
        all_masks, all_boxes = [], []
        for r in res_list:
            if getattr(r, "masks", None) is not None:
                m = r.masks.data.cpu().numpy()
                all_masks.append(m)
            if getattr(r, "boxes", None) is not None and hasattr(r.boxes, "xyxy"):
                all_boxes.append(r.boxes.xyxy.cpu().numpy())
        masks = np.concatenate(all_masks, axis=0) if all_masks else None
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else None

        # Save overlay
        try:
            from ultralytics.utils.plotting import Annotator, colors
        except ImportError:
            Annotator, colors = None, None

        out_vis = output_dir / f"{name}_sam3_overlay.png"
        if Annotator is not None and masks is not None and masks.size > 0:
            import cv2
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            annotator = Annotator(im, pil=False)
            n_masks = len(masks) if masks.ndim == 3 else 0
            annotator.masks(masks, [colors(i, True) for i in range(n_masks)])
            cv2.imwrite(str(out_vis), annotator.result())
        else:
            pil_image.save(out_vis)

        # Save raw masks as NPZ for downstream use
        out_npz = output_dir / f"{name}_sam3_masks.npz"
        np.savez_compressed(
            out_npz,
            masks=np.asarray(masks) if masks is not None else np.array([]),
            boxes=np.asarray(boxes) if boxes is not None else np.array([]),
            prompts=np.array(text_prompts, dtype=object),
        )
        print(f"  Saved: {out_vis}, {out_npz}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM3 zero-shot segmentation on single-plane and dual-plane fluoroscopic images."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data" / "samples",
        help="Directory containing .tif/.tiff images",
    )
    parser.add_argument(
        "--single-plane",
        type=str,
        default="C_SUBN_02_dkb_01_009.tif",
        help="Single-plane fluoroscopic image filename",
    )
    parser.add_argument(
        "--dual-plane",
        type=str,
        default="bs_000009.tif",
        help="Dual-plane fluoroscopic image filename",
    )
    parser.add_argument(
        "--extra",
        type=str,
        nargs="*",
        default=[],
        help="Extra image filenames (e.g. walking_96_2.tiff)",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        default=["bone", "knee", "leg"],
        help="Text prompts for zero-shot segmentation (e.g. bone knee leg)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("sam3.pt"),
        help="Path to SAM3 weights (sam3.pt). Download from https://huggingface.co/facebook/sam3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <project>/results/seg_sam3_demo)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 (use if you see errors on CPU or older GPU)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = args.output_dir or (project_root / "results" / "seg_sam3_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

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
        print("No images found. Check --data-dir and filenames.")
        return

    if not args.model.exists():
        print(
            f"Model not found: {args.model}\n"
            "Download SAM3 weights from https://huggingface.co/facebook/sam3\n"
            "Place sam3.pt in this directory or set --model /path/to/sam3.pt"
        )
        return

    print(f"Text prompts: {args.text}")
    print(f"Output dir:   {output_dir}\n")

    run_ultralytics_sam3(
        image_paths=image_paths,
        text_prompts=args.text,
        model_path=args.model,
        output_dir=output_dir,
        conf=args.conf,
        half=not args.no_half,
    )
    print("Done.")


if __name__ == "__main__":
    main()
