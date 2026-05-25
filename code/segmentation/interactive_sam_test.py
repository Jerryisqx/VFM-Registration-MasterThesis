"""
Interactive SAM2 GUI -- adapted from test_sam.py
Interactive segmentation using a fine-tuned MedSAM2 checkpoint.

Controls:
  left click    -> positive point (foreground)
  right click   -> negative point (background)
  1             -> switch to femur mode (clears points)
  2             -> switch to tibia mode (clears points)
  3             -> switch to patella mode (clears points)
  4             -> switch to leg mode (clears points)
  5/6/7/8/9     -> switch preprocessing mode (raw/denoise/clahe/sharpen/all)
  r             -> clear current points and mask
  s             -> save current mask and overlay next to the image
  q             -> quit
"""

import os
import sys
import time

import cv2
import numpy as np
import torch
import tifffile
from skimage import exposure

# -- Hydra init (must run before build_sam2) --------------------------------
from hydra import initialize_config_module
try:
    initialize_config_module("sam2", version_base="1.2")
except Exception:
    pass

# Auto-locate directories (this script lives in code/segmentation/, the sam2 library in code/sam2_lib/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
SAM2_REPO_ROOT = os.path.join(_PROJECT_ROOT, "code", "sam2_lib")
if os.path.isdir(SAM2_REPO_ROOT) and SAM2_REPO_ROOT not in sys.path:
    sys.path.insert(0, SAM2_REPO_ROOT)

MEDSAM2_ROOT = os.path.join(_PROJECT_ROOT, "code", "MedSAM2")
if os.path.isdir(MEDSAM2_ROOT) and MEDSAM2_ROOT not in sys.path:
    sys.path.insert(0, MEDSAM2_ROOT)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ──────────────────────────────────────────────────────────────────────────────
# User config (edit these two lines)
# ──────────────────────────────────────────────────────────────────────────────
SAM2_CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "medsam2_finetune", "checkpoint_robust_v1.pt")

IMG_PATH = os.path.join(_PROJECT_ROOT, "data", "samples", "bs_000009.tif")
# ──────────────────────────────────────────────────────────────────────────────

# Hydra takes a config name relative to the sam2 package config path; an absolute path will not work
# Options: configs/sam2.1/sam2.1_hiera_t.yaml (tiny), sam2.1_hiera_s.yaml (small), sam2.1_hiera_b+.yaml (base+), sam2.1_hiera_l.yaml (large)
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_image_bgr(path: str) -> np.ndarray:
    """Read tif/tiff/png/jpg -> BGR uint8."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[..., :3]
        elif arr.ndim == 3:
            arr = arr[..., 0]
        # normalize to uint8
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
        arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2 cannot read: {path}")
        return img


def overlay_mask_bgr(img_bgr, mask01, color_bgr=(0, 0, 255), alpha=0.40):
    mask   = (mask01 > 0).astype(np.uint8)
    overlay = img_bgr.copy()
    overlay[mask == 1] = color_bgr
    return cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)


def draw_points(img_bgr, points_xy, labels):
    out = img_bgr.copy()
    for (x, y), lb in zip(points_xy, labels):
        color = (0, 255, 0) if lb == 1 else (0, 0, 255)
        cv2.circle(out, (int(x), int(y)), 5, color, -1)
        cv2.circle(out, (int(x), int(y)), 7, (255, 255, 255), 1)
    return out


# label → overlay colour (BGR)
LABEL_COLORS = {
    "femur":   (255, 80,  80),   # blue-ish
    "tibia":   (80,  160, 255),  # orange-ish
    "patella": (80,  255, 80),   # green
    "leg":     (200, 80,  200),  # purple
}


def preprocess_image(img, mode):
    if mode == 0:
        return img.copy()
    elif mode == 1:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    elif mode == 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = exposure.equalize_adapthist(gray, clip_limit=0.03)
        gray = (gray * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == 3:
        return cv2.GaussianBlur(img, (5, 5), 1.0)
    elif mode == 4:
        out  = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        gray = exposure.equalize_adapthist(gray, clip_limit=0.03)
        gray = (gray * 255).astype(np.uint8)
        out  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.GaussianBlur(out, (5, 5), 1.0)
    return img.copy()


# ── GUI class ─────────────────────────────────────────────────────────────────

class SAM2InteractiveGUI:
    def __init__(self, img_path: str, ckpt: str, cfg: str = DEFAULT_CFG):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(img_path)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)

        self.img_path = img_path

        # ── load image ────────────────────────────────────────────────────────
        self.img_raw = load_image_bgr(img_path)
        self.preprocess_mode = 0
        self.img_bgr = self.img_raw.copy()
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)

        # ── load SAM2 ─────────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] device: {self.device}")
        print(f"[INFO] loading checkpoint: {ckpt}")
        model = build_sam2(cfg, ckpt, device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        self.predictor.set_image(self.img_rgb)
        print("[INFO] model ready.")

        # ── state ─────────────────────────────────────────────────────────────
        self.points_xy    : list = []
        self.labels       : list = []
        self.cur_mask01           = None
        self.cur_score            = None
        self.current_label        = "femur"
        self.masks                = {k: None for k in LABEL_COLORS}
        self.status_msg           = ""
        self.status_time          = 0.0

        # ── window ────────────────────────────────────────────────────────────
        self.win_name = "SAM2 Interactive"
        cv2.namedWindow(self.win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win_name, self._mouse_cb)
        self._render()

    # ── internal ──────────────────────────────────────────────────────────────

    def _predict(self):
        if not self.points_xy:
            return
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(self.points_xy, dtype=np.float32),
                point_labels=np.array(self.labels,    dtype=np.int32),
                multimask_output=False,
            )
        self.cur_mask01 = masks[0].astype(np.uint8)
        self.cur_score  = float(scores[0])
        self.masks[self.current_label] = self.cur_mask01

    def _render(self):
        base = self.img_bgr.copy()

        # draw all saved masks (faint)
        for label, mask in self.masks.items():
            if mask is not None and label != self.current_label:
                base = overlay_mask_bgr(base, mask,
                                        color_bgr=LABEL_COLORS[label], alpha=0.25)

        # draw current mask (bold)
        if self.cur_mask01 is not None:
            base = overlay_mask_bgr(base, self.cur_mask01,
                                    color_bgr=LABEL_COLORS[self.current_label],
                                    alpha=0.45)

        base = draw_points(base, self.points_xy, self.labels)

        score_str  = f"{self.cur_score:.3f}" if self.cur_score is not None else "—"
        mode_names = ["raw", "denoise", "clahe", "sharpen", "all"]
        cv2.putText(base, f"label : {self.current_label}",
                    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(base, f"score : {score_str}",
                    (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0),  2)
        cv2.putText(base, f"view  : {mode_names[self.preprocess_mode]}",
                    (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0),  2)
        cv2.putText(base, f"points: {len(self.points_xy)}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if self.status_msg:
            if time.time() - self.status_time < 2.5:
                cv2.putText(base, self.status_msg,
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                self.status_msg = ""

        cv2.imshow(self.win_name, base)

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points_xy.append([x, y])
            self.labels.append(1)
            self._predict()
            self._render()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points_xy.append([x, y])
            self.labels.append(0)
            self._predict()
            self._render()

    # ── public ────────────────────────────────────────────────────────────────

    def reset(self):
        self.points_xy  = []
        self.labels     = []
        self.cur_mask01 = None
        self.cur_score  = None
        self._render()

    def update_preprocess(self):
        self.img_bgr = preprocess_image(self.img_raw, self.preprocess_mode)
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        # SAM2: re-embed the new image
        self.predictor.set_image(self.img_rgb)

    def save_outputs(self):
        if self.cur_mask01 is None:
            print("[WARN] No mask — click some points first.")
            return

        out_dir  = os.path.dirname(self.img_path)
        stem     = os.path.splitext(os.path.basename(self.img_path))[0]
        lbl      = self.current_label

        mask_u8  = (self.cur_mask01 > 0).astype(np.uint8) * 255
        mask_p   = os.path.join(out_dir, f"{stem}_sam2_mask_{lbl}.png")
        cv2.imwrite(mask_p, mask_u8)

        overlay  = overlay_mask_bgr(self.img_bgr, self.cur_mask01,
                                     LABEL_COLORS[lbl], alpha=0.40)
        overlay  = draw_points(overlay, self.points_xy, self.labels)
        over_p   = os.path.join(out_dir, f"{stem}_sam2_overlay_{lbl}.png")
        cv2.imwrite(over_p, overlay)

        print(f"[INFO] mask    → {mask_p}")
        print(f"[INFO] overlay → {over_p}")
        self.status_msg  = f"Saved {lbl}"
        self.status_time = time.time()
        self._render()

    def loop(self):
        print("\n  1-4 : switch label (femur/tibia/patella/leg)")
        print("  5-9 : preprocessing mode")
        print("  r   : reset points")
        print("  s   : save mask + overlay")
        print("  q   : quit\n")

        while True:
            if cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(20) & 0xFF
            if   key == ord('q'): break
            elif key == ord('r'): self.reset()
            elif key == ord('s'): self.save_outputs()
            elif key == ord('1'): self.current_label = "femur";   self.reset()
            elif key == ord('2'): self.current_label = "tibia";   self.reset()
            elif key == ord('3'): self.current_label = "patella"; self.reset()
            elif key == ord('4'): self.current_label = "leg";     self.reset()
            elif key == ord('5'): self.preprocess_mode = 0; self.update_preprocess(); self.reset()
            elif key == ord('6'): self.preprocess_mode = 1; self.update_preprocess(); self.reset()
            elif key == ord('7'): self.preprocess_mode = 2; self.update_preprocess(); self.reset()
            elif key == ord('8'): self.preprocess_mode = 3; self.update_preprocess(); self.reset()
            elif key == ord('9'): self.preprocess_mode = 4; self.update_preprocess(); self.reset()

        cv2.destroyAllWindows()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  default=SAM2_CKPT)
    ap.add_argument("--image", default=IMG_PATH)
    ap.add_argument("--cfg",   default=DEFAULT_CFG)
    args = ap.parse_args()

    app = SAM2InteractiveGUI(args.image, args.ckpt, args.cfg)
    app.loop()
