import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import tifffile
from PIL import Image, ImageDraw, ImageFont


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SAM3_SRC_ROOT = os.path.join(THIS_DIR, "sam3_code")
if os.path.isdir(SAM3_SRC_ROOT) and SAM3_SRC_ROOT not in sys.path:
    sys.path.insert(0, SAM3_SRC_ROOT)

from sam3 import build_sam3_image_model  # noqa: E402


# Defaults are resolved relative to this file's directory so the GUI works on
# any host without per-machine path edits. Override at the command line if needed.
DEFAULT_CKPT = os.path.join(THIS_DIR, "weights", "sam3.pt")
DEFAULT_BPE = os.path.join(THIS_DIR, "sam3_code", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
# Falls back to the project's test image folder, then to data/data/test for repo users.
_REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
_CANDIDATE_IMAGE_DIRS = [
    os.path.join(_REPO_ROOT, "data", "data", "test"),
    os.path.join(THIS_DIR, "..", "02_view_results", "images"),
]
DEFAULT_IMAGE = next((p for p in _CANDIDATE_IMAGE_DIRS if os.path.isdir(p)), _CANDIDATE_IMAGE_DIRS[0])
DEFAULT_FT_CKPT = ""  # optional fine-tune checkpoint; leave empty unless you have one

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MODE_NAMES = ["Raw", "Denoise", "CLAHE", "Smooth", "Combined"]
LABEL_COLORS = {"femur": (255, 120, 120), "tibia": (120, 190, 255)}

_FONT_CACHE = {}
_TEXT_BITMAP_CACHE = {}


@dataclass
class Button:
    label: str
    action: str
    rect: Tuple[int, int, int, int]


@dataclass
class Slider:
    key: str
    rect: Tuple[int, int, int, int]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    # Cross-platform CJK-capable font candidates: macOS first (so this script
    # works out of the box on Mac), then Windows, then common Linux paths.
    candidates = [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Songti.ttc",
        # Windows
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        # Linux (Noto / WenQuanYi / Adobe Source Han)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf",
    ]
    font = None
    for fp in candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size=size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
        print(f"[WARN] No CJK font found; Chinese labels will appear as boxes. "
              f"Install Noto Sans CJK or PingFang and add the path to get_font().")
    _FONT_CACHE[size] = font
    return font


def draw_cn_text(img_bgr: np.ndarray, x: int, y: int, text: str, size: int, color_bgr=(30, 30, 30)):
    key = (text, int(size), int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
    bmp = _TEXT_BITMAP_CACHE.get(key)
    if bmp is None:
        font = get_font(size)
        probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        probe_draw = ImageDraw.Draw(probe)
        bbox = probe_draw.textbbox((0, 0), text, font=font)
        tw = max(1, int(bbox[2] - bbox[0]))
        th = max(1, int(bbox[3] - bbox[1]))
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0], 255)
        draw.text((-bbox[0], -bbox[1]), text, fill=color_rgb, font=font)
        rgba = np.array(img, dtype=np.uint8)
        bmp = rgba[..., [2, 1, 0, 3]]  # BGRA
        _TEXT_BITMAP_CACHE[key] = bmp

    h, w = img_bgr.shape[:2]
    bh, bw = bmp.shape[:2]
    if x >= w or y >= h or x + bw <= 0 or y + bh <= 0:
        return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    bx0, by0 = x0 - x, y0 - y
    bx1, by1 = bx0 + (x1 - x0), by0 + (y1 - y0)
    src = bmp[by0:by1, bx0:bx1]
    alpha = (src[..., 3:4].astype(np.float32) / 255.0)
    if np.max(alpha) <= 0:
        return
    roi = img_bgr[y0:y1, x0:x1].astype(np.float32)
    fg = src[..., :3].astype(np.float32)
    out = roi * (1.0 - alpha) + fg * alpha
    img_bgr[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)


def discover_images(path_str: str) -> Tuple[List[str], int]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(str(path))

    def valid(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_sam3_" not in p.name.lower()

    if path.is_dir():
        files = sorted(str(p) for p in path.iterdir() if valid(p))
        if not files:
            raise RuntimeError(f"No images in directory: {path}")
        return files, 0

    files = sorted(str(p) for p in path.parent.iterdir() if valid(p))
    if not files:
        raise RuntimeError(f"No images in directory: {path.parent}")
    idx = files.index(str(path)) if str(path) in files else 0
    return files, idx


def load_image_bgr(path: str) -> np.ndarray:
    def cv2_read_unicode(p: str, flags: int):
        try:
            data = np.fromfile(p, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, flags)
        except Exception:
            return None

    def norm_u8(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0, 1)
        return (arr * 255).astype(np.uint8)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = None
        try:
            arr = tifffile.imread(path)
        except Exception:
            arr = None
        if arr is None:
            arr = cv2_read_unicode(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise RuntimeError(f"Failed to read image: {path}")
            if arr.dtype != np.uint8:
                arr = norm_u8(arr)
            if arr.ndim == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return arr[..., :3]

        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[..., :3]
        elif arr.ndim == 3:
            arr = arr[..., 0]
        if arr.dtype != np.uint8:
            arr = norm_u8(arr)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr

    img = cv2_read_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def save_image_any(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        ext = ".png"
        path = path + ext
    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        raise RuntimeError(f"Encoding failed: {path}")
    try:
        buf.tofile(path)
    except Exception as e:
        raise RuntimeError(f"Write failed: {path} ({e})")


def preprocess_image(img: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return img.copy()
    if mode == 1:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    if mode == 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == 3:
        return cv2.GaussianBlur(img, (5, 5), 1.0)
    out = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_gamma_u8(img: np.ndarray, gamma: float) -> np.ndarray:
    g = max(0.05, float(gamma))
    lut = ((np.arange(256, dtype=np.float32) / 255.0) ** g * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


class App:
    def __init__(self, ckpt: str, ft_ckpt: str, bpe: str, image: str, device: str, out_dir: str = ""):
        self.files, self.idx = discover_images(image)
        self.path = self.files[self.idx]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        default_out = os.path.join(os.path.dirname(self.path), "sam3_outputs")
        self.output_dir = os.path.abspath(out_dir) if out_dir else default_out
        os.makedirs(self.output_dir, exist_ok=True)

        self.win_w, self.win_h = 1600, 940
        # Panel widened from 430 -> 510 so English labels fit without truncation.
        self.margin, self.panel_w = 12, 510
        self.view_x, self.view_y = self.margin, self.margin
        self.view_w = self.win_w - self.panel_w - self.margin * 3
        self.view_h = self.win_h - self.margin * 2
        self.panel_x = self.view_x + self.view_w + self.margin
        self.panel_y = self.margin
        self.buttons: List[Button] = []
        self.sliders: List[Slider] = []
        self.active_slider_key = ""
        self._slider_needs_commit = False
        self.hover = ""

        self.preprocess_mode = 0
        self.current_label = "femur"
        self.points_xy: List[List[float]] = []
        self.labels: List[int] = []
        self.prompt_bank = {
            "femur": {"points": [], "labels": []},
            "tibia": {"points": [], "labels": []},
        }
        self.cur_mask = None
        self.cur_logits = None
        self.cur_score = None
        self.masks = {"femur": None, "tibia": None}
        self.threshold = 0.0
        self.smooth_ksize = 0
        self.exclude_other_class = True
        self.auto_neg_from_other = True
        self.auto_neg_points = 12
        self.auto_neg_erode = 3
        self.last_auto_neg_used = 0
        self.annotation_mode = "sam"
        self.poly_points: List[List[float]] = []
        self.poly_prompt_pos = 24
        self.poly_prompt_neg = 12
        self.poly_ring_dilate = 10
        self.poly_close_dist_px = 14.0
        self.eraser_mode = False
        self.eraser_radius = 22
        self.eraser_radius_min = 4
        self.eraser_radius_max = 120
        self.eraser_dragging = False
        self.cursor_img_xy = None
        self.seg_fill_holes = True
        self.seg_dilate_px = 0
        self.seg_dilate_min = 0
        self.seg_dilate_max = 14
        self.enhance_cfg = {
            "gamma": {"name": "Gamma", "enabled": False, "value": 0.50},
            "window": {"name": "Bone window", "enabled": False, "value": 0.35},
            "clahe": {"name": "CLAHE", "enabled": False, "value": 0.35},
            "bilateral": {"name": "Bilateral", "enabled": False, "value": 0.30},
            "sharpen": {"name": "Light sharpen", "enabled": False, "value": 0.25},
        }
        self.status = "Ready"
        self.status_time = 0.0
        self.unsaved = False

        self.zoom = 1.0
        self.min_zoom, self.max_zoom = 0.5, 12.0
        self.base_scale = 1.0
        self.pan_x, self.pan_y = 0.0, 0.0
        self.dragging = False
        self.drag_last = (0, 0)
        self.last_render_time = 0.0

        self.model = build_sam3_image_model(
            checkpoint_path=ckpt,
            bpe_path=bpe,
            device=self.device,
            eval_mode=True,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
        )
        if ft_ckpt:
            if os.path.isfile(ft_ckpt):
                self._load_ft(ft_ckpt)
                print(f"[INFO] finetuned detector loaded: {ft_ckpt}")
            else:
                print(f"[WARN] finetuned checkpoint not found, skip: {ft_ckpt}")
        self.predictor = self.model.inst_interactive_predictor
        if getattr(self.predictor.model, "backbone", None) is None:
            self.predictor.model.backbone = self.model.backbone

        self.raw = None
        self.img = None
        self.load_current(reset_masks=True)

        self.win = "SAM3 Interactive"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.win_w, self.win_h)
        cv2.setMouseCallback(self.win, self.on_mouse)
        self.render()

    def _load_ft(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
            ck = ck["model"]
        if any(k.startswith("detector.") for k in ck):
            ck = {k.replace("detector.", ""): v for k, v in ck.items() if k.startswith("detector.")}
        else:
            prefixes = ("backbone.", "transformer.", "input_geometry_encoder.", "segmentation_head.", "dot_prod_scoring.")
            ck = {k: v for k, v in ck.items() if k.startswith(prefixes)}
        self.model.load_state_dict(ck, strict=False)

    def set_status(self, msg: str):
        self.status = msg
        self.status_time = time.time()

    def fit_view(self):
        h, w = self.img.shape[:2]
        self.base_scale = min(self.view_w / w, self.view_h / h)
        self.zoom = 1.0
        sw, sh = w * self.base_scale, h * self.base_scale
        self.pan_x = (self.view_w - sw) / 2
        self.pan_y = (self.view_h - sh) / 2

    def clamp_pan(self):
        h, w = self.img.shape[:2]
        sc = self.base_scale * self.zoom
        sw, sh = w * sc, h * sc
        if sw <= self.view_w:
            self.pan_x = (self.view_w - sw) / 2
        else:
            self.pan_x = min(0, max(self.view_w - sw, self.pan_x))
        if sh <= self.view_h:
            self.pan_y = (self.view_h - sh) / 2
        else:
            self.pan_y = min(0, max(self.view_h - sh, self.pan_y))

    def screen_to_img(self, sx: int, sy: int):
        if not (self.view_x <= sx < self.view_x + self.view_w and self.view_y <= sy < self.view_y + self.view_h):
            return None
        sc = self.base_scale * self.zoom
        x = (sx - self.view_x - self.pan_x) / sc
        y = (sy - self.view_y - self.pan_y) / sc
        h, w = self.img.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return float(x), float(y)

    def zoom_at(self, factor: float, sx: int, sy: int):
        p = self.screen_to_img(sx, sy)
        if p is None:
            return
        ox, oy = p
        old = self.zoom
        self.zoom = min(self.max_zoom, max(self.min_zoom, self.zoom * factor))
        if abs(self.zoom - old) < 1e-6:
            return
        sc = self.base_scale * self.zoom
        self.pan_x = (sx - self.view_x) - ox * sc
        self.pan_y = (sy - self.view_y) - oy * sc
        self.clamp_pan()

    def _in_view_rect(self, x: int, y: int) -> bool:
        return self.view_x <= x < self.view_x + self.view_w and self.view_y <= y < self.view_y + self.view_h

    def _compose_work_image(self) -> np.ndarray:
        out = preprocess_image(self.raw, self.preprocess_mode)
        out = self._apply_ct_enhance(out)
        return out

    def _refresh_predictor_image(self, repredict_current: bool):
        self.img = self._compose_work_image()
        self.predictor.set_image(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        if repredict_current and self.points_xy:
            self.predict()

    def _hit_button(self, x: int, y: int):
        for b in self.buttons:
            bx, by, bw, bh = b.rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return b
        return None

    def _short_path(self, p: str, max_len: int = 42) -> str:
        if len(p) <= max_len:
            return p
        keep = max(10, (max_len - 3) // 2)
        return f"{p[:keep]}...{p[-keep:]}"

    def _current_image_dir(self) -> str:
        if self.files:
            return os.path.dirname(self.files[0])
        return os.path.dirname(self.path)

    def _sync_active_prompts_to_bank(self, label: str = ""):
        cls = label or self.current_label
        if cls not in self.prompt_bank:
            return
        if cls == self.current_label:
            pts = [[float(p[0]), float(p[1])] for p in self.points_xy]
            lbs = [int(v) for v in self.labels]
            self.prompt_bank[cls] = {"points": pts, "labels": lbs}

    def _choose_image_dir(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            self.set_status(f"Directory picker unavailable: {e}")
            return
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            picked = filedialog.askdirectory(
                title="Select image directory",
                initialdir=self._current_image_dir(),
            )
            root.destroy()
        except Exception as e:
            self.set_status(f"Failed to pick image dir: {e}")
            return
        if not picked:
            self.set_status("Image dir unchanged")
            return
        try:
            files, idx = discover_images(picked)
        except Exception as e:
            self.set_status(f"Directory unavailable: {e}")
            return
        self.save_auto_bundle(auto=True)
        self.files, self.idx = files, idx
        self.load_current(reset_masks=True)
        self.set_status(f"Image dir switched to: {self._short_path(picked, 56)}")

    def _choose_output_dir(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            self.set_status(f"Directory picker unavailable: {e}")
            return
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            picked = filedialog.askdirectory(
                title="Select output directory",
                initialdir=self.output_dir if os.path.isdir(self.output_dir) else os.path.dirname(self.path),
            )
            root.destroy()
        except Exception as e:
            self.set_status(f"Failed to pick directory: {e}")
            return
        if not picked:
            self.set_status("Output dir unchanged")
            return
        self.output_dir = os.path.abspath(picked)
        os.makedirs(self.output_dir, exist_ok=True)
        self.set_status(f"Output dir: {self._short_path(self.output_dir, 56)}")

    def _enh_keys(self):
        return ["gamma", "window", "clahe", "bilateral", "sharpen"]

    def _get_slider(self, key: str):
        for s in self.sliders:
            if s.key == key:
                return s
        return None

    def _hit_slider(self, x: int, y: int):
        for s in self.sliders:
            rx, ry, rw, rh = s.rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return s
        return None

    def _toggle_enhancement(self, key: str):
        cfg = self.enhance_cfg.get(key)
        if cfg is None:
            return
        cfg["enabled"] = not cfg["enabled"]
        self._refresh_predictor_image(repredict_current=False)
        self.set_status(f"{cfg['name']}: {'ON' if cfg['enabled'] else 'OFF'}")

    def _set_enhancement_value(self, key: str, v: float, commit_predictor: bool = True):
        cfg = self.enhance_cfg.get(key)
        if cfg is None:
            return
        old = float(cfg["value"])
        v = float(np.clip(v, 0.0, 1.0))
        if abs(v - old) < 1e-6:
            return
        cfg["value"] = v
        if commit_predictor:
            self._refresh_predictor_image(repredict_current=False)
        else:
            self.img = self._compose_work_image()
        if key == "gamma":
            self.set_status(f"Gamma: {self._gamma_from_norm(v):.2f}")
        else:
            self.set_status(f"{cfg['name']} strength: {v:.2f}")

    def _update_slider_by_mouse(self, key: str, x: int):
        s = self._get_slider(key)
        if s is None:
            return
        rx, ry, rw, rh = s.rect
        if rw <= 1:
            return
        value = (x - rx) / float(rw)
        if key == "eraser_radius":
            self._set_eraser_radius_from_norm(value)
        elif key == "seg_dilate":
            self._set_seg_dilate_from_norm(value)
        else:
            self._set_enhancement_value(key, value, commit_predictor=False)
            self._slider_needs_commit = True

    def _gamma_from_norm(self, v: float) -> float:
        # v in [0,1] -> gamma in ~[0.43, 2.30], 0.5 maps to ~1.0
        return float(2.0 ** ((float(v) - 0.5) * 2.4))

    def _eraser_norm(self) -> float:
        return float((self.eraser_radius - self.eraser_radius_min) / (self.eraser_radius_max - self.eraser_radius_min))

    def _set_eraser_radius_from_norm(self, v: float):
        v = float(np.clip(v, 0.0, 1.0))
        r = int(round(self.eraser_radius_min + v * (self.eraser_radius_max - self.eraser_radius_min)))
        r = int(np.clip(r, self.eraser_radius_min, self.eraser_radius_max))
        if r == self.eraser_radius:
            return
        self.eraser_radius = r
        self.set_status(f"Eraser radius: {self.eraser_radius}px")

    def _seg_dilate_norm(self) -> float:
        return float((self.seg_dilate_px - self.seg_dilate_min) / (self.seg_dilate_max - self.seg_dilate_min))

    def _set_seg_dilate_from_norm(self, v: float):
        v = float(np.clip(v, 0.0, 1.0))
        r = int(round(self.seg_dilate_min + v * (self.seg_dilate_max - self.seg_dilate_min)))
        r = int(np.clip(r, self.seg_dilate_min, self.seg_dilate_max))
        if r == self.seg_dilate_px:
            return
        self.seg_dilate_px = r
        self.rebuild_from_logits()
        self.set_status(f"Postproc dilation: {self.seg_dilate_px}px")

    def _fill_holes_binary(self, mask01: np.ndarray) -> np.ndarray:
        m = (mask01 > 0).astype(np.uint8) * 255
        h, w = m.shape[:2]
        # Find a background seed on border for flood fill.
        seed = None
        border_coords = []
        for x in range(w):
            border_coords.append((x, 0))
            border_coords.append((x, h - 1))
        for y in range(h):
            border_coords.append((0, y))
            border_coords.append((w - 1, y))
        for sx, sy in border_coords:
            if m[sy, sx] == 0:
                seed = (sx, sy)
                break
        if seed is None:
            return (m > 0).astype(np.uint8)
        flood = m.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, ff_mask, seedPoint=seed, newVal=255)
        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(m, flood_inv)
        return (filled > 0).astype(np.uint8)

    def _apply_ct_enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        out = img_bgr

        # 1) Gamma.
        cfg = self.enhance_cfg["gamma"]
        if cfg["enabled"] and cfg["value"] > 1e-6:
            g = self._gamma_from_norm(cfg["value"])
            out = apply_gamma_u8(out, g)

        # 2) Bone window (fixed window/level in u8 space) + strength blending.
        cfg = self.enhance_cfg["window"]
        if cfg["enabled"] and cfg["value"] > 1e-6:
            wl, ww = 170.0, 180.0
            low, high = wl - ww / 2.0, wl + ww / 2.0
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
            win = np.clip((gray - low) / max(1.0, high - low), 0.0, 1.0)
            win_u8 = (win * 255.0).astype(np.uint8)
            win_bgr = cv2.cvtColor(win_u8, cv2.COLOR_GRAY2BGR)
            alpha = float(cfg["value"])
            out = cv2.addWeighted(out, 1.0 - alpha, win_bgr, alpha, 0)

        # 3) Light CLAHE.
        cfg = self.enhance_cfg["clahe"]
        if cfg["enabled"] and cfg["value"] > 1e-6:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            clip = 1.0 + 3.0 * float(cfg["value"])
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            eq = clahe.apply(gray)
            eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            alpha = 0.8 * float(cfg["value"])
            out = cv2.addWeighted(out, 1.0 - alpha, eq_bgr, alpha, 0)

        # 4) Light bilateral denoise.
        cfg = self.enhance_cfg["bilateral"]
        if cfg["enabled"] and cfg["value"] > 1e-6:
            v = float(cfg["value"])
            sigma = 8.0 + 40.0 * v
            h, w = out.shape[:2]
            if v < 0.34:
                scale = 1.0
            elif v < 0.67:
                scale = 0.75
            else:
                scale = 0.5
            if scale < 0.999:
                sw, sh = max(64, int(w * scale)), max(64, int(h * scale))
                work = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
            else:
                work = out
            # Faster approximation: run bilateral on grayscale channel only.
            work_gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            den_gray = cv2.bilateralFilter(work_gray, d=5, sigmaColor=sigma, sigmaSpace=sigma)
            den = cv2.cvtColor(den_gray, cv2.COLOR_GRAY2BGR)
            if scale < 0.999:
                den = cv2.resize(den, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = 0.75 * v
            out = cv2.addWeighted(out, 1.0 - alpha, den, alpha, 0)

        # 5) Weak unsharp mask.
        cfg = self.enhance_cfg["sharpen"]
        if cfg["enabled"] and cfg["value"] > 1e-6:
            v = float(cfg["value"])
            blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0 + 1.2 * v)
            sharp = cv2.addWeighted(out, 1.0 + 0.9 * v, blur, -0.9 * v, 0)
            alpha = 0.7 * v
            out = cv2.addWeighted(out, 1.0 - alpha, sharp, alpha, 0)

        return np.clip(out, 0, 255).astype(np.uint8)

    def _toggle_annotation_mode(self):
        if self.annotation_mode == "sam":
            self.annotation_mode = "polygon"
            self.eraser_mode = False
            self.eraser_dragging = False
            self._sync_active_prompts_to_bank()
            self.points_xy, self.labels = [], []
            self.poly_points = []
            self.cur_logits, self.cur_score = None, None
            self.last_auto_neg_used = 0
            self.set_status("Switched to polygon mode")
        else:
            self.annotation_mode = "sam"
            self._sync_active_prompts_to_bank()
            self.poly_points = []
            self.set_status("Switched to SAM point mode")

    def _toggle_eraser_mode(self):
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.annotation_mode = "sam"
            self.poly_points = []
            self.set_status(f"Eraser: ON (radius {self.eraser_radius}px)")
        else:
            self.set_status("Eraser: OFF")

    def _apply_eraser_at(self, px: float, py: float):
        cur = self.masks.get(self.current_label)
        if cur is None:
            return
        cv2.circle(cur, (int(round(px)), int(round(py))), int(self.eraser_radius), 0, -1)
        self.masks[self.current_label] = cur
        self.cur_mask = cur
        self.cur_logits, self.cur_score = None, None
        self.unsaved = True

    def _sample_points_from_binary(self, mask01: np.ndarray, n: int):
        yx = np.column_stack(np.where(mask01 > 0))
        if len(yx) == 0 or n <= 0:
            return []
        n_pick = min(int(n), len(yx))
        idx = np.linspace(0, len(yx) - 1, num=n_pick, dtype=np.int32)
        picked = yx[idx]
        return [[float(x), float(y)] for y, x in picked]

    def _build_polygon_mask(self):
        h, w = self.img.shape[:2]
        pts = np.array([[int(round(x)), int(round(y))] for x, y in self.poly_points], dtype=np.int32)
        poly = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly, [pts], 1)
        return poly

    def _apply_polygon_to_mask(self, auto_closed: bool = False):
        if self.annotation_mode != "polygon":
            self.set_status("Not in polygon mode")
            return
        if len(self.poly_points) < 3:
            self.set_status("Polygon needs at least 3 points")
            return

        poly = self._build_polygon_mask()

        ring_k = self.poly_ring_dilate * 2 + 1
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_k, ring_k))
        dilated = cv2.dilate(poly, ring_kernel, iterations=1)
        ring = ((dilated > 0) & (poly == 0)).astype(np.uint8)

        pos_points = self._sample_points_from_binary(poly, self.poly_prompt_pos)
        neg_points = self._sample_points_from_binary(ring, self.poly_prompt_neg)
        if not pos_points:
            self.set_status("Polygon area too small to sample prompts")
            return

        for p in pos_points:
            self.points_xy.append(p)
            self.labels.append(1)
        for p in neg_points:
            self.points_xy.append(p)
            self.labels.append(0)

        self.poly_points = []
        self.predict()
        suffix = " (auto-closed)" if auto_closed else ""
        self.set_status(f"Polygon -> point prompts: +{len(pos_points)} / -{len(neg_points)}{suffix}")

    def _handle_button_action(self, action: str):
        if action.startswith("toggle_enh_"):
            self._toggle_enhancement(action.replace("toggle_enh_", "", 1))
        elif action == "toggle_seg_fill_holes":
            self.seg_fill_holes = not self.seg_fill_holes
            self.rebuild_from_logits()
            self.set_status(f"Hole filling: {'ON' if self.seg_fill_holes else 'OFF'}")
        elif action == "toggle_annotation_mode":
            self._toggle_annotation_mode()
        elif action == "toggle_eraser_mode":
            self._toggle_eraser_mode()
        elif action == "commit_polygon":
            self._apply_polygon_to_mask()
        elif action == "clear_polygon":
            self.poly_points = []
            self.set_status("Polygon points cleared")
        elif action == "choose_image_dir":
            self._choose_image_dir()
        elif action == "choose_output_dir":
            self._choose_output_dir()

    def load_current(self, reset_masks: bool):
        self.path = self.files[self.idx]
        self.raw = load_image_bgr(self.path)
        self.img = self._compose_work_image()
        self.predictor.set_image(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        self.points_xy, self.labels = [], []
        self.poly_points = []
        self.prompt_bank = {
            "femur": {"points": [], "labels": []},
            "tibia": {"points": [], "labels": []},
        }
        self.cur_mask, self.cur_logits, self.cur_score = None, None, None
        self.last_auto_neg_used = 0
        self.eraser_dragging = False
        self.cursor_img_xy = None
        self._slider_needs_commit = False
        if reset_masks:
            self.masks = {"femur": None, "tibia": None}
        self.fit_view()
        self.set_status(f"Loaded: {os.path.basename(self.path)}")
        self.unsaved = False

    def postprocess(self, logits: np.ndarray) -> np.ndarray:
        if logits.shape[:2] != self.img.shape[:2]:
            logits = cv2.resize(logits, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = (logits > self.threshold).astype(np.uint8)
        if self.smooth_ksize >= 3:
            k = self.smooth_ksize if self.smooth_ksize % 2 == 1 else self.smooth_ksize + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if self.seg_fill_holes:
            mask = self._fill_holes_binary(mask)
        if self.seg_dilate_px > 0:
            dk = self.seg_dilate_px * 2 + 1
            dker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
            mask = cv2.dilate(mask, dker, iterations=1)
        if self.exclude_other_class:
            other_label = "tibia" if self.current_label == "femur" else "femur"
            other_mask = self.masks.get(other_label)
            if other_mask is not None:
                mask[other_mask > 0] = 0
        return mask

    def rebuild_from_logits(self):
        if self.cur_logits is None:
            return
        self.cur_mask = self.postprocess(self.cur_logits)
        self.masks[self.current_label] = self.cur_mask

    def _build_prompt_with_auto_neg(self):
        pts = list(self.points_xy)
        lbs = list(self.labels)
        extra_neg = 0

        if not self.auto_neg_from_other:
            return (
                np.array(pts, dtype=np.float32),
                np.array(lbs, dtype=np.int32),
                extra_neg,
            )

        other_label = "tibia" if self.current_label == "femur" else "femur"
        other_mask = self.masks.get(other_label)
        if other_mask is None:
            return (
                np.array(pts, dtype=np.float32),
                np.array(lbs, dtype=np.int32),
                extra_neg,
            )

        other_u8 = (other_mask > 0).astype(np.uint8)
        if not np.any(other_u8):
            return (
                np.array(pts, dtype=np.float32),
                np.array(lbs, dtype=np.int32),
                extra_neg,
            )

        inner = other_u8
        if self.auto_neg_erode > 0:
            k = self.auto_neg_erode * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            eroded = cv2.erode(other_u8, kernel, iterations=1)
            if np.any(eroded):
                inner = eroded

        yx = np.column_stack(np.where(inner > 0))
        if len(yx) == 0:
            return (
                np.array(pts, dtype=np.float32),
                np.array(lbs, dtype=np.int32),
                extra_neg,
            )

        n = min(self.auto_neg_points, len(yx))
        pick_idx = np.linspace(0, len(yx) - 1, num=n, dtype=np.int32)
        picked = yx[pick_idx]
        for y, x in picked:
            pts.append([float(x), float(y)])
            lbs.append(0)
        extra_neg = int(n)
        return (
            np.array(pts, dtype=np.float32),
            np.array(lbs, dtype=np.int32),
            extra_neg,
        )

    def predict(self):
        if not self.points_xy:
            return
        point_coords, point_labels, extra_neg = self._build_prompt_with_auto_neg()
        self.last_auto_neg_used = extra_neg
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
            return_logits=True,
        )
        self.cur_logits = masks[0]
        self.cur_score = float(scores[0])
        self.cur_mask = self.postprocess(self.cur_logits)
        self.masks[self.current_label] = self.cur_mask
        self.unsaved = True

    def compose_label_map(self) -> np.ndarray:
        h, w = self.img.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        f = self.masks["femur"]
        t = self.masks["tibia"]
        if f is not None:
            out[f > 0] = 1
        if t is not None:
            idx = t > 0
            overlap = (out == 1) & idx
            out[idx] = 2
            out[overlap] = 3
        return out

    def compose_prompt_vis(self) -> np.ndarray:
        base = self.raw.copy() if self.raw is not None else self.img.copy()
        for cls in ("femur", "tibia"):
            cls_col = LABEL_COLORS[cls]
            bank = self.prompt_bank.get(cls, {"points": [], "labels": []})
            pts = bank.get("points", [])
            lbs = bank.get("labels", [])
            for (x, y), lb in zip(pts, lbs):
                px, py = int(round(x)), int(round(y))
                if lb == 1:
                    cv2.circle(base, (px, py), 6, cls_col, -1, lineType=cv2.LINE_AA)
                    cv2.circle(base, (px, py), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                else:
                    cv2.line(base, (px - 7, py - 7), (px + 7, py + 7), cls_col, 2, lineType=cv2.LINE_AA)
                    cv2.line(base, (px - 7, py + 7), (px + 7, py - 7), cls_col, 2, lineType=cv2.LINE_AA)
                    cv2.circle(base, (px, py), 10, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        return base

    def save_auto_bundle(self, auto=True):
        self._sync_active_prompts_to_bank()
        if self.masks["femur"] is None and self.masks["tibia"] is None:
            return
        out_dir = self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.path))[0]
        label_map = self.compose_label_map()
        vis = self.img.copy()
        for cls, col in LABEL_COLORS.items():
            m = self.masks[cls]
            if m is not None:
                vis = np.where((m > 0)[..., None], (0.55 * vis + 0.45 * np.array(col, dtype=np.float32)).astype(np.uint8), vis)
        label_color = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
        label_color[label_map == 1] = LABEL_COLORS["femur"]
        label_color[label_map == 2] = LABEL_COLORS["tibia"]
        label_color[label_map == 3] = (220, 160, 255)
        points_vis = self.compose_prompt_vis()

        json_path = os.path.join(out_dir, f"{stem}_labels.json")
        vis_path = os.path.join(out_dir, f"{stem}_vis.png")
        map_path = os.path.join(out_dir, f"{stem}_label_map.png")
        color_path = os.path.join(out_dir, f"{stem}_label_color.png")
        points_path = os.path.join(out_dir, f"{stem}_points.png")

        meta = {
            "image": self.path,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "classes": {"0": "background", "1": "femur", "2": "tibia", "3": "overlap"},
            "prompts": {
                "femur": {
                    "positive_points_xy": [
                        self.prompt_bank["femur"]["points"][i]
                        for i, lb in enumerate(self.prompt_bank["femur"]["labels"])
                        if lb == 1
                    ],
                    "negative_points_xy": [
                        self.prompt_bank["femur"]["points"][i]
                        for i, lb in enumerate(self.prompt_bank["femur"]["labels"])
                        if lb == 0
                    ],
                },
                "tibia": {
                    "positive_points_xy": [
                        self.prompt_bank["tibia"]["points"][i]
                        for i, lb in enumerate(self.prompt_bank["tibia"]["labels"])
                        if lb == 1
                    ],
                    "negative_points_xy": [
                        self.prompt_bank["tibia"]["points"][i]
                        for i, lb in enumerate(self.prompt_bank["tibia"]["labels"])
                        if lb == 0
                    ],
                },
            },
            "stats": {
                "femur_pixels": int(np.sum(label_map == 1)),
                "tibia_pixels": int(np.sum(label_map == 2)),
                "overlap_pixels": int(np.sum(label_map == 3)),
                "femur_prompt_pos": int(sum(1 for v in self.prompt_bank["femur"]["labels"] if v == 1)),
                "femur_prompt_neg": int(sum(1 for v in self.prompt_bank["femur"]["labels"] if v == 0)),
                "tibia_prompt_pos": int(sum(1 for v in self.prompt_bank["tibia"]["labels"] if v == 1)),
                "tibia_prompt_neg": int(sum(1 for v in self.prompt_bank["tibia"]["labels"] if v == 0)),
            },
            "files": {
                "label_map_png": map_path,
                "label_color_png": color_path,
                "visualization_png": vis_path,
                "points_png": points_path,
            },
        }
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            save_image_any(vis_path, vis)
            save_image_any(map_path, label_map)
            save_image_any(color_path, label_color)
            save_image_any(points_path, points_vis)
        except Exception as e:
            self.set_status(f"Save failed: {e}")
            return

        self.unsaved = False
        msg = "Auto-save done" if auto else "Save done"
        self.set_status(f"{msg}: {os.path.basename(json_path)}")

    def switch_image(self, step: int):
        self.save_auto_bundle(auto=True)
        next_idx = self.idx + step
        if next_idx < 0:
            self.idx = 0
            self.set_status("First image reached, not wrapping.")
            print("[INFO] First image reached, not wrapping.")
            return
        if next_idx >= len(self.files):
            self.idx = len(self.files) - 1
            self.set_status("Last image reached, not wrapping. Press Esc/X to exit.")
            print("[INFO] Last image reached, not wrapping.")
            return
        self.idx = next_idx
        self.load_current(reset_masks=True)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            p_now = self.screen_to_img(x, y)
            self.cursor_img_xy = p_now
            if self.active_slider_key:
                self._update_slider_by_mouse(self.active_slider_key, x)
                self.render_throttled()
                return
            if self.eraser_mode and self.eraser_dragging and p_now is not None:
                self._apply_eraser_at(p_now[0], p_now[1])
                self.render_throttled()
                return
            hit = self._hit_button(x, y)
            new_hover = "" if hit is None else hit.action
            if new_hover != self.hover:
                self.hover = new_hover
                self.render_throttled()
            if self.dragging:
                self.pan_x += x - self.drag_last[0]
                self.pan_y += y - self.drag_last[1]
                self.drag_last = (x, y)
                self.clamp_pan()
                self.render_throttled()
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            self.zoom_at(1.15 if flags > 0 else 1 / 1.15, x, y)
            self.render()
            return
        if event == cv2.EVENT_MBUTTONDOWN:
            self.dragging = True
            self.drag_last = (x, y)
            return
        if event == cv2.EVENT_MBUTTONUP:
            self.dragging = False
            return
        if event == cv2.EVENT_LBUTTONUP:
            self.active_slider_key = ""
            self.eraser_dragging = False
            if self._slider_needs_commit:
                self._refresh_predictor_image(repredict_current=False)
                self._slider_needs_commit = False
                self.render()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            sl = self._hit_slider(x, y)
            if sl is not None:
                self.active_slider_key = sl.key
                self._update_slider_by_mouse(sl.key, x)
                self.render()
                return
            hit = self._hit_button(x, y)
            if hit is not None:
                self._handle_button_action(hit.action)
                self.render()
                return

        p = self.screen_to_img(x, y)
        if p is None:
            return
        px, py = p
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.eraser_mode:
                self.eraser_dragging = True
                self._apply_eraser_at(px, py)
                self.render()
                return
            if self.annotation_mode == "polygon":
                if self.poly_points and (flags & cv2.EVENT_FLAG_SHIFTKEY):
                    lx, ly = self.poly_points[-1]
                    if abs(px - lx) >= abs(py - ly):
                        py = ly
                    else:
                        px = lx
                if len(self.poly_points) >= 3:
                    fx, fy = self.poly_points[0]
                    if (px - fx) * (px - fx) + (py - fy) * (py - fy) <= self.poly_close_dist_px * self.poly_close_dist_px:
                        self._apply_polygon_to_mask(auto_closed=True)
                        self.render()
                        return
                self.poly_points.append([px, py])
                self.set_status(f"Polygon points: {len(self.poly_points)}")
                self.render()
                return
            self.points_xy.append([px, py]); self.labels.append(1); self.predict(); self.render()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.eraser_mode:
                return
            if self.annotation_mode == "polygon":
                if self.poly_points:
                    self.poly_points.pop()
                    self.set_status(f"Polygon points: {len(self.poly_points)}")
                    self.render()
                return
            self.points_xy.append([px, py]); self.labels.append(0); self.predict(); self.render()

    def draw_view(self, canvas: np.ndarray):
        cv2.rectangle(canvas, (self.view_x, self.view_y), (self.view_x + self.view_w, self.view_y + self.view_h), (220, 230, 232), -1)
        composed = self.img.copy()
        for cls, mask in self.masks.items():
            if mask is None:
                continue
            col = LABEL_COLORS[cls]
            alpha = 0.25 if cls != self.current_label else 0.45
            overlay = composed.copy()
            overlay[mask > 0] = col
            composed = cv2.addWeighted(composed, 1 - alpha, overlay, alpha, 0)
        for (x, y), lb in zip(self.points_xy, self.labels):
            col = (50, 180, 80) if lb == 1 else (60, 80, 220)
            cv2.circle(composed, (int(x), int(y)), 5, col, -1)
            cv2.circle(composed, (int(x), int(y)), 8, (255, 255, 255), 1)
        if self.poly_points:
            poly = np.array([[int(x), int(y)] for x, y in self.poly_points], dtype=np.int32)
            fill_col, line_col, close_col = (200, 224, 245), (70, 135, 200), (120, 180, 230)
            if len(poly) >= 3:
                overlay = composed.copy()
                cv2.fillPoly(overlay, [poly], fill_col)
                composed = cv2.addWeighted(composed, 0.82, overlay, 0.18, 0)
            cv2.polylines(composed, [poly], False, line_col, 2, lineType=cv2.LINE_AA)
            if len(poly) >= 2:
                cv2.line(composed, tuple(poly[-1]), tuple(poly[0]), close_col, 1, lineType=cv2.LINE_AA)
            if len(poly) >= 3:
                fx, fy = int(poly[0][0]), int(poly[0][1])
                cv2.circle(composed, (fx, fy), int(self.poly_close_dist_px), close_col, 1, lineType=cv2.LINE_AA)
            for x, y in poly:
                cv2.circle(composed, (int(x), int(y)), 4, line_col, -1)
        if self.eraser_mode and self.cursor_img_xy is not None:
            ex, ey = int(self.cursor_img_xy[0]), int(self.cursor_img_xy[1])
            cv2.circle(composed, (ex, ey), int(self.eraser_radius), (30, 40, 220), 1, lineType=cv2.LINE_AA)
        h, w = composed.shape[:2]
        sc = self.base_scale * self.zoom
        sw, sh = max(1, int(w * sc)), max(1, int(h * sc))
        big = cv2.resize(composed, (sw, sh), interpolation=cv2.INTER_LINEAR)
        px, py = int(self.view_x + self.pan_x), int(self.view_y + self.pan_y)
        x0, y0 = max(self.view_x, px), max(self.view_y, py)
        x1, y1 = min(self.view_x + self.view_w, px + sw), min(self.view_y + self.view_h, py + sh)
        if x1 > x0 and y1 > y0:
            sx0, sy0 = x0 - px, y0 - py
            canvas[y0:y1, x0:x1] = big[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        cv2.rectangle(canvas, (self.view_x, self.view_y), (self.view_x + self.view_w, self.view_y + self.view_h), (180, 200, 204), 1)

    def draw_panel(self, canvas: np.ndarray):
        self.buttons = []
        self.sliders = []
        cv2.rectangle(canvas, (self.panel_x, self.panel_y), (self.panel_x + self.panel_w, self.panel_y + self.view_h), (247, 252, 252), -1)
        cv2.rectangle(canvas, (self.panel_x, self.panel_y), (self.panel_x + self.panel_w, self.panel_y + self.view_h), (185, 210, 214), 1)
        draw_cn_text(canvas, self.panel_x + 14, self.panel_y + 12, "Control panel", 28, (35, 118, 142))

        # Top: four CT enhancement controls (toggle + slider).
        y0 = self.panel_y + 50
        draw_cn_text(canvas, self.panel_x + 14, y0, "CT enhancement (toggle + slider strength)", 17, (42, 96, 114))
        # Widened from 118 -> 170 so "Light sharpen:OFF" / "Bone window:OFF" fit.
        ctl_btn_w, ctl_btn_h = 170, 26
        ctl_gap_y = 38
        for i, key in enumerate(self._enh_keys()):
            cfg = self.enhance_cfg[key]
            row_y = y0 + 24 + i * ctl_gap_y
            bx = self.panel_x + 14
            by = row_y
            action = f"toggle_enh_{key}"
            btn = Button(label=f"{cfg['name']}:{'ON' if cfg['enabled'] else 'OFF'}", action=action, rect=(bx, by, ctl_btn_w, ctl_btn_h))
            self.buttons.append(btn)
            hover = (self.hover == action)
            fill = (216, 240, 234) if cfg["enabled"] else (236, 240, 242)
            if hover:
                fill = (204, 233, 226) if cfg["enabled"] else (225, 232, 235)
            edge = (72, 140, 98) if cfg["enabled"] else (132, 158, 166)
            cv2.rectangle(canvas, (bx, by), (bx + ctl_btn_w, by + ctl_btn_h), fill, -1)
            cv2.rectangle(canvas, (bx, by), (bx + ctl_btn_w, by + ctl_btn_h), edge, 1)
            draw_cn_text(canvas, bx + 8, by + 4, btn.label, 16, (36, 88, 94))

            sx = bx + ctl_btn_w + 10
            sw = self.panel_w - (sx - self.panel_x) - 20
            sh = 12
            sy = by + (ctl_btn_h - sh) // 2
            self.sliders.append(Slider(key=key, rect=(sx, sy, sw, sh)))
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), (226, 232, 236), -1)
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), (172, 188, 196), 1)
            v = float(cfg["value"])
            px = int(sx + v * sw)
            cv2.rectangle(canvas, (sx, sy), (px, sy + sh), (170, 210, 225), -1)
            cv2.circle(canvas, (px, sy + sh // 2), 7, (86, 146, 160), -1)
            right_text = f"{self._gamma_from_norm(v):.2f}" if key == "gamma" else f"{int(v * 100)}%"
            draw_cn_text(canvas, sx + sw - 44, by + 4, right_text, 15, (50, 88, 98))

        # Eraser control: toggle + radius slider.
        er_y = y0 + 24 + len(self._enh_keys()) * ctl_gap_y + 4
        er_btn_w, er_btn_h = 170, 26
        er_bx = self.panel_x + 14
        er_by = er_y
        er_btn = Button(
            label=f"Eraser:{'ON' if self.eraser_mode else 'OFF'}",
            action="toggle_eraser_mode",
            rect=(er_bx, er_by, er_btn_w, er_btn_h),
        )
        self.buttons.append(er_btn)
        hover = (self.hover == er_btn.action)
        er_fill = (244, 226, 226) if self.eraser_mode else (236, 240, 242)
        if hover:
            er_fill = (236, 212, 212) if self.eraser_mode else (225, 232, 235)
        er_edge = (156, 90, 90) if self.eraser_mode else (132, 158, 166)
        cv2.rectangle(canvas, (er_bx, er_by), (er_bx + er_btn_w, er_by + er_btn_h), er_fill, -1)
        cv2.rectangle(canvas, (er_bx, er_by), (er_bx + er_btn_w, er_by + er_btn_h), er_edge, 1)
        draw_cn_text(canvas, er_bx + 8, er_by + 4, er_btn.label, 16, (86, 48, 48))

        er_sx = er_bx + er_btn_w + 10
        er_sw = self.panel_w - (er_sx - self.panel_x) - 20
        er_sh = 12
        er_sy = er_by + (er_btn_h - er_sh) // 2
        self.sliders.append(Slider(key="eraser_radius", rect=(er_sx, er_sy, er_sw, er_sh)))
        cv2.rectangle(canvas, (er_sx, er_sy), (er_sx + er_sw, er_sy + er_sh), (226, 232, 236), -1)
        cv2.rectangle(canvas, (er_sx, er_sy), (er_sx + er_sw, er_sy + er_sh), (172, 188, 196), 1)
        er_v = self._eraser_norm()
        er_px = int(er_sx + er_v * er_sw)
        cv2.rectangle(canvas, (er_sx, er_sy), (er_px, er_sy + er_sh), (224, 196, 196), -1)
        cv2.circle(canvas, (er_px, er_sy + er_sh // 2), 7, (156, 90, 90), -1)
        draw_cn_text(canvas, er_sx + er_sw - 62, er_by + 4, f"{self.eraser_radius}px", 15, (84, 56, 56))

        # Seg postprocess repair control: fill holes + dilation slider.
        seg_y = er_y + ctl_gap_y
        seg_btn = Button(
            label=f"Hole-fill:{'ON' if self.seg_fill_holes else 'OFF'}",
            action="toggle_seg_fill_holes",
            rect=(er_bx, seg_y, er_btn_w, er_btn_h),
        )
        self.buttons.append(seg_btn)
        hover = (self.hover == seg_btn.action)
        seg_fill = (222, 236, 248) if self.seg_fill_holes else (236, 240, 242)
        if hover:
            seg_fill = (208, 228, 244) if self.seg_fill_holes else (225, 232, 235)
        seg_edge = (82, 120, 168) if self.seg_fill_holes else (132, 158, 166)
        cv2.rectangle(canvas, (er_bx, seg_y), (er_bx + er_btn_w, seg_y + er_btn_h), seg_fill, -1)
        cv2.rectangle(canvas, (er_bx, seg_y), (er_bx + er_btn_w, seg_y + er_btn_h), seg_edge, 1)
        draw_cn_text(canvas, er_bx + 8, seg_y + 4, seg_btn.label, 16, (52, 82, 120))

        seg_sx = er_sx
        seg_sw = er_sw
        seg_sh = 12
        seg_sy = seg_y + (er_btn_h - seg_sh) // 2
        self.sliders.append(Slider(key="seg_dilate", rect=(seg_sx, seg_sy, seg_sw, seg_sh)))
        cv2.rectangle(canvas, (seg_sx, seg_sy), (seg_sx + seg_sw, seg_sy + seg_sh), (226, 232, 236), -1)
        cv2.rectangle(canvas, (seg_sx, seg_sy), (seg_sx + seg_sw, seg_sy + seg_sh), (172, 188, 196), 1)
        seg_v = self._seg_dilate_norm()
        seg_px = int(seg_sx + seg_v * seg_sw)
        cv2.rectangle(canvas, (seg_sx, seg_sy), (seg_px, seg_sy + seg_sh), (190, 216, 236), -1)
        cv2.circle(canvas, (seg_px, seg_sy + seg_sh // 2), 7, (82, 120, 168), -1)
        draw_cn_text(canvas, seg_sx + seg_sw - 62, seg_y + 4, f"{self.seg_dilate_px}px", 15, (62, 82, 110))

        score = "--" if self.cur_score is None else f"{self.cur_score:.3f}"
        gamma_show = self._gamma_from_norm(self.enhance_cfg["gamma"]["value"])
        info = [
            f"File: [{self.idx+1}/{len(self.files)}] {os.path.basename(self.path)}",
            f"Image dir: {self._short_path(self._current_image_dir())}",
            f"Output dir: {self._short_path(self.output_dir)}",
            f"Class: {self.current_label}  Mode: {'Polygon' if self.annotation_mode == 'polygon' else 'SAM points'}",
            f"Tool: {'Eraser' if self.eraser_mode else 'Annotate'}  Pts: {len(self.points_xy)}  Score: {score}",
            f"Polygon pts: {len(self.poly_points)}  Thr: {self.threshold:.2f}  Smooth: {self.smooth_ksize}",
            f"Postproc: hole-fill={self.seg_fill_holes}  dilate={self.seg_dilate_px}px",
            f"Anti-bleed: {self.exclude_other_class}  Auto-neg: {self.auto_neg_from_other}  Gamma: {gamma_show:.2f}",
            f"Status: {self.status}",
            "Mouse: L-click (+pt / add)  R-click (-pt / undo)  M-drag pan  wheel zoom",
            "Eraser ON: hold L-click to erase continuously; slider sets radius",
            "Polygon: Shift+L-click snap; click near first vertex to auto-close + submit",
            "Keys: A/D prev/next  I pick image dir  U pick out dir  S save",
            "Keys: M mode  B eraser  Enter submit polygon  T anti-bleed  G auto-neg",
            "Keys: Q/E smooth kernel  [/] threshold  Esc/X exit",
            "Saves: labels.json / vis.png / label_map.png / label_color.png / points.png",
        ]
        y = seg_y + er_btn_h + 12
        for line in info:
            # Smaller font (13) so long English help lines don't overflow the panel.
            draw_cn_text(canvas, self.panel_x + 14, y, line, 13, (36, 78, 88))
            y += 16

        btn_w, btn_h = self.panel_w - 28, 34
        bx = self.panel_x + 14
        gap = 8
        by_commit = self.panel_y + self.view_h - btn_h - 14
        by_mode = by_commit - gap - btn_h
        by_outdir = by_mode - gap - btn_h
        by_imagedir = by_outdir - gap - btn_h

        btn_mode = Button(
            label=f"Mode: {'Polygon' if self.annotation_mode == 'polygon' else 'SAM points'}",
            action="toggle_annotation_mode",
            rect=(bx, by_mode, btn_w, btn_h),
        )
        self.buttons.append(btn_mode)
        hover = (self.hover == btn_mode.action)
        fill = (212, 235, 239) if hover else (227, 242, 245)
        edge = (86, 146, 160) if self.annotation_mode == "polygon" else (132, 170, 178)
        cv2.rectangle(canvas, (bx, by_mode), (bx + btn_w, by_mode + btn_h), fill, -1)
        cv2.rectangle(canvas, (bx, by_mode), (bx + btn_w, by_mode + btn_h), edge, 1)
        draw_cn_text(canvas, bx + 10, by_mode + 7, btn_mode.label, 18, (28, 92, 106))

        btn_imagedir = Button(
            label=f"Image dir: {self._short_path(self._current_image_dir(), 28)}",
            action="choose_image_dir",
            rect=(bx, by_imagedir, btn_w, btn_h),
        )
        self.buttons.append(btn_imagedir)
        hover = (self.hover == btn_imagedir.action)
        fill = (234, 244, 236) if hover else (242, 249, 243)
        edge = (98, 146, 112)
        cv2.rectangle(canvas, (bx, by_imagedir), (bx + btn_w, by_imagedir + btn_h), fill, -1)
        cv2.rectangle(canvas, (bx, by_imagedir), (bx + btn_w, by_imagedir + btn_h), edge, 1)
        draw_cn_text(canvas, bx + 10, by_imagedir + 7, btn_imagedir.label, 17, (46, 96, 56))

        btn_outdir = Button(
            label=f"Output dir: {self._short_path(self.output_dir, 28)}",
            action="choose_output_dir",
            rect=(bx, by_outdir, btn_w, btn_h),
        )
        self.buttons.append(btn_outdir)
        hover = (self.hover == btn_outdir.action)
        fill = (239, 239, 224) if hover else (246, 246, 234)
        edge = (144, 144, 92)
        cv2.rectangle(canvas, (bx, by_outdir), (bx + btn_w, by_outdir + btn_h), fill, -1)
        cv2.rectangle(canvas, (bx, by_outdir), (bx + btn_w, by_outdir + btn_h), edge, 1)
        draw_cn_text(canvas, bx + 10, by_outdir + 7, btn_outdir.label, 17, (86, 86, 40))

        btn_commit = Button(
            label=f"Submit polygon ({len(self.poly_points)} pts)",
            action="commit_polygon",
            rect=(bx, by_commit, btn_w, btn_h),
        )
        self.buttons.append(btn_commit)
        hover = (self.hover == btn_commit.action)
        fill = (210, 240, 224) if hover else (227, 245, 234)
        edge = (72, 140, 98) if self.annotation_mode == "polygon" else (150, 180, 160)
        text_col = (36, 98, 62)
        cv2.rectangle(canvas, (bx, by_commit), (bx + btn_w, by_commit + btn_h), fill, -1)
        cv2.rectangle(canvas, (bx, by_commit), (bx + btn_w, by_commit + btn_h), edge, 1)
        draw_cn_text(canvas, bx + 10, by_commit + 7, btn_commit.label, 18, text_col)

    def render(self):
        canvas = np.full((self.win_h, self.win_w, 3), (233, 244, 246), dtype=np.uint8)
        self.draw_view(canvas)
        self.draw_panel(canvas)
        cv2.imshow(self.win, canvas)
        self.last_render_time = time.time()

    def render_throttled(self, min_interval: float = 1 / 45):
        now = time.time()
        if now - self.last_render_time >= min_interval:
            self.render()

    def loop(self):
        print("A/D = prev/next (auto-save on flip: labels.json + vis.png + label_map.png + label_color.png + points.png; no wrap)")
        print("Q/E smooth kernel; T anti-bleed; G auto-neg; M mode; B eraser toggle")
        print("I = pick image dir; U = pick output dir; Enter = submit polygon")
        print("Polygon is hint-only; eraser uses round-radius slider; Esc or X to exit")
        while True:
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                break
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k in (ord("m"), ord("M")):
                self._toggle_annotation_mode()
                self.render()
                continue
            if k in (ord("b"), ord("B")):
                self._toggle_eraser_mode()
                self.render()
                continue
            if k in (ord("i"), ord("I")):
                self._choose_image_dir()
                self.render()
                continue
            if k in (ord("u"), ord("U")):
                self._choose_output_dir()
                self.render()
                continue
            if k in (13, 10):
                if self.annotation_mode == "polygon":
                    self._apply_polygon_to_mask()
                    self.render()
                    continue
            if k == ord("1"):
                self._sync_active_prompts_to_bank()
                self.current_label = "femur"
                self.points_xy, self.labels = [], []
                self.poly_points = []
                self.cur_logits, self.cur_score = None, None
                self.cur_mask = self.masks["femur"]
                self.last_auto_neg_used = 0
                self.set_status("Class: femur (prompts cleared)")
                self.render()
                continue
            if k == ord("2"):
                self._sync_active_prompts_to_bank()
                self.current_label = "tibia"
                self.points_xy, self.labels = [], []
                self.poly_points = []
                self.cur_logits, self.cur_score = None, None
                self.cur_mask = self.masks["tibia"]
                self.last_auto_neg_used = 0
                self.set_status("Class: tibia (prompts cleared)")
                self.render()
                continue
            if k in (ord("r"), ord("R")):
                self.points_xy, self.labels = [], []
                self.poly_points = []
                self.cur_mask, self.cur_logits, self.cur_score = None, None, None
                self.masks[self.current_label] = None
                self.prompt_bank[self.current_label] = {"points": [], "labels": []}
                self.last_auto_neg_used = 0
                self.set_status(f"Reset current class: {self.current_label}")
                self.render()
                continue
            if k in (ord("c"), ord("C")):
                self.points_xy, self.labels = [], []
                self.poly_points = []
                self.cur_mask, self.cur_logits, self.cur_score = None, None, None
                self.masks = {"femur": None, "tibia": None}
                self.prompt_bank = {
                    "femur": {"points": [], "labels": []},
                    "tibia": {"points": [], "labels": []},
                }
                self.last_auto_neg_used = 0
                self.set_status("All classes cleared")
                self.render()
                continue
            if k in (ord("t"), ord("T")):
                self.exclude_other_class = not self.exclude_other_class
                self.rebuild_from_logits()
                self.set_status(f"Anti-bleed: {self.exclude_other_class}")
                self.render()
                continue
            if k in (ord("g"), ord("G")):
                self.auto_neg_from_other = not self.auto_neg_from_other
                if self.points_xy:
                    self.predict()
                else:
                    self.last_auto_neg_used = 0
                self.set_status(f"Auto-neg: {self.auto_neg_from_other}")
                self.render()
                continue
            if k in (27, ord("x"), ord("X")):
                self.save_auto_bundle(auto=True)
                break
            elif k in (ord("a"), ord("A")):
                self.switch_image(-1)
            elif k in (ord("d"), ord("D")):
                self.switch_image(1)
            elif k in (ord("z"), ord("Z")):
                if self.annotation_mode == "polygon":
                    if self.poly_points:
                        self.poly_points.pop()
                        self.set_status(f"Polygon points: {len(self.poly_points)}")
                elif self.points_xy:
                    self.points_xy.pop()
                    self.labels.pop()
                    self.predict()
            elif k in (ord("s"), ord("S")):
                self.save_auto_bundle(auto=False)
            elif k == ord("5"):
                self.preprocess_mode = 0; self.load_current(reset_masks=True)
            elif k == ord("6"):
                self.preprocess_mode = 1; self.load_current(reset_masks=True)
            elif k == ord("7"):
                self.preprocess_mode = 2; self.load_current(reset_masks=True)
            elif k == ord("8"):
                self.preprocess_mode = 3; self.load_current(reset_masks=True)
            elif k == ord("9"):
                self.preprocess_mode = 4; self.load_current(reset_masks=True)
            elif k == ord("["):
                self.threshold = max(-2.0, self.threshold - 0.05)
                self.rebuild_from_logits()
            elif k == ord("]"):
                self.threshold = min(2.0, self.threshold + 0.05)
                self.rebuild_from_logits()
            elif k in (ord("q"), ord("Q"), ord(",")):
                self.smooth_ksize = max(0, self.smooth_ksize - 2)
                self.rebuild_from_logits()
                self.set_status(f"Smooth kernel: {self.smooth_ksize}")
            elif k in (ord("e"), ord("E"), ord(".")):
                self.smooth_ksize = min(121, self.smooth_ksize + 2)
                if self.smooth_ksize == 2:
                    self.smooth_ksize = 3
                self.rebuild_from_logits()
                self.set_status(f"Smooth kernel: {self.smooth_ksize}")
            self.render()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--ft-ckpt", default=DEFAULT_FT_CKPT)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--bpe", default=DEFAULT_BPE)
    ap.add_argument("--device", default="", choices=["", "cpu", "cuda"])
    ap.add_argument("--out-dir", default="", help="Annotation output directory; default = <image_dir>/sam3_outputs")
    args = ap.parse_args()

    app = App(
        ckpt=args.ckpt,
        ft_ckpt=args.ft_ckpt,
        bpe=args.bpe,
        image=args.image,
        device=args.device,
        out_dir=args.out_dir,
    )
    app.loop()
