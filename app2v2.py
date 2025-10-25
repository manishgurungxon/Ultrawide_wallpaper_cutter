"""
Ultrawide Wallpaper Cutter — Streamlit App (full, production-ready)

Features
- Crop or letterbox to common monitor ratios (16:9, 21:9, 32:9, 16:10, etc.)
- Presets & custom ratios, suggested resolutions
- Modes: Fill (crop) or Fit (letterbox/pillarbox)
- Smart crop (edge-aware + optional face keep) with OpenCV when available
- Manual crops: 9-point gravity, focal point sliders, or draw-box (interactive)
- Output options: PNG/JPEG, JPEG quality, ICC profile preservation, avoid upscaling
- Batch ZIP and per-image downloads (unique filenames every run)
- Sidebar “About & Legal” + footer disclaimer

Run
  pip install streamlit pillow numpy
  pip install opencv-python-headless  # optional for smart crop
  pip install streamlit-cropper       # optional for draw-box tool
  streamlit run app.py
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Optional deps (OpenCV for saliency/face, Cropper for draw-box)
try:
    import cv2
    _CV2_OK = True
except Exception:
    _CV2_OK = False

try:
    from streamlit_cropper import st_cropper
    _CROPPER_OK = True
except Exception:
    _CROPPER_OK = False


# ----------------------------
# Data & helpers
# ----------------------------

@dataclass(frozen=True)
class AspectPreset:
    name: str
    ratio_w: int
    ratio_h: int
    resolutions: Tuple[Tuple[int, int], ...]


ASPECT_PRESETS: Tuple[AspectPreset, ...] = (
    AspectPreset("32:9 (Super Ultrawide)", 32, 9, (
        (5120, 1440),
        (7680, 2160),
        (3840, 1080),
        (6880, 2160),
    )),
    AspectPreset("21:9 (Ultrawide)", 21, 9, (
        (3440, 1440),
        (2560, 1080),
        (3840, 1600),
        (5120, 2160),
    )),
    AspectPreset("16:9 (Widescreen)", 16, 9, (
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
        (1280, 720),
    )),
    AspectPreset("16:10", 16, 10, (
        (1920, 1200),
        (2560, 1600),
        (3840, 2400),
        (2880, 1800),
    )),
    AspectPreset("4:3", 4, 3, (
        (1024, 768),
        (1600, 1200),
        (2048, 1536),
    )),
    AspectPreset("3:2", 3, 2, (
        (3000, 2000),
        (4500, 3000),
    )),
)

GRAVITIES = [
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
]
CROP_STRATEGIES = [
    "Center",
    "Edge-aware (saliency)",
    "Manual (9-point)",
    "Manual (focal point)",
    "Manual (draw box)",
]


# ---- Core image utils ----

def apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Normalize EXIF orientation and keep RGB/RGBA when appropriate."""
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def target_dims_for_ratio(src_w: int, src_h: int, ratio_w: int, ratio_h: int) -> Tuple[int, int, float]:
    target_ratio = ratio_w / ratio_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        th = src_h
        tw = int(round(th * target_ratio))
    else:
        tw = src_w
        th = int(round(tw / target_ratio))
    return tw, th, target_ratio


def crop_box_with_gravity(src_w: int, src_h: int, tw: int, th: int, gravity: str = "center") -> Tuple[int, int, int, int]:
    mapping = {
        "top-left": (0, 0),
        "top-center": ((src_w - tw) // 2, 0),
        "top-right": (src_w - tw, 0),
        "center-left": (0, (src_h - th) // 2),
        "center": ((src_w - tw) // 2, (src_h - th) // 2),
        "center-right": (src_w - tw, (src_h - th) // 2),
        "bottom-left": (0, src_h - th),
        "bottom-center": ((src_w - tw) // 2, src_h - th),
        "bottom-right": (src_w - tw, src_h - th),
    }
    if gravity not in mapping:
        gravity = "center"
    left, top = mapping[gravity]
    return (left, top, left + tw, top + th)


def crop_box_centered_aspect(src_w: int, src_h: int, ratio_w: int, ratio_h: int, cx: int, cy: int) -> Tuple[int, int, int, int]:
    """Largest box of the given aspect centered at (cx, cy) within image bounds."""
    target_ratio = ratio_w / ratio_h
    max_left = cx
    max_right = src_w - cx
    max_top = cy
    max_bottom = src_h - cy
    a1 = min(max_left, max_right)
    a2 = min(max_top, max_bottom) * target_ratio
    a = max(1, int(min(a1, a2)))
    half_w = a
    half_h = int(round(a / target_ratio))
    x1 = max(0, cx - half_w)
    x2 = min(src_w, cx + half_w)
    y1 = max(0, cy - half_h)
    y2 = min(src_h, cy + half_h)
    tw = x2 - x1
    th = y2 - y1
    if abs((tw / max(1, th)) - target_ratio) > 1e-3:
        tw2, th2, _ = target_dims_for_ratio(src_w, src_h, ratio_w, ratio_h)
        box = crop_box_with_gravity(src_w, src_h, tw2, th2, "center")
        return _nudge_box_to_include_point(box, (cx, cy), (src_w, src_h))
    return (x1, y1, x2, y2)


def resize_high_quality(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.Resampling.LANCZOS)


def fit_letterbox(img: Image.Image, out_w: int, out_h: int, bgcolor=(0, 0, 0)) -> Image.Image:
    src_w, src_h = img.size
    scale = min(out_w / src_w, out_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = resize_high_quality(img, (new_w, new_h))
    mode = "RGBA" if img.mode == "RGBA" else "RGB"
    canvas = Image.new(mode, (out_w, out_h), (*bgcolor, 0) if mode == "RGBA" else bgcolor)
    paste_x = (out_w - new_w) // 2
    paste_y = (out_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y), resized if mode == "RGBA" else None)
    return canvas


# ---- Edge-aware crop helpers (OpenCV) ----

def _compute_integral(img_gray: np.ndarray) -> np.ndarray:
    return cv2.integral(img_gray)[1:, 1:]


def _box_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x2, y2 = x + w - 1, y + h - 1
    A = ii[y2, x2]
    B = ii[y - 1, x2] if y > 0 else 0
    C = ii[y2, x - 1] if x > 0 else 0
    D = ii[y - 1, x - 1] if (x > 0 and y > 0) else 0
    return float(A - B - C + D)


def _smart_crop_edge(img: Image.Image, ratio_w: int, ratio_h: int, stride: int = 32) -> Image.Image:
    src_w, src_h = img.size
    tw, th, _ = target_dims_for_ratio(src_w, src_h, ratio_w, ratio_h)
    arr = np.array(img.convert("L"))
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.0)
    ii = _compute_integral(mag)

    best = None
    max_score = -1e18
    sx = max(8, min(stride, max(1, tw // 4)))
    sy = max(8, min(stride, max(1, th // 4)))
    for y in range(0, max(1, src_h - th + 1), sy):
        for x in range(0, max(1, src_w - tw + 1), sx):
            score = _box_sum(ii, x, y, tw, th)
            if score > max_score:
                max_score = score
                best = (x, y, x + tw, y + th)

    if best is None:
        best = crop_box_with_gravity(src_w, src_h, tw, th, "center")
    return img.crop(best)


@st.cache_resource(show_spinner=False)
def _get_face_cascade():
    if not _CV2_OK:
        return None
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        return None


def _detect_faces_box(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    cascade = _get_face_cascade()
    if cascade is None:
        return None
    try:
        gray = np.array(img.convert("L"))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (int(x), int(y), int(x + w), int(y + h))
    except Exception:
        return None


def _nudge_box_to_include_point(box: Tuple[int, int, int, int], pt: Tuple[int, int], bounds: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    px, py = pt
    w, h = x2 - x1, y2 - y1
    src_w, src_h = bounds
    if x1 <= px <= x2 and y1 <= py <= y2:
        return box
    cx = min(max(px - w // 2, 0), src_w - w)
    cy = min(max(py - h // 2, 0), src_h - h)
    return (cx, cy, cx + w, cy + h)


def fill_smart_crop(img: Image.Image, ratio_w: int, ratio_h: int, gravity: str = "center") -> Image.Image:
    src_w, src_h = img.size
    tw, th, _ = target_dims_for_ratio(src_w, src_h, ratio_w, ratio_h)
    box = crop_box_with_gravity(src_w, src_h, tw, th, gravity)
    return img.crop(box)


def fill_manual_focal(img: Image.Image, ratio_w: int, ratio_h: int, fx: float, fy: float) -> Image.Image:
    src_w, src_h = img.size
    cx = int(round(fx * (src_w - 1)))
    cy = int(round(fy * (src_h - 1)))
    box = crop_box_centered_aspect(src_w, src_h, ratio_w, ratio_h, cx, cy)
    return img.crop(box)


def fill_smart_crop_auto(
    img: Image.Image,
    ratio_w: int,
    ratio_h: int,
    strategy: str = "Center",
    gravity: str = "center",
    focal: Optional[Tuple[float, float]] = None,
    roi_center: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    src_w, src_h = img.size
    if strategy == "Edge-aware (saliency)" and _CV2_OK:
        base = _smart_crop_edge(img, ratio_w, ratio_h)
        fbox = _detect_faces_box(img)
        if fbox is not None:
            tw, th, _ = target_dims_for_ratio(src_w, src_h, ratio_w, ratio_h)
            x1, y1, x2, y2 = fbox
            fx, fy = (x1 + x2) // 2, (y1 + y2) // 2
            best_box = crop_box_with_gravity(src_w, src_h, tw, th, "center")
            best_box = _nudge_box_to_include_point(best_box, (fx, fy), (src_w, src_h))
            base = img.crop(best_box)
        return base
    elif strategy == "Manual (9-point)":
        return fill_smart_crop(img, ratio_w, ratio_h, gravity=gravity)
    elif strategy == "Manual (focal point)" and focal is not None:
        return fill_manual_focal(img, ratio_w, ratio_h, focal[0], focal[1])
    elif strategy == "Manual (draw box)" and roi_center is not None:
        cx, cy = roi_center
        box = crop_box_centered_aspect(src_w, src_h, ratio_w, ratio_h, cx, cy)
        return img.crop(box)
    else:
        return fill_smart_crop(img, ratio_w, ratio_h, gravity="center")


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Ultrawide Wallpaper Cutter", layout="wide")

# Sidebar About/Legal
with st.sidebar:
    st.markdown("### About & Legal")
    st.info(
        "Ultrawide Wallpaper Cutter helps you crop or fit wallpapers for ultrawide displays.\n\n"
        "**Open-source components:** Streamlit (Apache 2.0), Pillow (liberal PIL license), NumPy (BSD), "
        "OpenCV (Apache 2.0), streamlit-cropper (MIT).\n\n"
        "**Privacy:** Images are processed locally in your session. No persistent uploads/storage."
    )
    st.caption("Developed with ❤️ by VNepal")

st.title("🖼️ Ultrawide Wallpaper Cutter")
colA, colB = st.columns([1, 1])

with colA:
    st.markdown("#### 1) Upload images")
    files = st.file_uploader(
        "Upload JPG/PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

with colB:
    st.markdown("#### 2) Choose aspect ratio & output size")
    preset_names = [p.name for p in ASPECT_PRESETS] + ["Custom…"]
    preset_choice = st.selectbox("Aspect ratio preset", preset_names, index=0)

    if preset_choice != "Custom…":
        preset = next(p for p in ASPECT_PRESETS if p.name == preset_choice)
        ratio_w, ratio_h = preset.ratio_w, preset.ratio_h
        res_options = [f"{w} × {h}" for w, h in preset.resolutions]
        res_options.insert(0, "Keep source size (after crop/fit)")
        res_label = st.selectbox("Output resolution", res_options, index=1)
        if res_label != "Keep source size (after crop/fit)":
            out_w, out_h = map(int, res_label.replace(" × ", "x").split("x"))
        else:
            out_w = out_h = None
    else:
        ratio_w = st.number_input("Ratio width", min_value=1, value=32)
        ratio_h = st.number_input("Ratio height", min_value=1, value=9)
        out_w = st.number_input("Output width (px, optional)", min_value=0, value=0)
        out_h = st.number_input("Output height (px, optional)", min_value=0, value=0)
        if out_w == 0 or out_h == 0:
            out_w = out_h = None

st.markdown("#### 3) Mode & options")
mode = st.radio("Mode", ["Fill (crop)", "Fit (letterbox)"])

strategy = st.selectbox(
    "Crop strategy",
    CROP_STRATEGIES,
    index=1 if (isinstance(preset_choice, str) and preset_choice.startswith("32:9")) else 0,
)

# Manual controls
selected_gravity: Optional[str] = None
fp_x = fp_y = None
roi_center: Optional[Tuple[int, int]] = None

if strategy == "Manual (9-point)":
    selected_gravity = st.select_slider("9-point gravity", options=GRAVITIES, value="center")
elif strategy == "Manual (focal point)":
    c1, c2 = st.columns(2)
    with c1:
        fp_x = st.slider("Focal X (%)", 0, 100, 50) / 100.0
    with c2:
        fp_y = st.slider("Focal Y (%)", 0, 100, 50) / 100.0
elif strategy == "Manual (draw box)":
    if not _CROPPER_OK:
        st.warning("Install optional dependency: pip install streamlit-cropper")
    elif not files or len(files) != 1:
        st.info("Draw-box crop works when exactly one image is uploaded. Please upload a single image.")
    else:
        _preview = Image.open(files[0])
        _preview = apply_exif_orientation(_preview)
        st.write("Drag/resize the crop box; it’s constrained to the chosen aspect ratio.")
        crop_box = st_cropper(
            _preview,
            realtime_update=True,
            box_color='#00FF00',
            aspect_ratio=(ratio_w, ratio_h),
            return_type='box',
            key='roi_box',
        )
        if crop_box:
            cx = int(crop_box['left'] + crop_box['width'] / 2)
            cy = int(crop_box['top'] + crop_box['height'] / 2)
            roi_center = (cx, cy)

background = st.color_picker("Letterbox background (Fit mode)", value="#000000")
# Convert hex to RGB tuple
bg_rgb = tuple(int(background[i: i + 2], 16) for i in (1, 3, 5))

# Advanced output options
st.markdown("#### 4) Output format")
fmt = st.selectbox("Format", ["PNG", "JPEG"], index=0)
jpeg_q = st.slider("JPEG quality", 60, 100, 92)
no_upscale = st.checkbox("Avoid upscaling when resizing", value=True)

# Helper to validate/adjust custom output sizes
def _maybe_adjust_size_for_ratio(w: Optional[int], h: Optional[int], rw: int, rh: int) -> Tuple[Optional[int], Optional[int]]:
    if w is None or h is None:
        return w, h
    target_ratio = rw / rh
    actual_ratio = w / h
    if abs(actual_ratio - target_ratio) < 1e-3:
        return w, h
    adj_h = int(round(w / target_ratio))
    return w, max(1, adj_h)

if preset_choice == "Custom…" and (out_w and out_h):
    adj_w, adj_h = _maybe_adjust_size_for_ratio(out_w, out_h, ratio_w, ratio_h)
    if (adj_w, adj_h) != (out_w, out_h):
        st.info(f"Adjusted output to {adj_w}×{adj_h} to match the {ratio_w}:{ratio_h} aspect.")
        out_w, out_h = adj_w, adj_h

st.markdown("---")
run = st.button("▶️ Process Images")

# ----------------------------
# Processing
# ----------------------------

outputs: List[Tuple[str, bytes]] = []  # (filename, data)

def _unique_name(base: str, suffix: str, ext: str) -> str:
    ts = int(time.time())
    uid = uuid.uuid4().hex[:8]
    return f"{base}{suffix}_{ts}_{uid}.{ext}"

if run:
    if not files:
        st.warning("Please upload at least one image.")
    else:
        if (strategy == "Edge-aware (saliency)") and not _CV2_OK:
            st.warning("Edge-aware saliency requires OpenCV. Falling back to Center crop.")
        with st.spinner("Processing…"):
            for f in files:
                try:
                    img = Image.open(f)
                    icc = img.info.get("icc_profile")
                    img = apply_exif_orientation(img)
                    base_name = f.name.rsplit(".", 1)[0]

                    if mode.startswith("Fill"):
                        cropped = fill_smart_crop_auto(
                            img, ratio_w, ratio_h,
                            strategy=strategy,
                            gravity=selected_gravity or "center",
                            focal=(fp_x, fp_y) if (fp_x is not None and fp_y is not None) else None,
                            roi_center=roi_center if roi_center is not None else None,
                        )
                        if out_w and out_h:
                            if no_upscale and (out_w > cropped.width or out_h > cropped.height):
                                scale = min(cropped.width / out_w, cropped.height / out_h)
                                out_w2 = max(1, int(out_w * scale))
                                out_h2 = max(1, int(out_h * scale))
                                final = resize_high_quality(cropped, (out_w2, out_h2))
                            else:
                                final = resize_high_quality(cropped, (out_w, out_h))
                        else:
                            final = cropped
                        mode_suffix = f"_{final.size[0]}x{final.size[1]}_fill"
                    else:
                        if out_w and out_h:
                            final = fit_letterbox(img, out_w, out_h, bgcolor=bg_rgb)
                        else:
                            src_w, src_h = img.size
                            auto_h = src_h
                            auto_w = int(round(auto_h * (ratio_w / ratio_h)))
                            final = fit_letterbox(img, auto_w, auto_h, bgcolor=bg_rgb)
                        mode_suffix = f"_{final.size[0]}x{final.size[1]}_fit"

                    # Preview
                    st.image(final, caption=f"{f.name} → {final.size[0]}×{final.size[1]}")

                    # Save buffer
                    out_bytes = io.BytesIO()
                    if fmt == "PNG":
                        save_kwargs = {"format": "PNG", "compress_level": 6}
                        if icc:
                            save_kwargs["icc_profile"] = icc
                        final.save(out_bytes, **save_kwargs)
                        ext = "png"
                    else:
                        save_img = final.convert("RGB") if final.mode == "RGBA" else final
                        save_kwargs = {"format": "JPEG", "quality": jpeg_q, "optimize": True}
                        if icc:
                            save_kwargs["icc_profile"] = icc
                        save_img.save(out_bytes, **save_kwargs)
                        ext = "jpg"

                    filename = _unique_name(base_name, mode_suffix, ext)
                    outputs.append((filename, out_bytes.getvalue()))
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")

        if outputs:
            st.markdown("##### Downloads")
            # Individual downloads (unique filenames)
            for fn, data in outputs:
                with st.container(border=True):
                    st.write(fn)
                    st.download_button(
                        label="⬇️ Download image",
                        data=data,
                        file_name=fn,
                        mime="image/png" if fn.lower().endswith(".png") else "image/jpeg",
                        use_container_width=True,
                    )

            # Batch ZIP with unique name
            zip_buf = io.BytesIO()
            zip_name = f"wallpapers_{int(time.time())}_{uuid.uuid4().hex[:6]}.zip"
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fn, data in outputs:
                    zf.writestr(fn, data)
            st.download_button(
                label="📦 Download ZIP of all images",
                data=zip_buf.getvalue(),
                file_name=zip_name,
                mime="application/zip",
                use_container_width=True,
            )

# ----------------------------
# Footer (Tips + Legal)
# ----------------------------
st.markdown(
    """
---
**Tips**
- *Fill (crop)* is best for true edge-to-edge wallpapers.
- *Fit (letterbox)* preserves the entire image with bars as needed.
- For 32:9 monitors, popular sizes are **5120×1440** and **3840×1080**.

**Notes**
- Saliency/face features need OpenCV; if missing, the app falls back to center crop.
- Draw-box requires `streamlit-cropper` and works with a single uploaded image.

**Disclaimer**
This app processes user-supplied images only. The developer does not claim ownership or rights to the content users upload.
All trademarks and images belong to their respective owners. Users are responsible for ensuring they have rights to any images
they process or distribute with this app.
"""
)
