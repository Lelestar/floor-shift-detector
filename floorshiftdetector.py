"""
FloorShiftDetector - baseline pipeline

Goal:
- Detect new/moved objects between a reference image and a current image.
- Focus only on objects on the floor (ROI).
- Produce bounding boxes around detected objects.

Dependencies:
    pip install opencv-python numpy
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class PipelineConfig:
    # --- Input / resize ---
    enable_resize: bool = True
    target_width: int = 960  # keep aspect ratio

    # --- Color space / normalization ---
    color_space: str = "LAB"  # "LAB" or "YCrCb" or "HSV"
    enable_clahe_on_luminance: bool = True  # helps with lighting variations

    # --- Floor ROI ---
    # Quick heuristic: floor is the bottom portion of the image.
    # You can later replace by a polygon mask or an auto segmentation.
    enable_floor_roi: bool = True
    floor_roi_ratio: float = 0.45  # bottom 45% of the image

    # --- Change detection (diff) ---
    enable_chroma_diff: bool = True
    enable_edge_diff: bool = True
    enable_texture_diff: bool = False
    enable_shadow_mask: bool = False

    # Thresholds (you will tune these)
    chroma_diff_thresh: int = 25
    edge_diff_thresh: int = 40
    texture_diff_thresh: int = 20
    shadow_luma_thresh: int = 25
    shadow_chroma_small_thresh: int = 10

    # Combine masks: OR is generally safer than AND
    combine_mode: str = "OR"  # "OR" or "AND"

    # --- Morphology cleanup ---
    enable_morphology: bool = True
    morph_close_ksize: int = 9
    morph_open_ksize: int = 5
    morph_iterations: int = 1

    # --- Connected components / contours filtering ---
    enable_area_filter: bool = True
    min_area: int = 350
    max_area: int = 200000

    # Bounding box filtering (optional heuristics)
    enable_bbox_filters: bool = True
    min_aspect_ratio: float = 0.15  # width/height
    max_aspect_ratio: float = 6.0
    must_be_in_floor_roi: bool = True  # bbox center must be in floor ROI

    # --- Debug ---
    show_debug_windows: bool = True
    debug_scale: float = 1.0  # 0.5 for smaller windows
    debug_mode: str = "grid"  # "grid" or "windows"
    debug_grid_cols: int = 3
    debug_grid_pad: int = 8
    debug_window_name: str = "Pipeline Debug"
    debug_label_font_scale: float = 0.5


# -----------------------------
# Utility functions
# -----------------------------

def resize_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / float(w)
    new_size = (target_width, int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def convert_color_space(bgr: np.ndarray, color_space: str) -> np.ndarray:
    if color_space.upper() == "LAB":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    if color_space.upper() == "YCRCB":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    if color_space.upper() == "HSV":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raise ValueError(f"Unsupported color_space: {color_space}")


def apply_clahe_to_luminance(img_cs: np.ndarray, color_space: str) -> np.ndarray:
    """
    Apply CLAHE to the luminance-like channel:
      - LAB: L channel
      - YCrCb: Y channel
      - HSV: V channel
    """
    out = img_cs.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    cs = color_space.upper()
    if cs == "LAB":
        out[:, :, 0] = clahe.apply(out[:, :, 0])
    elif cs == "YCRCB":
        out[:, :, 0] = clahe.apply(out[:, :, 0])
    elif cs == "HSV":
        out[:, :, 2] = clahe.apply(out[:, :, 2])
    else:
        raise ValueError(f"Unsupported for CLAHE: {color_space}")

    return out


def luminance_channel(img_cs: np.ndarray, color_space: str) -> np.ndarray:
    cs = color_space.upper()
    if cs in ["LAB", "YCRCB"]:
        return img_cs[:, :, 0]
    if cs == "HSV":
        return img_cs[:, :, 2]
    raise ValueError("Unsupported color space for luminance extraction")


def build_floor_roi_mask(shape_hw: Tuple[int, int], floor_ratio: float) -> np.ndarray:
    """
    Simple ROI: keep only bottom floor_ratio of image.
    Returns a binary mask (uint8 0/255).
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * (1.0 - floor_ratio))
    mask[y_start:h, :] = 255
    return mask


def shadow_mask(ref_cs: np.ndarray, cur_cs: np.ndarray, color_space: str,
                t_luma: int = 25, t_chroma_small: int = 10) -> np.ndarray:
    """
    Shadow-like pixels:
    - big luminance change
    - small chroma change
    """
    l_ref = luminance_channel(ref_cs, color_space)
    l_cur = luminance_channel(cur_cs, color_space)
    diff_l = cv2.absdiff(l_ref, l_cur)

    cs = color_space.upper()
    if cs == "LAB":
        ref_ab = ref_cs[:, :, 1:3]
        cur_ab = cur_cs[:, :, 1:3]
    elif cs == "YCRCB":
        ref_ab = ref_cs[:, :, 1:3]
        cur_ab = cur_cs[:, :, 1:3]
    elif cs == "HSV":
        ref_ab = ref_cs[:, :, 0:2]
        cur_ab = cur_cs[:, :, 0:2]
    else:
        raise ValueError("Unsupported color space for shadow mask")

    diff_c = cv2.absdiff(ref_ab, cur_ab)
    diff_csum = (diff_c[:, :, 0].astype(np.int16) + diff_c[:, :, 1].astype(np.int16)).astype(np.uint8)

    m_l = (diff_l > t_luma).astype(np.uint8) * 255
    m_csmall = (diff_csum < t_chroma_small).astype(np.uint8) * 255

    return cv2.bitwise_and(m_l, m_csmall)


def chroma_diff_mask(ref_cs: np.ndarray, cur_cs: np.ndarray, color_space: str, thresh: int) -> np.ndarray:
    """
    Compute difference on chroma-like channels to reduce sensitivity to illumination.
    For LAB: use a,b
    For YCrCb: use Cr,Cb
    For HSV: use H,S (often helps, but can be unstable with low saturation)
    """
    cs = color_space.upper()

    if cs == "LAB":
        ref_ab = ref_cs[:, :, 1:3]
        cur_ab = cur_cs[:, :, 1:3]
        diff = cv2.absdiff(ref_ab, cur_ab)
        diff_sum = diff[:, :, 0].astype(np.int16) + diff[:, :, 1].astype(np.int16)
    elif cs == "YCRCB":
        ref_crcb = ref_cs[:, :, 1:3]
        cur_crcb = cur_cs[:, :, 1:3]
        diff = cv2.absdiff(ref_crcb, cur_crcb)
        diff_sum = diff[:, :, 0].astype(np.int16) + diff[:, :, 1].astype(np.int16)
    elif cs == "HSV":
        ref_hs = ref_cs[:, :, 0:2]
        cur_hs = cur_cs[:, :, 0:2]
        diff = cv2.absdiff(ref_hs, cur_hs)
        diff_sum = diff[:, :, 0].astype(np.int16) + diff[:, :, 1].astype(np.int16)
    else:
        raise ValueError(f"Unsupported color space for chroma diff: {color_space}")

    diff_sum = np.clip(diff_sum, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(diff_sum, thresh, 255, cv2.THRESH_BINARY)
    return mask


def texture_diff_mask(ref_bgr: np.ndarray, cur_bgr: np.ndarray, thresh: int = 20) -> np.ndarray:
    """
    Texture/gradient difference mask (often detects same-color objects on similar backgrounds).
    """
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

    def sobel_mag(g):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mag

    mag_ref = sobel_mag(ref_gray)
    mag_cur = sobel_mag(cur_gray)

    diff = cv2.absdiff(mag_ref, mag_cur)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return mask


def edges_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Compute edge map using Sobel magnitude.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Mild blur helps reduce noise edges
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag


def edge_diff_mask(ref_bgr: np.ndarray, cur_bgr: np.ndarray, thresh: int) -> np.ndarray:
    """
    Compare edges to reduce sensitivity to smooth illumination changes.
    """
    e_ref = edges_mask(ref_bgr)
    e_cur = edges_mask(cur_bgr)
    diff = cv2.absdiff(e_ref, e_cur)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return mask


def combine_masks(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray], mode: str) -> np.ndarray:
    if mask_a is None and mask_b is None:
        raise ValueError("At least one mask must be provided")
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a

    if mode.upper() == "OR":
        return cv2.bitwise_or(mask_a, mask_b)
    if mode.upper() == "AND":
        return cv2.bitwise_and(mask_a, mask_b)
    raise ValueError(f"Unsupported combine_mode: {mode}")


def apply_morphology(mask: np.ndarray, close_ksize: int, open_ksize: int, iterations: int) -> np.ndarray:
    out = mask.copy()

    if close_ksize > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=iterations)

    if open_ksize > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=iterations)

    return out


def find_bounding_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (x, y, w, h) bounding boxes from contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes


def filter_boxes(
    boxes: List[Tuple[int, int, int, int]],
    mask_shape: Tuple[int, int],
    floor_roi_mask: Optional[np.ndarray],
    cfg: PipelineConfig
) -> List[Tuple[int, int, int, int]]:
    h_img, w_img = mask_shape
    filtered = []

    for (x, y, w, h) in boxes:
        area = w * h

        if cfg.enable_area_filter:
            if area < cfg.min_area or area > cfg.max_area:
                continue

        if cfg.enable_bbox_filters:
            aspect = w / float(h + 1e-6)
            if aspect < cfg.min_aspect_ratio or aspect > cfg.max_aspect_ratio:
                continue

            if cfg.must_be_in_floor_roi and floor_roi_mask is not None:
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                cx = np.clip(cx, 0, w_img - 1)
                cy = np.clip(cy, 0, h_img - 1)
                if floor_roi_mask[cy, cx] == 0:
                    continue

        filtered.append((x, y, w, h))

    return filtered


def draw_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


def debug_show(name: str, img: np.ndarray, cfg: PipelineConfig) -> None:
    if not cfg.show_debug_windows:
        return
    if cfg.debug_scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * cfg.debug_scale), int(h * cfg.debug_scale)))
    cv2.imshow(name, img)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img
    return img[:, :, :3]


def overlay_label(img: np.ndarray, text: str, font_scale: float) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    bar_h = max(20, int(24 * font_scale))
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (6, bar_h - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    return out


def build_debug_grid(items: List[Tuple[str, np.ndarray]], cfg: PipelineConfig) -> np.ndarray:
    if not items:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    base_img = ensure_bgr(items[0][1])
    h, w = base_img.shape[:2]
    tile_w = max(1, int(w * cfg.debug_scale))
    tile_h = max(1, int(h * cfg.debug_scale))

    cols = max(1, cfg.debug_grid_cols)
    rows = int(np.ceil(len(items) / float(cols)))
    pad = max(0, cfg.debug_grid_pad)

    grid_w = cols * tile_w + (cols - 1) * pad
    grid_h = rows * tile_h + (rows - 1) * pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, (name, img) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = c * (tile_w + pad)
        y0 = r * (tile_h + pad)
        tile = ensure_bgr(img)
        tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tile = overlay_label(tile, name, cfg.debug_label_font_scale)
        grid[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    return grid


def debug_show_grid(items: List[Tuple[str, np.ndarray]], cfg: PipelineConfig) -> None:
    if not cfg.show_debug_windows:
        return
    grid = build_debug_grid(items, cfg)
    cv2.imshow(cfg.debug_window_name, grid)


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline_images(ref_bgr: np.ndarray, cur_bgr: np.ndarray, cfg: PipelineConfig) -> Dict[str, np.ndarray]:
    # --- Optional resize to standardize / speed up ---
    if cfg.enable_resize:
        ref_bgr = resize_keep_aspect(ref_bgr, cfg.target_width)
        cur_bgr = resize_keep_aspect(cur_bgr, cfg.target_width)

    # --- Convert color space ---
    ref_cs = convert_color_space(ref_bgr, cfg.color_space)
    cur_cs = convert_color_space(cur_bgr, cfg.color_space)

    # --- Optional luminance normalization (helps with lighting) ---
    if cfg.enable_clahe_on_luminance:
        ref_cs = apply_clahe_to_luminance(ref_cs, cfg.color_space)
        cur_cs = apply_clahe_to_luminance(cur_cs, cfg.color_space)

    # --- Floor ROI mask ---
    floor_mask = None
    if cfg.enable_floor_roi:
        h, w = ref_bgr.shape[:2]
        floor_mask = build_floor_roi_mask((h, w), cfg.floor_roi_ratio)

    # --- Change detection masks ---
    m_chroma = None
    if cfg.enable_chroma_diff:
        m_chroma = chroma_diff_mask(ref_cs, cur_cs, cfg.color_space, cfg.chroma_diff_thresh)

    m_edge = None
    if cfg.enable_edge_diff:
        m_edge = edge_diff_mask(ref_bgr, cur_bgr, cfg.edge_diff_thresh)

    m_texture = None
    if cfg.enable_texture_diff:
        m_texture = texture_diff_mask(ref_bgr, cur_bgr, cfg.texture_diff_thresh)

    m_shadow = None
    if cfg.enable_shadow_mask:
        m_shadow = shadow_mask(
            ref_cs,
            cur_cs,
            cfg.color_space,
            t_luma=cfg.shadow_luma_thresh,
            t_chroma_small=cfg.shadow_chroma_small_thresh
        )

    # --- Combine masks ---
    combined = None
    if m_chroma is not None or m_edge is not None:
        combined = combine_masks(m_chroma, m_edge, cfg.combine_mode)

    if m_texture is not None:
        combined = m_texture if combined is None else combine_masks(combined, m_texture, "OR")

    if combined is None:
        raise ValueError("At least one change detection mask must be enabled")

    if m_shadow is not None:
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(m_shadow))

    # Apply floor ROI if enabled
    if floor_mask is not None:
        combined = cv2.bitwise_and(combined, floor_mask)

    # --- Morphology cleanup ---
    cleaned = combined
    if cfg.enable_morphology:
        cleaned = apply_morphology(
            cleaned,
            close_ksize=cfg.morph_close_ksize,
            open_ksize=cfg.morph_open_ksize,
            iterations=cfg.morph_iterations
        )

    # --- Find contours / boxes ---
    boxes = find_bounding_boxes(cleaned)

    # --- Filter boxes ---
    boxes = filter_boxes(
        boxes,
        mask_shape=cleaned.shape[:2],
        floor_roi_mask=floor_mask,
        cfg=cfg
    )

    # --- Draw results ---
    result = draw_boxes(cur_bgr, boxes)

    # --- Debug visualization ---
    if cfg.debug_mode.lower() == "windows":
        debug_show("Reference (BGR)", ref_bgr, cfg)
        debug_show("Current (BGR)", cur_bgr, cfg)
        if m_chroma is not None:
            debug_show("Mask - Chroma Diff", m_chroma, cfg)
        if m_edge is not None:
            debug_show("Mask - Edge Diff", m_edge, cfg)
        if m_texture is not None:
            debug_show("Mask - Texture Diff", m_texture, cfg)
        if m_shadow is not None:
            debug_show("Mask - Shadow", m_shadow, cfg)
        if floor_mask is not None:
            debug_show("Floor ROI Mask", floor_mask, cfg)
        debug_show("Mask - Combined", combined, cfg)
        debug_show("Mask - Cleaned", cleaned, cfg)
        debug_show("Result (Bounding Boxes)", result, cfg)
    else:
        items = [
            ("Reference", ref_bgr),
            ("Current", cur_bgr),
            ("Mask - Chroma", m_chroma if m_chroma is not None else np.zeros_like(cleaned)),
            ("Mask - Edge", m_edge if m_edge is not None else np.zeros_like(cleaned)),
            ("Mask - Texture", m_texture if m_texture is not None else np.zeros_like(cleaned)),
            ("Mask - Shadow", m_shadow if m_shadow is not None else np.zeros_like(cleaned)),
            ("Floor ROI", floor_mask if floor_mask is not None else np.zeros_like(cleaned)),
            ("Mask - Combined", combined),
            ("Mask - Cleaned", cleaned),
            ("Result", result)
        ]
        debug_show_grid(items, cfg)

    return {
        "ref_bgr": ref_bgr,
        "cur_bgr": cur_bgr,
        "mask_chroma": m_chroma if m_chroma is not None else np.zeros_like(cleaned),
        "mask_edge": m_edge if m_edge is not None else np.zeros_like(cleaned),
        "mask_texture": m_texture if m_texture is not None else np.zeros_like(cleaned),
        "mask_shadow": m_shadow if m_shadow is not None else np.zeros_like(cleaned),
        "floor_mask": floor_mask if floor_mask is not None else np.zeros_like(cleaned),
        "mask_combined": combined,
        "mask_cleaned": cleaned,
        "result": result
    }


def run_pipeline(ref_path: str, cur_path: str, cfg: PipelineConfig) -> Dict[str, np.ndarray]:
    # --- Load images ---
    ref_bgr = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    cur_bgr = cv2.imread(cur_path, cv2.IMREAD_COLOR)

    if ref_bgr is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_path}")
    if cur_bgr is None:
        raise FileNotFoundError(f"Could not read current image: {cur_path}")

    return run_pipeline_images(ref_bgr, cur_bgr, cfg)


def main() -> None:
    # Example usage (replace paths with your own)
    ref_path = "Images/Chambre/Reference.jpg"
    cur_path = "Images/Chambre/IMG_6567.jpg"

    cfg = PipelineConfig(
        enable_resize=True,
        target_width=960,
        color_space="LAB",
        enable_clahe_on_luminance=True,
        enable_floor_roi=True,
        floor_roi_ratio=0.45,
        enable_chroma_diff=True,
        enable_edge_diff=True,
        enable_texture_diff=False,
        enable_shadow_mask=False,
        chroma_diff_thresh=25,
        edge_diff_thresh=40,
        texture_diff_thresh=20,
        shadow_luma_thresh=25,
        shadow_chroma_small_thresh=10,
        combine_mode="OR",
        enable_morphology=True,
        morph_close_ksize=9,
        morph_open_ksize=5,
        morph_iterations=1,
        min_area=350,
        max_area=200000,
        show_debug_windows=True,
        debug_mode="grid",
        debug_scale=0.5,
        debug_grid_cols=3
    )

    _ = run_pipeline(ref_path, cur_path, cfg)

    # Press any key to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
