from __future__ import annotations

from typing import List, Tuple, Dict
import cv2
import numpy as np

from .config import PipelineConfig
from .masks import (
    resize_keep_aspect,
    convert_color_space,
    apply_clahe_to_luminance,
    chroma_diff_mask,
    edge_diff_mask,
    texture_diff_mask,
    local_contrast_diff_mask,
    shadow_mask,
    combine_masks,
    apply_morphology,
    apply_edge_fill,
)
from .roi import build_floor_roi_mask, floor_mask_multicluster
from .boxes import (
    find_bounding_boxes,
    filter_boxes,
    merge_boxes,
    dilate_mask,
    refine_boxes_to_mask,
    draw_boxes,
)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure an image is 3-channel BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img
    return img[:, :, :3]


def overlay_label(img: np.ndarray, text: str, font_scale: float) -> np.ndarray:
    """Draw a label bar on the top of an image."""
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
        cv2.LINE_AA,
    )
    return out


def build_debug_grid(items: List[Tuple[str, np.ndarray]], cfg: PipelineConfig) -> np.ndarray:
    """Build a labeled image grid for debug visualization."""
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
    """Display the debug grid window if enabled."""
    if not cfg.show_debug_windows:
        return
    grid = build_debug_grid(items, cfg)
    cv2.imshow(cfg.debug_window_name, grid)


def run_pipeline_images(ref_bgr: np.ndarray, cur_bgr: np.ndarray, cfg: PipelineConfig) -> Dict[str, np.ndarray]:
    """Run the full pipeline on already loaded BGR images."""
    # 1) Resize to a common target width (optional).
    if cfg.enable_resize:
        ref_bgr = resize_keep_aspect(ref_bgr, cfg.target_width)
        cur_bgr = resize_keep_aspect(cur_bgr, cfg.target_width)

    # 2) Convert to selected color space and normalize luminance (optional).
    ref_cs = convert_color_space(ref_bgr, cfg.color_space)
    cur_cs = convert_color_space(cur_bgr, cfg.color_space)

    if cfg.enable_clahe_on_luminance:
        ref_cs = apply_clahe_to_luminance(ref_cs, cfg.color_space)
        cur_cs = apply_clahe_to_luminance(cur_cs, cfg.color_space)

    # 3) Build floor ROI mask (mask override or multicluster detection).
    floor_mask = None
    if cfg.enable_floor_roi:
        mode = cfg.floor_roi_mode.lower()
        if mode == "mask":
            if not cfg.floor_mask_override_path:
                raise ValueError("floor_roi_mode=mask requires floor_mask_override_path")
            override = cv2.imread(cfg.floor_mask_override_path, cv2.IMREAD_GRAYSCALE)
            if override is None:
                raise FileNotFoundError(f"Could not read floor mask: {cfg.floor_mask_override_path}")
            if override.shape[:2] != ref_bgr.shape[:2]:
                override = cv2.resize(
                    override,
                    (ref_bgr.shape[1], ref_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            floor_mask = (override > 127).astype(np.uint8) * 255
        elif mode == "detection":
            floor_mask = floor_mask_multicluster(ref_bgr, cfg)
        else:
            raise ValueError(f"Unsupported floor_roi_mode: {cfg.floor_roi_mode}")

    # 4) Compute change-detection masks (chroma/edge/texture/local contrast).
    m_chroma = None
    if cfg.enable_chroma_diff:
        m_chroma = chroma_diff_mask(ref_cs, cur_cs, cfg.color_space, cfg.chroma_diff_thresh)

    m_edge = None
    if cfg.enable_edge_diff:
        m_edge = edge_diff_mask(ref_bgr, cur_bgr, cfg.edge_diff_thresh)

    m_texture = None
    if cfg.enable_texture_diff:
        m_texture = texture_diff_mask(ref_bgr, cur_bgr, cfg.texture_diff_thresh)

    m_local_contrast = None
    if cfg.enable_local_contrast_diff:
        m_local_contrast = local_contrast_diff_mask(
            ref_bgr, cur_bgr, cfg.local_contrast_diff_thresh, cfg.local_contrast_ksize
        )

    # 5) Optional shadow mask to suppress lighting changes.
    m_shadow = None
    if cfg.enable_shadow_mask:
        m_shadow = shadow_mask(
            ref_cs,
            cur_cs,
            cfg.color_space,
            t_luma=cfg.shadow_luma_thresh,
            t_chroma_small=cfg.shadow_chroma_small_thresh,
        )

    # 6) Combine masks into a single change map.
    combined = None
    if m_chroma is not None or m_edge is not None:
        combined = combine_masks(m_chroma, m_edge, cfg.combine_mode)

    if m_texture is not None:
        combined = m_texture if combined is None else combine_masks(combined, m_texture, "OR")

    if m_local_contrast is not None:
        combined = m_local_contrast if combined is None else combine_masks(combined, m_local_contrast, "OR")

    if combined is None:
        raise ValueError("At least one change detection mask must be enabled")

    # 7) Apply shadow suppression and floor ROI restriction.
    if m_shadow is not None:
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(m_shadow))

    if floor_mask is not None:
        combined = cv2.bitwise_and(combined, floor_mask)

    # 8) Morphology/edge fill cleanup.
    cleaned = combined
    if cfg.enable_edge_fill:
        cleaned = apply_edge_fill(cleaned, cfg.edge_thicken_ksize, cfg.edge_fill_close_ksize)
    if cfg.enable_morphology:
        cleaned = apply_morphology(
            cleaned,
            close_ksize=cfg.morph_close_ksize,
            open_ksize=cfg.morph_open_ksize,
            iterations=cfg.morph_iterations,
        )

    # 9) Extract detections as boxes (with optional merge).
    if cfg.merge_close_boxes and cfg.merge_mode.lower() == "mask":
        merged_mask = dilate_mask(cleaned, cfg.merge_mask_ksize)
        boxes = find_bounding_boxes(merged_mask)
        boxes = refine_boxes_to_mask(boxes, cleaned)
    else:
        boxes = find_bounding_boxes(cleaned)

    boxes = filter_boxes(
        boxes,
        mask_shape=cleaned.shape[:2],
        floor_roi_mask=floor_mask,
        cfg=cfg,
    )
    if cfg.merge_close_boxes and cfg.merge_mode.lower() == "bbox":
        boxes = merge_boxes(boxes, cfg.merge_distance)

    # 10) Draw final result.
    result = draw_boxes(cur_bgr, boxes)

    # 11) Debug visualization (grid of intermediate masks).
    if cfg.debug_mode.lower() == "grid":
        items = [("Reference", ref_bgr), ("Current", cur_bgr)]
        if m_chroma is not None:
            items.append(("Mask - Chroma", m_chroma))
        if m_edge is not None:
            items.append(("Mask - Edge", m_edge))
        if m_texture is not None:
            items.append(("Mask - Texture", m_texture))
        if m_local_contrast is not None:
            items.append(("Mask - Local Contrast", m_local_contrast))
        if m_shadow is not None:
            items.append(("Mask - Shadow", m_shadow))
        if floor_mask is not None:
            items.append(("Floor ROI", floor_mask))
        items.append(("Mask - Combined", combined))
        items.append(("Mask - Cleaned", cleaned))
        items.append(("Result", result))
        debug_show_grid(items, cfg)

    return {
        "ref_bgr": ref_bgr,
        "cur_bgr": cur_bgr,
        "mask_chroma": m_chroma if m_chroma is not None else np.zeros_like(cleaned),
        "mask_edge": m_edge if m_edge is not None else np.zeros_like(cleaned),
        "mask_texture": m_texture if m_texture is not None else np.zeros_like(cleaned),
        "mask_local_contrast": m_local_contrast if m_local_contrast is not None else np.zeros_like(cleaned),
        "mask_shadow": m_shadow if m_shadow is not None else np.zeros_like(cleaned),
        "floor_mask": floor_mask if floor_mask is not None else np.zeros_like(cleaned),
        "mask_combined": combined,
        "mask_cleaned": cleaned,
        "result": result,
    }


def run_pipeline(ref_path: str, cur_path: str, cfg: PipelineConfig) -> Dict[str, np.ndarray]:
    """Load images from paths and run the full pipeline."""
    ref_bgr = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    cur_bgr = cv2.imread(cur_path, cv2.IMREAD_COLOR)

    if ref_bgr is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_path}")
    if cur_bgr is None:
        raise FileNotFoundError(f"Could not read current image: {cur_path}")

    return run_pipeline_images(ref_bgr, cur_bgr, cfg)
