from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the floor shift detection pipeline."""
    # --- Input / resize ---
    enable_resize: bool = True  # resize inputs to target_width
    target_width: int = 960  # keep aspect ratio

    # --- Color space / normalization ---
    color_space: str = "LAB"  # "LAB" or "YCrCb" or "HSV"
    enable_clahe_on_luminance: bool = True  # helps with lighting variations

    # --- Floor ROI ---
    enable_floor_roi: bool = True  # enable floor region restriction
    floor_roi_mode: str = "detection"  # "detection" or "mask"
    floor_roi_ratio: float = 0.45  # bottom 45% of the image
    floor_seed_ratio: float = 0.38  # bottom band height for multicluster seed
    floor_seed_x_ratio: float = 0.92  # seed band width ratio (centered)
    floor_seed_luma_clip_low: float = 0.05  # remove darkest seed pixels (quantile)
    floor_seed_luma_clip_high: float = 0.91  # remove brightest seed pixels (quantile)
    floor_k: int = 4  # number of clusters in seed band
    floor_expand_enabled: bool = True  # expand mask locally with relaxed threshold
    floor_expand_quantile: float = 0.95  # relaxed threshold (quantile)
    floor_expand_ksize: int = 11  # dilation kernel for expansion area
    floor_l_norm: str = "none"  # "none", "global", "seed"
    floor_seed_quantile: float = 0.87  # distance quantile for threshold
    floor_texture_window: int = 19  # texture window size for structure tensor
    floor_texture_blur: int = 6  # pre-blur before texture computation
    floor_w_l: float = 0.5  # weight for L channel
    floor_w_ab: float = 0.75  # weight for A/B channels
    floor_w_tex: float = 1  # weight for texture channel
    floor_clean_close_ksize: int = 11  # close kernel for floor mask cleanup
    floor_clean_open_ksize: int = 15  # open kernel for floor mask cleanup
    floor_keep_bottom_connected: bool = True  # keep only components touching bottom
    floor_min_seed_pixels: int = 500  # minimum seed pixels to proceed
    floor_mask_override_path: Optional[str] = None  # use a fixed mask instead of detection

    # --- Change detection (diff) ---
    enable_chroma_diff: bool = False  # detect color-only changes
    enable_edge_diff: bool = True  # detect edge/gradient changes
    enable_texture_diff: bool = True  # detect texture/gradient changes
    enable_local_contrast_diff: bool = False  # detect local contrast changes
    enable_shadow_mask: bool = False  # suppress shadow-like changes

    # Thresholds (you will tune these)
    chroma_diff_thresh: int = 25  # threshold for chroma diff mask
    edge_diff_thresh: int = 50  # threshold for edge diff mask
    texture_diff_thresh: int = 30  # threshold for texture diff mask
    local_contrast_diff_thresh: int = 20  # threshold for local contrast diff mask
    local_contrast_ksize: int = 9  # blur size for local contrast
    shadow_luma_thresh: int = 25  # luma change needed to mark shadow
    shadow_chroma_small_thresh: int = 40  # chroma change must stay below this

    # Combine masks: OR is generally safer than AND
    combine_mode: str = "OR"  # "OR" or "AND"

    # --- Morphology cleanup ---
    enable_morphology: bool = True  # apply close/open to combined mask
    morph_close_ksize: int = 17  # closing kernel size
    morph_open_ksize: int = 5  # opening kernel size
    morph_iterations: int = 1  # iterations for morphology ops
    enable_edge_fill: bool = True  # thicken edges then close
    edge_thicken_ksize: int = 3  # dilation size for edge fill
    edge_fill_close_ksize: int = 11  # closing size for edge fill

    # --- Connected components / contours filtering ---
    enable_area_filter: bool = True  # filter boxes by pixel area
    min_area: int = 400  # minimum area for a box
    max_area: int = 200000  # maximum area for a box

    # Bounding box filtering (optional heuristics)
    enable_bbox_filters: bool = True  # enable aspect/ROI checks
    min_aspect_ratio: float = 0.15  # width/height lower bound
    max_aspect_ratio: float = 6.0  # width/height upper bound
    must_be_in_floor_roi: bool = True  # bbox center must be in floor ROI
    merge_close_boxes: bool = True  # merge nearby detections
    merge_mode: str = "mask"  # "bbox" or "mask"
    merge_distance: int = 11  # max gap for bbox merge
    merge_mask_ksize: int = 11  # dilation size for mask merge

    # --- Debug ---
    show_debug_windows: bool = True  # display debug visualization
    debug_scale: float = 0.3  # scale for debug window
    debug_mode: str = "grid"  # "grid" or "windows"
    debug_grid_cols: int = 3  # columns in debug grid
    debug_grid_pad: int = 8  # padding between tiles
    debug_window_name: str = "Pipeline Debug"  # window title
    debug_label_font_scale: float = 0.5  # label size in grid
