from __future__ import annotations

from typing import List, Tuple, Optional
import cv2
import numpy as np

from .config import PipelineConfig


def find_bounding_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Extract external contour bounding boxes from a binary mask."""
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
    cfg: PipelineConfig,
) -> List[Tuple[int, int, int, int]]:
    """Filter boxes using area/aspect and optional floor ROI center check."""
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


def merge_boxes(boxes: List[Tuple[int, int, int, int]], max_gap: int) -> List[Tuple[int, int, int, int]]:
    """Merge nearby boxes based on max gap between their edges."""
    if not boxes:
        return boxes

    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        out: List[Tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            x, y, w, h = merged[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                xj, yj, wj, hj = merged[j]
                xj1, yj1, xj2, yj2 = xj, yj, xj + wj, yj + hj
                gap_x = max(0, max(x1 - xj2, xj1 - x2))
                gap_y = max(0, max(y1 - yj2, yj1 - y2))
                if gap_x <= max_gap and gap_y <= max_gap:
                    x1 = min(x1, xj1)
                    y1 = min(y1, yj1)
                    x2 = max(x2, xj2)
                    y2 = max(y2, yj2)
                    used[j] = True
                    changed = True
            out.append((x1, y1, x2 - x1, y2 - y1))
        merged = out
    return merged


def dilate_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Dilate a binary mask with an elliptical kernel."""
    if ksize <= 1:
        return mask
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, kernel, iterations=1)


def refine_boxes_to_mask(
    boxes: List[Tuple[int, int, int, int]],
    mask: np.ndarray
) -> List[Tuple[int, int, int, int]]:
    """Shrink boxes to the tight bounds of mask pixels inside each box."""
    refined: List[Tuple[int, int, int, int]] = []
    h, w = mask.shape[:2]
    for x, y, bw, bh in boxes:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        roi = mask[y1:y2, x1:x2]
        ys, xs = np.where(roi > 0)
        if ys.size == 0:
            continue
        rx1 = x1 + int(xs.min())
        ry1 = y1 + int(ys.min())
        rx2 = x1 + int(xs.max()) + 1
        ry2 = y1 + int(ys.max()) + 1
        refined.append((rx1, ry1, rx2 - rx1, ry2 - ry1))
    return refined


def draw_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Draw green rectangles on a copy of the input image."""
    out = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out
