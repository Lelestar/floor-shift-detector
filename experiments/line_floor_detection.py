"""
Floor detection experiment using lines / horizon estimation.

Two modes:
1) baseboard: detect dominant horizontal line in lower image (HoughLinesP).
2) horizon: estimate vanishing point (intersection of non-horizontal lines), then
   take its y-coordinate as the horizon. Floor = below that line.

Run:
  python experiments/line_floor_detection.py \
    --image Images/Chambre/Reference.jpg \
    --outdir experiments/output_lines \
    --mode baseboard
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def line_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def detect_lines(
    edges: np.ndarray,
    min_line_length: int,
    max_line_gap: int,
    threshold: int,
) -> List[Tuple[int, int, int, int]]:
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    return [tuple(map(int, l[0])) for l in lines]


def select_baseboard_line(
    lines: List[Tuple[int, int, int, int]],
    angle_thresh_deg: float,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    best = None
    best_score = -1.0
    for x1, y1, x2, y2 in lines:
        angle = line_angle_deg(x1, y1, x2, y2)
        if abs(angle) > angle_thresh_deg:
            continue
        length = np.hypot(x2 - x1, y2 - y1)
        y_mean = 0.5 * (y1 + y2)
        # Prefer long lines that are lower in the image.
        score = length + 0.5 * (y_mean / max(1, img_h))
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)
    return best


def line_intersection(
    l1: Tuple[int, int, int, int],
    l2: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (float(px), float(py))


def estimate_horizon_y(
    lines: List[Tuple[int, int, int, int]],
    img_shape: Tuple[int, int],
    min_angle_deg: float,
    max_angle_deg: float,
    min_intersections: int,
) -> Optional[float]:
    h, w = img_shape
    filtered: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in lines:
        angle = abs(line_angle_deg(x1, y1, x2, y2))
        if min_angle_deg <= angle <= max_angle_deg:
            filtered.append((x1, y1, x2, y2))

    intersections: List[Tuple[float, float]] = []
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            angle_i = line_angle_deg(*filtered[i])
            angle_j = line_angle_deg(*filtered[j])
            if abs(angle_i - angle_j) < 15.0:
                continue
            pt = line_intersection(filtered[i], filtered[j])
            if pt is None:
                continue
            x, y = pt
            if -0.5 * w <= x <= 1.5 * w and -0.5 * h <= y <= 1.5 * h:
                intersections.append((x, y))

    if len(intersections) < min_intersections:
        return None

    ys = np.array([p[1] for p in intersections], dtype=np.float32)
    horizon_y = float(np.median(ys))
    return horizon_y


def draw_lines(bgr: np.ndarray, lines: List[Tuple[int, int, int, int]], color: Tuple[int, int, int]) -> np.ndarray:
    out = bgr.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(out, (x1, y1), (x2, y2), color, 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Floor detection via line cues.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--outdir", default="experiments/output_lines", help="Output directory.")
    parser.add_argument("--mode", choices=["baseboard", "horizon"], default="baseboard")
    parser.add_argument("--canny1", type=int, default=60, help="Canny low threshold.")
    parser.add_argument("--canny2", type=int, default=150, help="Canny high threshold.")
    parser.add_argument("--blur-ksize", type=int, default=5, help="Pre-blur size.")
    parser.add_argument("--hough-threshold", type=int, default=60, help="Hough accumulator threshold.")
    parser.add_argument("--min-line-length", type=int, default=60, help="Minimum line length.")
    parser.add_argument("--max-line-gap", type=int, default=10, help="Maximum line gap.")
    parser.add_argument("--angle-thresh", type=float, default=10.0, help="Baseboard angle threshold (deg).")
    parser.add_argument("--horizon-min-angle", type=float, default=20.0, help="Min angle for horizon lines (deg).")
    parser.add_argument("--horizon-max-angle", type=float, default=70.0, help="Max angle for horizon lines (deg).")
    parser.add_argument("--horizon-min-intersections", type=int, default=30, help="Min intersections to accept horizon.")
    parser.add_argument("--fallback-ratio", type=float, default=0.45, help="Fallback floor ratio if no line.")
    args = parser.parse_args()

    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    blur_ksize = ensure_odd(args.blur_ksize)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray, args.canny1, args.canny2)
    lines = detect_lines(edges, args.min_line_length, args.max_line_gap, args.hough_threshold)

    h, w = bgr.shape[:2]
    floor_mask = np.zeros((h, w), dtype=np.uint8)
    debug = draw_lines(bgr, lines, (0, 0, 255))

    horizon_y = None
    baseboard_line = None

    if args.mode == "baseboard":
        baseboard_line = select_baseboard_line(lines, args.angle_thresh, h)
        if baseboard_line is not None:
            x1, y1, x2, y2 = baseboard_line
            horizon_y = int(round(0.5 * (y1 + y2)))
    else:
        horizon_y = estimate_horizon_y(
            lines,
            img_shape=(h, w),
            min_angle_deg=args.horizon_min_angle,
            max_angle_deg=args.horizon_max_angle,
            min_intersections=args.horizon_min_intersections,
        )
        if horizon_y is not None:
            horizon_y = int(round(horizon_y))

    if horizon_y is None:
        horizon_y = int(round(h * (1.0 - args.fallback_ratio)))

    horizon_y = int(np.clip(horizon_y, 0, h - 1))
    floor_mask[horizon_y:h, :] = 255

    overlay = bgr.copy()
    overlay = np.where(
        floor_mask[:, :, None] > 0,
        (0.65 * overlay + 0.35 * np.array([0, 200, 0], dtype=np.float32)).astype(np.uint8),
        overlay,
    )

    debug2 = debug.copy()
    if baseboard_line is not None:
        x1, y1, x2, y2 = baseboard_line
        cv2.line(debug2, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.line(debug2, (0, horizon_y), (w - 1, horizon_y), (255, 0, 0), 2)

    cv2.imwrite(str(outdir / "input.png"), bgr)
    cv2.imwrite(str(outdir / "edges.png"), edges)
    cv2.imwrite(str(outdir / "lines.png"), debug)
    cv2.imwrite(str(outdir / "horizon.png"), debug2)
    cv2.imwrite(str(outdir / "floor_mask.png"), floor_mask)
    cv2.imwrite(str(outdir / "overlay.png"), overlay)

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
