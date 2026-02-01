"""
Texture-based floor segmentation experiment.

Idea:
- Compute a texture "roughness" map using a structure tensor (local gradient covariance).
- Segment smooth vs textured areas via Otsu thresholding.
- Assume floor is the largest smooth (or rough) region, optionally connected to bottom.

Run:
  python experiments/texture_floor_segmentation.py \
    --image Images/Chambre/Reference.jpg \
    --outdir experiments/output_texture
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def compute_texture_map(bgr: np.ndarray, window_ksize: int, blur_ksize: int) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy

    sxx = cv2.boxFilter(gxx, -1, (window_ksize, window_ksize), normalize=False)
    syy = cv2.boxFilter(gyy, -1, (window_ksize, window_ksize), normalize=False)
    sxy = cv2.boxFilter(gxy, -1, (window_ksize, window_ksize), normalize=False)

    trace = sxx + syy
    diff = sxx - syy
    delta = np.sqrt((diff * diff) + (4.0 * sxy * sxy))

    # Eigenvalues of 2x2 structure tensor
    lambda1 = 0.5 * (trace + delta)
    lambda2 = 0.5 * (trace - delta)

    # Texture energy (larger = more textured/rough)
    texture = lambda1 + lambda2
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
    return texture.astype(np.uint8)


def keep_bottom_connected(mask: np.ndarray) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return mask
    bottom_labels = np.unique(labels[-1, :])
    bottom_labels = bottom_labels[bottom_labels != 0]
    if bottom_labels.size == 0:
        return mask
    keep = np.isin(labels, bottom_labels).astype(np.uint8) * 255
    return keep


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = (labels == idx).astype(np.uint8) * 255
    return out


def segment_floor_by_texture(
    texture: np.ndarray,
    smooth_floor: bool,
    close_ksize: int,
    open_ksize: int,
    keep_bottom: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    _, rough_mask = cv2.threshold(texture, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smooth_mask = cv2.bitwise_not(rough_mask)

    candidate = smooth_mask if smooth_floor else rough_mask

    if close_ksize > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_close, iterations=1)
    if open_ksize > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k_open, iterations=1)

    if keep_bottom:
        floor_mask = keep_bottom_connected(candidate)
    else:
        floor_mask = keep_largest_component(candidate)

    return floor_mask, rough_mask


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = bgr.copy()
    color_img = np.zeros_like(out)
    color_img[:] = color
    alpha = 0.35
    out = np.where(mask[:, :, None] > 0, (1 - alpha) * out + alpha * color_img, out)
    return out.astype(np.uint8)


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Texture-based floor segmentation experiment.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--outdir", default="experiments/output_texture", help="Output directory.")
    parser.add_argument("--window-ksize", type=int, default=9, help="Window size for texture map.")
    parser.add_argument("--blur-ksize", type=int, default=5, help="Blur size before gradients.")
    parser.add_argument(
        "--smooth-floor",
        action="store_true",
        help="Assume floor is the smooth (low texture) region.",
    )
    parser.add_argument(
        "--rough-floor",
        action="store_true",
        help="Assume floor is the textured (high texture) region.",
    )
    parser.add_argument("--close-ksize", type=int, default=7, help="Morph close kernel.")
    parser.add_argument("--open-ksize", type=int, default=5, help="Morph open kernel.")
    parser.add_argument(
        "--keep-bottom",
        action="store_true",
        help="Keep only components touching the bottom row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    window_ksize = ensure_odd(args.window_ksize)
    blur_ksize = ensure_odd(args.blur_ksize) if args.blur_ksize > 0 else 0

    texture = compute_texture_map(bgr, window_ksize, blur_ksize)

    smooth_floor = True
    if args.rough_floor:
        smooth_floor = False
    elif args.smooth_floor:
        smooth_floor = True

    close_ksize = ensure_odd(args.close_ksize)
    open_ksize = ensure_odd(args.open_ksize)

    floor_mask, rough_mask = segment_floor_by_texture(
        texture,
        smooth_floor=smooth_floor,
        close_ksize=close_ksize,
        open_ksize=open_ksize,
        keep_bottom=args.keep_bottom,
    )

    overlay = overlay_mask(bgr, floor_mask, (0, 200, 0))

    cv2.imwrite(str(outdir / "input.png"), bgr)
    cv2.imwrite(str(outdir / "texture_map.png"), texture)
    cv2.imwrite(str(outdir / "rough_mask.png"), rough_mask)
    cv2.imwrite(str(outdir / "floor_mask.png"), floor_mask)
    cv2.imwrite(str(outdir / "overlay.png"), overlay)

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
