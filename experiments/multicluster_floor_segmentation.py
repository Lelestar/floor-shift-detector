"""
Floor segmentation experiment using multi-cluster appearance + bottom connectivity.

Idea:
- Extract features (LAB color + texture).
- Learn K clusters from a bottom seed band (likely floor).
- Compute distance to nearest cluster for all pixels.
- Threshold by a seed-distance quantile to keep only "floor-like" pixels.
- Keep only components connected to the bottom.

Run:
  python experiments/multicluster_floor_segmentation.py \
    --image Images/Bedroom/Reference.JPG \
    --outdir experiments/output_multicluster
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


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

    lambda1 = 0.5 * (trace + delta)
    lambda2 = 0.5 * (trace - delta)
    texture = lambda1 + lambda2
    texture = cv2.normalize(texture, None, 0, 1.0, cv2.NORM_MINMAX)
    return texture.astype(np.float32)


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


def build_features(
    bgr: np.ndarray,
    texture: np.ndarray,
    w_l: float,
    w_ab: float,
    w_tex: float,
) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] /= 255.0
    lab[:, :, 1] = (lab[:, :, 1] - 128.0) / 128.0
    lab[:, :, 2] = (lab[:, :, 2] - 128.0) / 128.0

    l = lab[:, :, 0:1] * w_l
    ab = lab[:, :, 1:3] * w_ab
    t = texture[:, :, None] * w_tex

    feat = np.concatenate([l, ab, t], axis=2)
    return feat


def normalize_features(feat: np.ndarray, seed_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = feat[seed_mask > 0].reshape(-1, feat.shape[2])
    mean = seed.mean(axis=0, keepdims=True)
    std = seed.std(axis=0, keepdims=True) + 1e-6
    norm = (feat - mean) / std
    return norm, mean, std


def run_kmeans(samples: np.ndarray, k: int) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    flags = cv2.KMEANS_PP_CENTERS
    _, _, centers = cv2.kmeans(samples.astype(np.float32), k, None, criteria, 5, flags)
    return centers


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-cluster floor segmentation experiment.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--outdir", default="experiments/output_multicluster", help="Output directory.")
    parser.add_argument("--seed-ratio", type=float, default=0.25, help="Bottom band used as seed.")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters in seed band.")
    parser.add_argument("--seed-quantile", type=float, default=0.9, help="Distance quantile for threshold.")
    parser.add_argument("--texture-window", type=int, default=9, help="Texture map window size.")
    parser.add_argument("--texture-blur", type=int, default=5, help="Texture map pre-blur size.")
    parser.add_argument("--w-l", type=float, default=0.3, help="Weight of L channel.")
    parser.add_argument("--w-ab", type=float, default=1.0, help="Weight of a,b channels.")
    parser.add_argument("--w-tex", type=float, default=0.8, help="Weight of texture.")
    parser.add_argument("--close-ksize", type=int, default=7, help="Morph close kernel.")
    parser.add_argument("--open-ksize", type=int, default=5, help="Morph open kernel.")
    args = parser.parse_args()

    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = bgr.shape[:2]
    seed_h = max(1, int(h * args.seed_ratio))
    seed_mask = np.zeros((h, w), dtype=np.uint8)
    seed_mask[h - seed_h : h, :] = 255

    texture = compute_texture_map(bgr, ensure_odd(args.texture_window), ensure_odd(args.texture_blur))
    feat = build_features(bgr, texture, args.w_l, args.w_ab, args.w_tex)
    feat_norm, mean, std = normalize_features(feat, seed_mask)

    seed_samples = feat_norm[seed_mask > 0].reshape(-1, feat.shape[2])
    centers = run_kmeans(seed_samples, args.k)

    all_samples = feat_norm.reshape(-1, feat.shape[2])
    diff = all_samples[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_dist = dists.min(axis=1)

    seed_dists = min_dist[seed_mask.reshape(-1) > 0]
    threshold = float(np.quantile(seed_dists, args.seed_quantile))
    candidate = (min_dist <= threshold).astype(np.uint8).reshape(h, w) * 255

    if args.close_ksize > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ensure_odd(args.close_ksize),) * 2)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_close, iterations=1)
    if args.open_ksize > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ensure_odd(args.open_ksize),) * 2)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k_open, iterations=1)

    floor_mask = keep_bottom_connected(candidate)

    overlay = bgr.copy()
    overlay = np.where(
        floor_mask[:, :, None] > 0,
        (0.65 * overlay + 0.35 * np.array([0, 200, 0], dtype=np.float32)).astype(np.uint8),
        overlay,
    )

    dist_map = min_dist.reshape(h, w)
    dist_map = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(str(outdir / "input.png"), bgr)
    cv2.imwrite(str(outdir / "texture.png"), (texture * 255).astype(np.uint8))
    cv2.imwrite(str(outdir / "distance.png"), dist_map)
    cv2.imwrite(str(outdir / "candidate.png"), candidate)
    cv2.imwrite(str(outdir / "floor_mask.png"), floor_mask)
    cv2.imwrite(str(outdir / "overlay.png"), overlay)

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
