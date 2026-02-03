from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np

from .config import PipelineConfig
from .masks import apply_morphology, ensure_odd


def build_floor_roi_mask(shape_hw: Tuple[int, int], floor_ratio: float) -> np.ndarray:
    """Create a bottom-band ROI mask based on a ratio of image height."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * (1.0 - floor_ratio))
    mask[y_start:h, :] = 255
    return mask


def keep_bottom_connected(mask: np.ndarray) -> np.ndarray:
    """Keep only connected components that touch the bottom row."""
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return mask
    bottom_labels = np.unique(labels[-1, :])
    bottom_labels = bottom_labels[bottom_labels != 0]
    if bottom_labels.size == 0:
        return mask
    keep = np.isin(labels, bottom_labels).astype(np.uint8) * 255
    return keep


def compute_texture_map(bgr: np.ndarray, window_ksize: int, blur_ksize: int) -> np.ndarray:
    """Compute a simple texture energy map using local structure tensor eigenvalues."""
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


def build_multicluster_features(
    bgr: np.ndarray,
    texture: np.ndarray,
    w_l: float,
    w_ab: float,
    w_tex: float,
    l_norm: str,
    l_stats: Tuple[float, float] | None,
) -> np.ndarray:
    """Build per-pixel feature vectors from LAB and texture channels."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    if l_norm != "none":
        if l_stats is None:
            raise ValueError("L stats required for L normalization")
        mean, std = l_stats
        if std > 1e-6:
            l = (l - mean) / std
    lab[:, :, 0] = np.clip(l, -3.0, 3.0)
    lab[:, :, 1] = (lab[:, :, 1] - 128.0) / 128.0
    lab[:, :, 2] = (lab[:, :, 2] - 128.0) / 128.0

    l = lab[:, :, 0:1] * w_l
    ab = lab[:, :, 1:3] * w_ab
    t = texture[:, :, None] * w_tex

    feat = np.concatenate([l, ab, t], axis=2)
    return feat


def normalize_features(feat: np.ndarray, seed_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features using mean/std from seed pixels."""
    seed = feat[seed_mask > 0].reshape(-1, feat.shape[2])
    mean = seed.mean(axis=0, keepdims=True)
    std = seed.std(axis=0, keepdims=True) + 1e-6
    norm = (feat - mean) / std
    return norm, mean, std


def run_kmeans(samples: np.ndarray, k: int) -> np.ndarray:
    """Fit k-means centers to samples and return the centers."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    flags = cv2.KMEANS_PP_CENTERS
    _, _, centers = cv2.kmeans(samples.astype(np.float32), k, None, criteria, 5, flags)
    return centers


def floor_mask_multicluster(ref_bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Compute a floor ROI mask using a multicluster texture/color seed."""
    h, w = ref_bgr.shape[:2]
    seed_h = max(1, int(h * cfg.floor_seed_ratio))
    seed_w = max(1, int(w * np.clip(cfg.floor_seed_x_ratio, 0.1, 1.0)))
    x0 = max(0, (w - seed_w) // 2)
    x1 = min(w, x0 + seed_w)
    seed_mask = np.zeros((h, w), dtype=np.uint8)
    seed_mask[h - seed_h:h, x0:x1] = 255

    seed_luma_mask = seed_mask.copy()
    low_q = float(np.clip(cfg.floor_seed_luma_clip_low, 0.0, 0.49))
    high_q = float(np.clip(cfg.floor_seed_luma_clip_high, 0.51, 1.0))
    if low_q > 0.0 or high_q < 1.0:
        l = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        seed_vals = l[seed_mask > 0]
        if seed_vals.size > 0:
            low_t = np.quantile(seed_vals, low_q) if low_q > 0.0 else None
            high_t = np.quantile(seed_vals, high_q) if high_q < 1.0 else None
            if low_t is not None:
                seed_luma_mask[l < low_t] = 0
            if high_t is not None:
                seed_luma_mask[l > high_t] = 0

    if np.count_nonzero(seed_luma_mask) < cfg.floor_min_seed_pixels:
        return build_floor_roi_mask((h, w), cfg.floor_roi_ratio)

    window_ksize = ensure_odd(int(cfg.floor_texture_window))
    blur_ksize = int(cfg.floor_texture_blur)
    if blur_ksize > 0:
        blur_ksize = ensure_odd(blur_ksize)
    texture = compute_texture_map(ref_bgr, window_ksize=window_ksize, blur_ksize=blur_ksize)

    l_norm = cfg.floor_l_norm.lower()
    if l_norm not in {"none", "global", "seed"}:
        raise ValueError(f"Unsupported floor_l_norm: {cfg.floor_l_norm}")

    l_stats = None
    if l_norm in {"global", "seed"}:
        l = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
        l_vals = l.reshape(-1) if l_norm == "global" else l[seed_luma_mask > 0]
        if l_vals.size > 0:
            l_stats = (float(l_vals.mean()), float(l_vals.std() + 1e-6))

    feat = build_multicluster_features(
        ref_bgr,
        texture,
        w_l=cfg.floor_w_l,
        w_ab=cfg.floor_w_ab,
        w_tex=cfg.floor_w_tex,
        l_norm=l_norm,
        l_stats=l_stats,
    )
    feat_norm, _, _ = normalize_features(feat, seed_luma_mask)

    seed_samples = feat_norm[seed_luma_mask > 0].reshape(-1, feat.shape[2])
    if seed_samples.shape[0] < cfg.floor_min_seed_pixels:
        return build_floor_roi_mask((h, w), cfg.floor_roi_ratio)

    k = max(1, int(cfg.floor_k))
    centers = run_kmeans(seed_samples, k)

    all_samples = feat_norm.reshape(-1, feat.shape[2])
    diff = all_samples[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_dist = dists.min(axis=1)

    seed_dists = min_dist[seed_luma_mask.reshape(-1) > 0]
    quant = float(np.clip(cfg.floor_seed_quantile, 0.5, 0.99))
    threshold = float(np.quantile(seed_dists, quant))
    candidate = (min_dist <= threshold).astype(np.uint8).reshape(h, w) * 255

    if cfg.floor_expand_enabled:
        expand_quant = float(np.clip(cfg.floor_expand_quantile, quant, 0.995))
        expand_threshold = float(np.quantile(seed_dists, expand_quant))
        expand_ksize = ensure_odd(int(cfg.floor_expand_ksize))
        if expand_ksize > 1:
            k_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_ksize, expand_ksize))
            expand_area = cv2.dilate(candidate, k_expand, iterations=1)
            relaxed = (min_dist <= expand_threshold).astype(np.uint8).reshape(h, w) * 255
            candidate = cv2.bitwise_and(relaxed, expand_area)

    if cfg.floor_clean_close_ksize > 0 or cfg.floor_clean_open_ksize > 0:
        candidate = apply_morphology(
            candidate,
            close_ksize=cfg.floor_clean_close_ksize,
            open_ksize=cfg.floor_clean_open_ksize,
            iterations=1,
        )

    if cfg.floor_keep_bottom_connected:
        candidate = keep_bottom_connected(candidate)

    if np.count_nonzero(candidate) == 0:
        return build_floor_roi_mask((h, w), cfg.floor_roi_ratio)

    return candidate
