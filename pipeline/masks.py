from __future__ import annotations

from typing import Optional
import cv2
import numpy as np


def ensure_odd(value: int) -> int:
    """Return an odd kernel size (increment if even)."""
    return value if value % 2 == 1 else value + 1


def resize_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize to target width while preserving aspect ratio."""
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / float(w)
    new_size = (target_width, int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def convert_color_space(bgr: np.ndarray, color_space: str) -> np.ndarray:
    """Convert BGR image to the requested color space."""
    if color_space.upper() == "LAB":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    if color_space.upper() == "YCRCB":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    if color_space.upper() == "HSV":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raise ValueError(f"Unsupported color_space: {color_space}")


def apply_clahe_to_luminance(img_cs: np.ndarray, color_space: str) -> np.ndarray:
    """Apply CLAHE on the luminance-like channel of the chosen color space."""
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
    """Extract luminance-like channel from LAB/YCrCb/HSV."""
    cs = color_space.upper()
    if cs in ["LAB", "YCRCB"]:
        return img_cs[:, :, 0]
    if cs == "HSV":
        return img_cs[:, :, 2]
    raise ValueError("Unsupported color space for luminance extraction")


def shadow_mask(ref_cs: np.ndarray, cur_cs: np.ndarray, color_space: str,
                t_luma: int = 25, t_chroma_small: int = 10) -> np.ndarray:
    """Detect shadow-like pixels (large luma change, small chroma change)."""
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
    """Chroma-only absolute difference mask for color-change detection."""
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
    """Texture/gradient magnitude difference mask (Sobel)."""
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
    """Edge magnitude map via Sobel."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag


def edge_diff_mask(ref_bgr: np.ndarray, cur_bgr: np.ndarray, thresh: int) -> np.ndarray:
    """Edge magnitude difference mask."""
    e_ref = edges_mask(ref_bgr)
    e_cur = edges_mask(cur_bgr)
    diff = cv2.absdiff(e_ref, e_cur)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return mask


def local_contrast_diff_mask(ref_bgr: np.ndarray, cur_bgr: np.ndarray, thresh: int, ksize: int) -> np.ndarray:
    """Local-contrast difference mask (gray minus blurred gray)."""
    if ksize <= 0:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    ref_blur = cv2.GaussianBlur(ref_gray, (ksize, ksize), 0)
    cur_blur = cv2.GaussianBlur(cur_gray, (ksize, ksize), 0)
    ref_lc = cv2.absdiff(ref_gray, ref_blur)
    cur_lc = cv2.absdiff(cur_gray, cur_blur)
    diff = cv2.absdiff(ref_lc, cur_lc)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return mask


def combine_masks(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray], mode: str) -> np.ndarray:
    """Combine two binary masks using OR/AND."""
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
    """Apply close then open to clean small gaps/noise."""
    out = mask.copy()

    if close_ksize > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=iterations)

    if open_ksize > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=iterations)

    return out


def apply_edge_fill(mask: np.ndarray, thicken_ksize: int, close_ksize: int) -> np.ndarray:
    """Thicken thin contours then close to fill interiors."""
    out = mask.copy()
    if thicken_ksize > 0:
        k_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thicken_ksize, thicken_ksize))
        out = cv2.dilate(out, k_thick, iterations=1)
    if close_ksize > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
    return out
