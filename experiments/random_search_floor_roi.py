"""
Randomized search to optimize floor ROI parameters against ground-truth masks.

Assumes:
  - Reference images: Images/<Scene>/Reference.JPG
  - Masks: masks/<Scene>.png (white = floor)

Run:
  python experiments/random_search_floor_roi.py --iters 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import PipelineConfig
from pipeline.roi import floor_mask_multicluster


SCENES = ["Bedroom", "Kitchen", "LivingRoom"]


def load_pair(scene: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a reference image and its binary floor mask for one scene."""
    img_path = Path("Images") / scene / "Reference.JPG"
    mask_path = Path("masks") / f"{scene}.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Missing image: {img_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Missing mask: {mask_path}")
    mask = (mask > 127).astype(np.uint8)
    return img, mask


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    pred_bin = pred > 0
    target_bin = target > 0
    inter = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def sample_params(rng: np.random.Generator) -> Dict[str, object]:
    """Sample a random configuration for floor ROI parameters."""
    return {
        "floor_seed_ratio": rng.uniform(0.15, 0.4),
        "floor_seed_x_ratio": rng.uniform(0.5, 1.0),
        "floor_seed_luma_clip_low": rng.uniform(0.0, 0.1),
        "floor_seed_luma_clip_high": rng.uniform(0.9, 1.0),
        "floor_k": int(rng.integers(2, 7)),
        "floor_seed_quantile": rng.uniform(0.85, 0.98),
        "floor_w_l": rng.uniform(0.2, 0.9),
        "floor_w_ab": rng.uniform(0.4, 1.2),
        "floor_w_tex": rng.uniform(0.4, 1.3),
        "floor_texture_window": int(rng.integers(7, 19)) | 1,
        "floor_texture_blur": int(rng.integers(0, 9)),
        "floor_expand_enabled": True,
        "floor_expand_quantile": rng.uniform(0.95, 0.99),
        "floor_expand_ksize": int(rng.integers(7, 21)) | 1,
        "floor_l_norm": rng.choice(["none", "global"]),
    }


def build_cfg(params: Dict[str, object]) -> PipelineConfig:
    """Build a PipelineConfig from a parameter dict."""
    cfg = PipelineConfig(show_debug_windows=False)
    for k, v in params.items():
        setattr(cfg, k, v)
    return cfg


def evaluate(cfg: PipelineConfig, data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """Evaluate mean IoU of a config across all scenes."""
    scores = []
    for img, gt in data:
        pred = floor_mask_multicluster(img, cfg)
        if pred.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        scores.append(iou(pred, gt))
    return float(np.mean(scores))


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Overlay a binary mask on top of the image with a fixed color."""
    out = bgr.copy()
    color_img = np.zeros_like(out)
    color_img[:] = color
    alpha = 0.35
    out = np.where(mask[:, :, None] > 0, (1 - alpha) * out + alpha * color_img, out)
    return out.astype(np.uint8)


def save_report(cfg: PipelineConfig, data: List[Tuple[np.ndarray, np.ndarray]], out_dir: Path) -> Dict[str, float]:
    """Save per-scene overlays and return per-scene IoU scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_scene: Dict[str, float] = {}
    for scene, (img, gt) in zip(SCENES, data):
        pred = floor_mask_multicluster(img, cfg)
        if pred.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        score = iou(pred, gt)
        per_scene[scene] = score

        overlay_pred = overlay_mask(img, pred, (0, 200, 0))
        overlay_gt = overlay_mask(img, (gt > 0).astype(np.uint8) * 255, (0, 0, 255))
        side = np.hstack([overlay_pred, overlay_gt])

        cv2.imwrite(str(out_dir / f"{scene}_overlay_pred.png"), overlay_pred)
        cv2.imwrite(str(out_dir / f"{scene}_overlay_gt.png"), overlay_gt)
        cv2.imwrite(str(out_dir / f"{scene}_side_by_side.png"), side)

    return per_scene


def main() -> None:
    """CLI entry point for randomized floor ROI search."""
    parser = argparse.ArgumentParser(description="Random search for floor ROI params.")
    parser.add_argument("--iters", type=int, default=200, help="Number of random trials.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--out", default="experiments/best_params.txt", help="Output file.")
    parser.add_argument("--report-dir", default="experiments/best_report", help="Report output directory.")
    parser.add_argument(
        "--eval-current",
        action="store_true",
        help="Evaluate current PipelineConfig defaults and print per-scene IoU.",
    )
    args = parser.parse_args()

    cv2.setRNGSeed(args.seed)
    data = [load_pair(scene) for scene in SCENES]

    if args.eval_current:
        cfg = PipelineConfig(show_debug_windows=False)
        per_scene = {}
        for scene, (img, gt) in zip(SCENES, data):
            pred = floor_mask_multicluster(img, cfg)
            if pred.shape[:2] != gt.shape[:2]:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            per_scene[scene] = iou(pred, gt)
        mean_iou = float(np.mean(list(per_scene.values()))) if per_scene else 0.0
        print(f"mean_iou: {mean_iou:.4f}")
        for scene, score in per_scene.items():
            print(f"{scene}_iou: {score:.4f}")
        return

    rng = np.random.default_rng(args.seed)
    best_score = -1.0
    best_params = None

    for i in range(args.iters):
        params = sample_params(rng)
        cfg = build_cfg(params)
        score = evaluate(cfg, data)
        if score > best_score:
            best_score = score
            best_params = params
            print(f"[{i+1}/{args.iters}] new best: {best_score:.4f}")
        elif (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{args.iters}] score: {score:.4f} | best: {best_score:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    per_scene = save_report(build_cfg(best_params), data, report_dir)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"best_score_mean_iou: {best_score:.4f}\n")
        for scene, score in per_scene.items():
            f.write(f"{scene}_iou: {score:.4f}\n")
        f.write("\n")
        for k in sorted(best_params.keys()):
            f.write(f"{k}: {best_params[k]}\n")

    print(f"Saved best params to: {out_path}")
    print(f"Saved report to: {report_dir}")


if __name__ == "__main__":
    main()
