"""
FloorShiftDetector - baseline pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2

from pipeline import PipelineConfig, run_pipeline


def infer_mask_path(ref_path: str) -> Optional[str]:
    """Infer a mask path based on the reference image parent folder name."""
    if not ref_path:
        return None
    scene = Path(ref_path).parent.name
    if not scene:
        return None
    mask_path = Path("masks") / f"{scene}.png"
    return str(mask_path) if mask_path.exists() else None


def main() -> None:
    """CLI entry point for running the pipeline on two images."""
    parser = argparse.ArgumentParser(description="FloorShiftDetector CLI")
    parser.add_argument("--ref", default="Images/Bedroom/Reference.JPG", help="Reference image path")
    parser.add_argument("--cur", default="Images/Bedroom/IMG_6567.JPG", help="Current image path")
    parser.add_argument("--floor-roi-mode", choices=["detection", "mask"], default="detection", help="Floor ROI mode")
    parser.add_argument("--debug", action="store_true", help="Show debug windows")
    parser.add_argument("--save-result", default="", help="Path to save result image")
    args = parser.parse_args()

    cfg = PipelineConfig(show_debug_windows=args.debug, debug_mode="grid")
    cfg.floor_roi_mode = args.floor_roi_mode
    if cfg.enable_floor_roi and cfg.floor_roi_mode == "mask":
        mask_path = infer_mask_path(args.ref)
        if not mask_path:
            raise FileNotFoundError("No mask found for this reference image in masks/ (expected masks/<Scene>.png).")
        cfg.floor_mask_override_path = mask_path

    outputs = run_pipeline(args.ref, args.cur, cfg)

    if args.save_result:
        cv2.imwrite(args.save_result, outputs["result"])

    if not args.debug:
        cv2.imshow("Result", outputs["result"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


__all__ = ["PipelineConfig", "run_pipeline", "infer_mask_path"]

if __name__ == "__main__":
    main()
