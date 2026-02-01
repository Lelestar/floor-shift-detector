"""
Simple mask creation tool (paint/erase) for reference images.

Usage:
  python experiments/mask_creator.py --image Images/Bedroom/Reference.JPG --out masks/Bedroom.png

Controls:
  - Left mouse: paint (white)
  - Right mouse: erase (black)
  - Mouse wheel: change brush size
  - [ / ] : decrease / increase brush size
  - c : clear mask
  - s : save mask
  - q : quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask creator (paint/erase).")
    parser.add_argument("--image", required=True, help="Path to reference image.")
    parser.add_argument("--out", required=True, help="Output mask path (png).")
    parser.add_argument("--brush", type=int, default=20, help="Initial brush size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    brush = max(1, int(args.brush))
    drawing = {"left": False, "right": False}

    win = "Mask Creator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal brush
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["left"] = True
            cv2.circle(mask, (x, y), brush, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["left"] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing["right"] = True
            cv2.circle(mask, (x, y), brush, 0, -1)
        elif event == cv2.EVENT_RBUTTONUP:
            drawing["right"] = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing["left"]:
                cv2.circle(mask, (x, y), brush, 255, -1)
            if drawing["right"]:
                cv2.circle(mask, (x, y), brush, 0, -1)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        overlay = img.copy()
        color = np.zeros_like(overlay)
        color[:] = (0, 200, 0)
        overlay = np.where(mask[:, :, None] > 0, (0.65 * overlay + 0.35 * color).astype(np.uint8), overlay)

        display = overlay.copy()
        cv2.putText(
            display,
            f"brush: {brush} | left=paint right=erase | +/- size | s=save c=clear q=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            mask[:] = 0
        if key == ord("s"):
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), mask)
            print(f"Saved mask: {out_path}")
        if key in (ord("-"), ord("_"), ord("[")):
            brush = max(1, brush - 2)
        if key in (ord("+"), ord("="), ord("]")):
            brush = min(200, brush + 2)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
