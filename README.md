# FloorShiftDetector

Computer vision project to detect new/moved objects on the floor by comparing a reference image with a current image.

## Installation
```bash
pip install -r requirements.txt
```

## Features
- Compare a reference image with a current image to detect new/moved objects on the floor.
- Floor ROI detection (automatic) or fixed mask mode (masks/<Scene>.png).
- Multiple change cues (chroma, edges, texture, local contrast) with tunable thresholds.
- Optional cleanup (edge fill + morphology), filtering, and box merging.
- Streamlit UI for interactive tuning + visualization.
- CLI (script) and TUI for quick runs.

## Usage (CLI)
Run with explicit image paths:
```bash
python floorshiftdetector.py --ref Images/Bedroom/Reference.JPG --cur Images/Bedroom/IMG_6567.JPG
```

Common flags:
- `--debug` show debug grid (instead of result window)
- `--save-result PATH` save output image
- `--floor-roi-mode detection|mask` choose floor ROI mode

Notes:
- In `mask` mode, the CLI auto-uses `masks/<Scene>.png` where `<Scene>` is the parent folder of the reference image.

## Usage (TUI)
```bash
python tui.py
```

## Usage (Streamlit UI)
```bash
streamlit run streamlit_app.py
```
The UI lets you:
- upload images or choose local files,
- tune parameters,
- visualize all pipeline steps,
- export the result.


## Experiments
Randomized search for floor ROI parameters using ground-truth masks:
```bash
python experiments/random_search_floor_roi.py --images-dir /path/to/images --iters 200
```
Command line arguments:
- `images-dir PATH` directory with scene subfolders containing reference/current images
- `--iters N` number of random parameter sets to try (default: 200)
- `--seed S` random seed (default: 123)
- `--report-dir PATH` output directory (default: `experiments/best_report/`)
- `--out PATH` save best parameters to file (default: `experiments/best_params.txt`)
- `--eval-current` evaluate current configuration instead of random search

Outputs:
- `experiments/best_params.txt` with the best parameters + per-scene IoU.
- `experiments/best_report/` with overlays per scene.

Mask creation:
- `experiments/mask_creator.html` provides a simple in-browser mask painter.
- Masks should be saved as `masks/<Scene>.png` (white = floor).

## Project structure
- `pipeline/`: core pipeline (config, masks, ROI, boxes, run)
- `floorshiftdetector.py`: CLI entry point (uses `pipeline/`)
- `streamlit_app.py`: web UI
- `tui.py`: terminal UI
- `Images/`: sample images
- `masks/`: floor masks per scene (`<Scene>.png`)
- `experiments/`: experiments + tools (random search, mask creator)
