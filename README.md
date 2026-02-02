# FloorShiftDetector

Computer vision project to detect new/moved objects on the floor by comparing a reference image with a current image.

## Installation
```bash
pip install -r requirements.txt
```

## Features
- Compare a reference image with a current image to detect new/moved objects on the floor.
- Floor ROI detection (automatic) or fixed mask mode.
- Multiple change cues (chroma, edges, texture, local contrast) with tunable thresholds.
- Optional cleanup (morphology), filtering, and box merging.
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

## Project structure
- `floorshiftdetector.py`: main pipeline
- `streamlit_app.py`: web UI
- `tui.py`: terminal UI
- `Images/`: sample images
- `experiments/`: experiments + tools
