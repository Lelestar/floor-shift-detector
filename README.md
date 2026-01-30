# FloorShiftDetector

Computer vision project to detect new/moved objects on the floor by comparing a reference image with a current image.

## Installation
```bash
pip install -r requirements.txt
```

## Usage (pipeline)
1. Open `floorshiftdetector.py` and adjust image paths in `main()`.
2. Run:
```bash
python floorshiftdetector.py
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
- `Images/`: sample images
- `Sujet - TP1.pdf`: assignment statement
