"""
Streamlit UI for FloorShiftDetector.

Run:
  streamlit run streamlit_app.py
"""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import streamlit as st

from floorshiftdetector import PipelineConfig, run_pipeline_images


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_image_from_path(path_str: str) -> Optional[np.ndarray]:
    if not path_str:
        return None
    return cv2.imread(path_str, cv2.IMREAD_COLOR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_display(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return bgr_to_rgb(img)
    return img[:, :, :3]


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def collect_image_paths(root: Path) -> List[str]:
    if not root.exists():
        return []
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(str(p) for p in paths)


def filter_reference_paths(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        name = Path(p).stem.lower()
        if name == "reference" or name.endswith("_reference"):
            out.append(p)
    return out


def filter_current_paths(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        name = Path(p).stem.lower()
        if name == "reference" or name.endswith("_reference"):
            continue
        out.append(p)
    return out


def build_config() -> PipelineConfig:
    cfg = st.session_state.get("cfg")
    if cfg is None:
        cfg = PipelineConfig(show_debug_windows=False)
        st.session_state["cfg"] = cfg
    else:
        # Handle older cached configs when PipelineConfig gains new fields.
        missing = [
            "enable_texture_diff",
            "texture_diff_thresh",
            "enable_shadow_mask",
            "shadow_luma_thresh",
            "shadow_chroma_small_thresh",
        ]
        if any(not hasattr(cfg, name) for name in missing):
            cfg = PipelineConfig(show_debug_windows=False)
            st.session_state["cfg"] = cfg

    with st.sidebar.form("params_form"):
        st.markdown("### Pipeline settings")
        enable_resize = st.checkbox("Resize images", value=cfg.enable_resize)
        target_width = st.slider("Target width", 320, 1920, cfg.target_width, step=40)

        color_space = st.selectbox(
            "Color space",
            ["LAB", "YCrCb", "HSV"],
            index=["LAB", "YCrCb", "HSV"].index(cfg.color_space),
        )
        enable_clahe = st.checkbox("Apply CLAHE (luminance)", value=cfg.enable_clahe_on_luminance)

        st.markdown("### Floor ROI")
        enable_floor_roi = st.checkbox("Enable floor ROI", value=cfg.enable_floor_roi)
        floor_ratio = st.slider("Floor ROI ratio", 0.1, 0.9, float(cfg.floor_roi_ratio), step=0.05)

        st.markdown("### Change detection")
        enable_chroma_diff = st.checkbox("Chroma diff", value=cfg.enable_chroma_diff)
        chroma_diff_thresh = st.slider("Chroma diff threshold", 0, 80, cfg.chroma_diff_thresh, step=1)
        enable_edge_diff = st.checkbox("Edge diff", value=cfg.enable_edge_diff)
        edge_diff_thresh = st.slider("Edge diff threshold", 0, 120, cfg.edge_diff_thresh, step=1)
        enable_texture_diff = st.checkbox("Texture diff", value=cfg.enable_texture_diff)
        texture_diff_thresh = st.slider("Texture diff threshold", 0, 80, cfg.texture_diff_thresh, step=1)
        enable_shadow_mask = st.checkbox("Shadow mask", value=cfg.enable_shadow_mask)
        shadow_luma_thresh = st.slider("Shadow luma threshold", 0, 80, cfg.shadow_luma_thresh, step=1)
        shadow_chroma_small_thresh = st.slider(
            "Shadow chroma small threshold",
            0,
            40,
            cfg.shadow_chroma_small_thresh,
            step=1
        )
        combine_mode = st.selectbox("Combine mode", ["OR", "AND"], index=0 if cfg.combine_mode == "OR" else 1)

        st.markdown("### Morphology")
        enable_morphology = st.checkbox("Enable morphology", value=cfg.enable_morphology)
        morph_close_ksize = st.slider("Close kernel size", 1, 25, cfg.morph_close_ksize, step=2)
        morph_open_ksize = st.slider("Open kernel size", 1, 25, cfg.morph_open_ksize, step=2)
        morph_iterations = st.slider("Morph iterations", 1, 4, cfg.morph_iterations, step=1)

        st.markdown("### Filtering")
        enable_area_filter = st.checkbox("Enable area filter", value=cfg.enable_area_filter)
        min_area = st.slider("Min area", 0, 2000, cfg.min_area, step=50)
        max_area = st.slider("Max area", 1000, 300000, cfg.max_area, step=1000)

        enable_bbox_filters = st.checkbox("Enable bbox filters", value=cfg.enable_bbox_filters)
        min_aspect_ratio = st.slider("Min aspect ratio", 0.05, 1.5, float(cfg.min_aspect_ratio), step=0.05)
        max_aspect_ratio = st.slider("Max aspect ratio", 1.5, 10.0, float(cfg.max_aspect_ratio), step=0.1)
        must_be_in_floor_roi = st.checkbox("BBox center must be in floor ROI", value=cfg.must_be_in_floor_roi)

        st.markdown("### Display")
        debug_scale = st.slider("Display scale", 0.3, 1.0, float(cfg.debug_scale), step=0.05)

        submitted = st.form_submit_button("Apply settings")

    if submitted:
        cfg = PipelineConfig(
            enable_resize=enable_resize,
            target_width=target_width,
            color_space=color_space,
            enable_clahe_on_luminance=enable_clahe,
            enable_floor_roi=enable_floor_roi,
            floor_roi_ratio=floor_ratio,
            enable_chroma_diff=enable_chroma_diff,
            enable_edge_diff=enable_edge_diff,
            chroma_diff_thresh=chroma_diff_thresh,
            edge_diff_thresh=edge_diff_thresh,
            enable_texture_diff=enable_texture_diff,
            enable_shadow_mask=enable_shadow_mask,
            texture_diff_thresh=texture_diff_thresh,
            shadow_luma_thresh=shadow_luma_thresh,
            shadow_chroma_small_thresh=shadow_chroma_small_thresh,
            combine_mode=combine_mode,
            enable_morphology=enable_morphology,
            morph_close_ksize=ensure_odd(morph_close_ksize),
            morph_open_ksize=ensure_odd(morph_open_ksize),
            morph_iterations=morph_iterations,
            enable_area_filter=enable_area_filter,
            min_area=min_area,
            max_area=max_area,
            enable_bbox_filters=enable_bbox_filters,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            must_be_in_floor_roi=must_be_in_floor_roi,
            show_debug_windows=False,
            debug_scale=debug_scale,
        )
        st.session_state["cfg"] = cfg

    return st.session_state["cfg"]


def main() -> None:
    st.set_page_config(page_title="FloorShiftDetector", layout="wide")
    st.title("FloorShiftDetector")
    st.caption("Interactive visualization for each pipeline step.")

    if "cfg_version" not in st.session_state:
        st.session_state["cfg_version"] = 0

    if st.sidebar.button("Reset settings"):
        st.session_state.pop("cfg", None)
        st.session_state["cfg_version"] += 1

    cfg = build_config()

    st.sidebar.markdown("### Images")
    source = st.sidebar.radio("Input source", ["Upload", "Local file"], index=0)

    ref_bgr = None
    cur_bgr = None

    if source == "Upload":
        ref_file = st.sidebar.file_uploader("Reference image", type=list(IMAGE_EXTS))
        cur_file = st.sidebar.file_uploader("Current image", type=list(IMAGE_EXTS))
        ref_bgr = load_image_from_upload(ref_file)
        cur_bgr = load_image_from_upload(cur_file)
    else:
        image_paths = collect_image_paths(Path("Images"))
        ref_paths = filter_reference_paths(image_paths)
        cur_paths = filter_current_paths(image_paths)
        if ref_paths:
            ref_path = st.sidebar.selectbox("Reference image path", ref_paths, index=0)
        else:
            ref_path = st.sidebar.text_input("Reference image path", "Images/Chambre/Reference.jpg")
        if cur_paths:
            cur_path = st.sidebar.selectbox("Current image path", cur_paths, index=0)
        else:
            cur_path = st.sidebar.text_input("Current image path", "Images/Chambre/IMG_6567.jpg")
        ref_bgr = load_image_from_path(ref_path)
        cur_bgr = load_image_from_path(cur_path)

    if ref_bgr is None or cur_bgr is None:
        st.info("Provide both a reference and a current image to run the pipeline.")
        return

    with st.spinner("Running pipeline..."):
        try:
            outputs = run_pipeline_images(ref_bgr, cur_bgr, cfg)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            return

    result = outputs["result"]
    st.subheader("Result")
    st.image(to_display(result), width="stretch")

    ok, png = cv2.imencode(".png", result)
    if ok:
        st.download_button("Download result (PNG)", png.tobytes(), file_name="result.png", mime="image/png")

    st.subheader("All steps")
    items = [
        ("Reference", outputs["ref_bgr"]),
        ("Current", outputs["cur_bgr"]),
        ("Mask - Chroma", outputs["mask_chroma"]),
        ("Mask - Edge", outputs["mask_edge"]),
        ("Mask - Texture", outputs["mask_texture"]),
        ("Mask - Shadow", outputs["mask_shadow"]),
        ("Floor ROI", outputs["floor_mask"]),
        ("Mask - Combined", outputs["mask_combined"]),
        ("Mask - Cleaned", outputs["mask_cleaned"]),
        ("Result", outputs["result"]),
    ]

    for row in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (name, img) in zip(cols, items[row:row + 3]):
            col.image(to_display(img), caption=name, use_container_width=True)


if __name__ == "__main__":
    main()
