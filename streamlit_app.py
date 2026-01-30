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

PARAM_KEYS = {
    "enable_resize": "params_enable_resize",
    "target_width": "params_target_width",
    "color_space": "params_color_space",
    "enable_clahe": "params_enable_clahe",
    "enable_floor_roi": "params_enable_floor_roi",
    "floor_ratio": "params_floor_ratio",
    "enable_chroma_diff": "params_enable_chroma_diff",
    "chroma_diff_thresh": "params_chroma_diff_thresh",
    "enable_edge_diff": "params_enable_edge_diff",
    "edge_diff_thresh": "params_edge_diff_thresh",
    "enable_texture_diff": "params_enable_texture_diff",
    "texture_diff_thresh": "params_texture_diff_thresh",
    "enable_shadow_mask": "params_enable_shadow_mask",
    "shadow_luma_thresh": "params_shadow_luma_thresh",
    "shadow_chroma_small_thresh": "params_shadow_chroma_small_thresh",
    "combine_mode": "params_combine_mode",
    "enable_morphology": "params_enable_morphology",
    "morph_close_ksize": "params_morph_close_ksize",
    "morph_open_ksize": "params_morph_open_ksize",
    "morph_iterations": "params_morph_iterations",
    "enable_area_filter": "params_enable_area_filter",
    "min_area": "params_min_area",
    "max_area": "params_max_area",
    "enable_bbox_filters": "params_enable_bbox_filters",
    "min_aspect_ratio": "params_min_aspect_ratio",
    "max_aspect_ratio": "params_max_aspect_ratio",
    "must_be_in_floor_roi": "params_must_be_in_floor_roi",
    "debug_scale": "params_debug_scale",
}


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


def init_param_defaults(cfg: PipelineConfig) -> None:
    defaults = {
        PARAM_KEYS["enable_resize"]: cfg.enable_resize,
        PARAM_KEYS["target_width"]: cfg.target_width,
        PARAM_KEYS["color_space"]: cfg.color_space,
        PARAM_KEYS["enable_clahe"]: cfg.enable_clahe_on_luminance,
        PARAM_KEYS["enable_floor_roi"]: cfg.enable_floor_roi,
        PARAM_KEYS["floor_ratio"]: float(cfg.floor_roi_ratio),
        PARAM_KEYS["enable_chroma_diff"]: cfg.enable_chroma_diff,
        PARAM_KEYS["chroma_diff_thresh"]: cfg.chroma_diff_thresh,
        PARAM_KEYS["enable_edge_diff"]: cfg.enable_edge_diff,
        PARAM_KEYS["edge_diff_thresh"]: cfg.edge_diff_thresh,
        PARAM_KEYS["enable_texture_diff"]: cfg.enable_texture_diff,
        PARAM_KEYS["texture_diff_thresh"]: cfg.texture_diff_thresh,
        PARAM_KEYS["enable_shadow_mask"]: cfg.enable_shadow_mask,
        PARAM_KEYS["shadow_luma_thresh"]: cfg.shadow_luma_thresh,
        PARAM_KEYS["shadow_chroma_small_thresh"]: cfg.shadow_chroma_small_thresh,
        PARAM_KEYS["combine_mode"]: cfg.combine_mode,
        PARAM_KEYS["enable_morphology"]: cfg.enable_morphology,
        PARAM_KEYS["morph_close_ksize"]: cfg.morph_close_ksize,
        PARAM_KEYS["morph_open_ksize"]: cfg.morph_open_ksize,
        PARAM_KEYS["morph_iterations"]: cfg.morph_iterations,
        PARAM_KEYS["enable_area_filter"]: cfg.enable_area_filter,
        PARAM_KEYS["min_area"]: cfg.min_area,
        PARAM_KEYS["max_area"]: cfg.max_area,
        PARAM_KEYS["enable_bbox_filters"]: cfg.enable_bbox_filters,
        PARAM_KEYS["min_aspect_ratio"]: float(cfg.min_aspect_ratio),
        PARAM_KEYS["max_aspect_ratio"]: float(cfg.max_aspect_ratio),
        PARAM_KEYS["must_be_in_floor_roi"]: cfg.must_be_in_floor_roi,
        PARAM_KEYS["debug_scale"]: float(cfg.debug_scale),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
            for key in PARAM_KEYS.values():
                st.session_state.pop(key, None)

    init_param_defaults(cfg)

    with st.sidebar.form("params_form"):
        st.markdown("### Pipeline settings")
        enable_resize = st.checkbox("Resize images", key=PARAM_KEYS["enable_resize"])
        target_width = st.slider("Target width", 320, 1920, step=40, key=PARAM_KEYS["target_width"])

        color_space = st.selectbox(
            "Color space",
            ["LAB", "YCrCb", "HSV"],
            index=["LAB", "YCrCb", "HSV"].index(st.session_state[PARAM_KEYS["color_space"]]),
            key=PARAM_KEYS["color_space"],
        )
        enable_clahe = st.checkbox("Apply CLAHE (luminance)", key=PARAM_KEYS["enable_clahe"])

        st.markdown("### Floor ROI")
        enable_floor_roi = st.checkbox("Enable floor ROI", key=PARAM_KEYS["enable_floor_roi"])
        floor_ratio = st.slider("Floor ROI ratio", 0.1, 0.9, step=0.05, key=PARAM_KEYS["floor_ratio"])

        st.markdown("### Change detection")
        enable_chroma_diff = st.checkbox("Chroma diff", key=PARAM_KEYS["enable_chroma_diff"])
        chroma_diff_thresh = st.slider("Chroma diff threshold", 0, 80, step=1, key=PARAM_KEYS["chroma_diff_thresh"])
        enable_edge_diff = st.checkbox("Edge diff", key=PARAM_KEYS["enable_edge_diff"])
        edge_diff_thresh = st.slider("Edge diff threshold", 0, 120, step=1, key=PARAM_KEYS["edge_diff_thresh"])
        enable_texture_diff = st.checkbox("Texture diff", key=PARAM_KEYS["enable_texture_diff"])
        texture_diff_thresh = st.slider("Texture diff threshold", 0, 80, step=1, key=PARAM_KEYS["texture_diff_thresh"])
        enable_shadow_mask = st.checkbox("Shadow mask", key=PARAM_KEYS["enable_shadow_mask"])
        shadow_luma_thresh = st.slider("Shadow luma threshold", 0, 80, step=1, key=PARAM_KEYS["shadow_luma_thresh"])
        shadow_chroma_small_thresh = st.slider(
            "Shadow chroma small threshold",
            0,
            40,
            step=1,
            key=PARAM_KEYS["shadow_chroma_small_thresh"],
        )
        combine_mode = st.selectbox(
            "Combine mode",
            ["OR", "AND"],
            index=0 if st.session_state[PARAM_KEYS["combine_mode"]] == "OR" else 1,
            key=PARAM_KEYS["combine_mode"],
        )

        st.markdown("### Morphology")
        enable_morphology = st.checkbox("Enable morphology", key=PARAM_KEYS["enable_morphology"])
        morph_close_ksize = st.slider("Close kernel size", 1, 25, step=2, key=PARAM_KEYS["morph_close_ksize"])
        morph_open_ksize = st.slider("Open kernel size", 1, 25, step=2, key=PARAM_KEYS["morph_open_ksize"])
        morph_iterations = st.slider("Morph iterations", 1, 4, step=1, key=PARAM_KEYS["morph_iterations"])

        st.markdown("### Filtering")
        enable_area_filter = st.checkbox("Enable area filter", key=PARAM_KEYS["enable_area_filter"])
        min_area = st.slider("Min area", 0, 2000, step=50, key=PARAM_KEYS["min_area"])
        max_area = st.slider("Max area", 1000, 300000, step=1000, key=PARAM_KEYS["max_area"])

        enable_bbox_filters = st.checkbox("Enable bbox filters", key=PARAM_KEYS["enable_bbox_filters"])
        min_aspect_ratio = st.slider("Min aspect ratio", 0.05, 1.5, step=0.05, key=PARAM_KEYS["min_aspect_ratio"])
        max_aspect_ratio = st.slider("Max aspect ratio", 1.5, 10.0, step=0.1, key=PARAM_KEYS["max_aspect_ratio"])
        must_be_in_floor_roi = st.checkbox("BBox center must be in floor ROI", key=PARAM_KEYS["must_be_in_floor_roi"])

        st.markdown("### Display")
        debug_scale = st.slider("Display scale", 0.3, 1.0, step=0.05, key=PARAM_KEYS["debug_scale"])

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
        for key in PARAM_KEYS.values():
            st.session_state.pop(key, None)
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
    ]

    if cfg.enable_chroma_diff:
        items.append(("Mask - Chroma", outputs["mask_chroma"]))
    if cfg.enable_edge_diff:
        items.append(("Mask - Edge", outputs["mask_edge"]))
    if cfg.enable_texture_diff:
        items.append(("Mask - Texture", outputs["mask_texture"]))
    if cfg.enable_shadow_mask:
        items.append(("Mask - Shadow", outputs["mask_shadow"]))
    if cfg.enable_floor_roi:
        items.append(("Floor ROI", outputs["floor_mask"]))

    any_change_mask = (
        cfg.enable_chroma_diff
        or cfg.enable_edge_diff
        or cfg.enable_texture_diff
        or cfg.enable_shadow_mask
    )
    if any_change_mask:
        items.append(("Mask - Combined", outputs["mask_combined"]))
        if cfg.enable_morphology:
            items.append(("Mask - Cleaned", outputs["mask_cleaned"]))

    items.append(("Result", outputs["result"]))

    for row in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (name, img) in zip(cols, items[row:row + 3]):
            col.image(to_display(img), caption=name, width="stretch")


if __name__ == "__main__":
    main()
