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
    "floor_seed_ratio": "params_floor_seed_ratio",
    "floor_k": "params_floor_k",
    "floor_seed_quantile": "params_floor_seed_quantile",
    "floor_texture_window": "params_floor_texture_window",
    "floor_texture_blur": "params_floor_texture_blur",
    "floor_w_l": "params_floor_w_l",
    "floor_w_ab": "params_floor_w_ab",
    "floor_w_tex": "params_floor_w_tex",
    "floor_clean_close_ksize": "params_floor_clean_close_ksize",
    "floor_clean_open_ksize": "params_floor_clean_open_ksize",
    "floor_keep_bottom_connected": "params_floor_keep_bottom_connected",
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
        PARAM_KEYS["floor_seed_ratio"]: float(cfg.floor_seed_ratio),
        PARAM_KEYS["floor_k"]: int(cfg.floor_k),
        PARAM_KEYS["floor_seed_quantile"]: float(cfg.floor_seed_quantile),
        PARAM_KEYS["floor_texture_window"]: int(cfg.floor_texture_window),
        PARAM_KEYS["floor_texture_blur"]: int(cfg.floor_texture_blur),
        PARAM_KEYS["floor_w_l"]: float(cfg.floor_w_l),
        PARAM_KEYS["floor_w_ab"]: float(cfg.floor_w_ab),
        PARAM_KEYS["floor_w_tex"]: float(cfg.floor_w_tex),
        PARAM_KEYS["floor_clean_close_ksize"]: cfg.floor_clean_close_ksize,
        PARAM_KEYS["floor_clean_open_ksize"]: cfg.floor_clean_open_ksize,
        PARAM_KEYS["floor_keep_bottom_connected"]: cfg.floor_keep_bottom_connected,
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
            "floor_seed_ratio",
            "floor_k",
            "floor_seed_quantile",
            "floor_texture_window",
            "floor_texture_blur",
            "floor_w_l",
            "floor_w_ab",
            "floor_w_tex",
            "floor_clean_close_ksize",
            "floor_clean_open_ksize",
            "floor_keep_bottom_connected",
        ]
        if any(not hasattr(cfg, name) for name in missing):
            cfg = PipelineConfig(show_debug_windows=False)
            st.session_state["cfg"] = cfg
            for key in PARAM_KEYS.values():
                st.session_state.pop(key, None)

    init_param_defaults(cfg)

    st.sidebar.markdown("### Settings")
    with st.sidebar.form("params_form"):
        with st.expander("Pipeline settings", expanded=False):
            enable_resize = st.checkbox(
                "Resize images",
                key=PARAM_KEYS["enable_resize"],
                help="Resize to target width to normalize scale and speed up processing.",
            )
            target_width = st.slider(
                "Target width",
                320,
                1920,
                step=40,
                key=PARAM_KEYS["target_width"],
                help="Resize width in pixels (aspect ratio preserved).",
            )

            color_space = st.selectbox(
                "Color space",
                ["LAB", "YCrCb", "HSV"],
                index=["LAB", "YCrCb", "HSV"].index(st.session_state[PARAM_KEYS["color_space"]]),
                key=PARAM_KEYS["color_space"],
                help="Color space used for chroma/shadow computations.",
            )
            enable_clahe = st.checkbox(
                "Apply CLAHE (luminance)",
                key=PARAM_KEYS["enable_clahe"],
                help="Normalize lighting on the luminance channel.",
            )

        with st.expander("Floor ROI (multicluster)", expanded=False):
            enable_floor_roi = st.checkbox(
                "Enable floor ROI",
                key=PARAM_KEYS["enable_floor_roi"],
                help="Restrict detection to the estimated floor mask.",
            )
            floor_seed_ratio = st.slider(
                "Floor seed ratio (bottom band)",
                0.05,
                0.5,
                step=0.05,
                key=PARAM_KEYS["floor_seed_ratio"],
                help="Bottom band used to learn floor appearance.",
            )
            floor_k = st.slider(
                "Number of clusters (K)",
                1,
                6,
                step=1,
                key=PARAM_KEYS["floor_k"],
                help="Number of appearance clusters learned from the bottom band.",
            )
            floor_seed_quantile = st.slider(
                "Seed distance quantile",
                0.5,
                0.99,
                step=0.01,
                key=PARAM_KEYS["floor_seed_quantile"],
                help="Threshold: higher keeps more pixels as floor.",
            )

            with st.expander("Floor ROI advanced", expanded=False):
                floor_texture_window = st.slider(
                    "Texture window",
                    5,
                    25,
                    step=2,
                    key=PARAM_KEYS["floor_texture_window"],
                    help="Window size for the texture map (odd).",
                )
                floor_texture_blur = st.slider(
                    "Texture blur",
                    0,
                    15,
                    step=1,
                    key=PARAM_KEYS["floor_texture_blur"],
                    help="Pre-blur before computing gradients (0 = none).",
                )
                floor_w_l = st.slider(
                    "Weight L",
                    0.0,
                    1.0,
                    step=0.05,
                    key=PARAM_KEYS["floor_w_l"],
                    help="Weight for luminance in the floor feature vector.",
                )
                floor_w_ab = st.slider(
                    "Weight a,b",
                    0.0,
                    2.0,
                    step=0.05,
                    key=PARAM_KEYS["floor_w_ab"],
                    help="Weight for chroma (a,b) in the floor feature vector.",
                )
                floor_w_tex = st.slider(
                    "Weight texture",
                    0.0,
                    2.0,
                    step=0.05,
                    key=PARAM_KEYS["floor_w_tex"],
                    help="Weight for texture in the floor feature vector.",
                )
                floor_clean_close_ksize = st.slider(
                    "Floor mask close kernel",
                    1,
                    25,
                    step=2,
                    key=PARAM_KEYS["floor_clean_close_ksize"],
                    help="Fill small holes in the floor mask.",
                )
                floor_clean_open_ksize = st.slider(
                    "Floor mask open kernel",
                    1,
                    25,
                    step=2,
                    key=PARAM_KEYS["floor_clean_open_ksize"],
                    help="Remove small noisy regions in the floor mask.",
                )
                floor_keep_bottom_connected = st.checkbox(
                    "Keep only components touching bottom",
                    key=PARAM_KEYS["floor_keep_bottom_connected"],
                    help="Keep only regions connected to the bottom row.",
                )

        with st.expander("Change detection", expanded=False):
            enable_chroma_diff = st.checkbox(
                "Chroma diff",
                key=PARAM_KEYS["enable_chroma_diff"],
                help="Detect color changes (less sensitive to lighting).",
            )
            chroma_diff_thresh = st.slider(
                "Chroma diff threshold",
                0,
                80,
                step=1,
                key=PARAM_KEYS["chroma_diff_thresh"],
                help="Higher = fewer detections.",
            )
            enable_edge_diff = st.checkbox(
                "Edge diff",
                key=PARAM_KEYS["enable_edge_diff"],
                help="Detect changes in edges/gradients.",
            )
            edge_diff_thresh = st.slider(
                "Edge diff threshold",
                0,
                120,
                step=1,
                key=PARAM_KEYS["edge_diff_thresh"],
                help="Higher = fewer detections.",
            )
            enable_texture_diff = st.checkbox(
                "Texture diff",
                key=PARAM_KEYS["enable_texture_diff"],
                help="Detect changes in local texture/roughness.",
            )
            texture_diff_thresh = st.slider(
                "Texture diff threshold",
                0,
                80,
                step=1,
                key=PARAM_KEYS["texture_diff_thresh"],
                help="Higher = fewer detections.",
            )
            enable_shadow_mask = st.checkbox(
                "Shadow mask",
                key=PARAM_KEYS["enable_shadow_mask"],
                help="Detect shadows to subtract from other masks.",
            )
            shadow_luma_thresh = st.slider(
                "Shadow luma threshold",
                0,
                80,
                step=1,
                key=PARAM_KEYS["shadow_luma_thresh"],
                help="Minimum luminance change to consider shadow.",
            )
            shadow_chroma_small_thresh = st.slider(
                "Shadow chroma small threshold",
                0,
                40,
                step=1,
                key=PARAM_KEYS["shadow_chroma_small_thresh"],
                help="Max chroma change to still be considered shadow.",
            )
            combine_mode = st.selectbox(
                "Combine mode",
                ["OR", "AND"],
                index=0 if st.session_state[PARAM_KEYS["combine_mode"]] == "OR" else 1,
                key=PARAM_KEYS["combine_mode"],
                help="OR = union of masks, AND = intersection.",
            )

        with st.expander("Morphology", expanded=False):
            enable_morphology = st.checkbox(
                "Enable morphology",
                key=PARAM_KEYS["enable_morphology"],
                help="Clean the combined mask.",
            )
            morph_close_ksize = st.slider(
                "Close kernel size",
                1,
                25,
                step=2,
                key=PARAM_KEYS["morph_close_ksize"],
                help="Fill small gaps in the mask.",
            )
            morph_open_ksize = st.slider(
                "Open kernel size",
                1,
                25,
                step=2,
                key=PARAM_KEYS["morph_open_ksize"],
                help="Remove small noisy blobs.",
            )
            morph_iterations = st.slider(
                "Morph iterations",
                1,
                4,
                step=1,
                key=PARAM_KEYS["morph_iterations"],
                help="Number of times to apply morphology.",
            )

        with st.expander("Filtering", expanded=False):
            enable_area_filter = st.checkbox(
                "Enable area filter",
                key=PARAM_KEYS["enable_area_filter"],
                help="Remove components that are too small/large.",
            )
            min_area = st.slider(
                "Min area",
                0,
                2000,
                step=50,
                key=PARAM_KEYS["min_area"],
                help="Minimum component area to keep.",
            )
            max_area = st.slider(
                "Max area",
                1000,
                300000,
                step=1000,
                key=PARAM_KEYS["max_area"],
                help="Maximum component area to keep.",
            )

            enable_bbox_filters = st.checkbox(
                "Enable bbox filters",
                key=PARAM_KEYS["enable_bbox_filters"],
                help="Filter bounding boxes by aspect ratio and ROI.",
            )
            min_aspect_ratio = st.slider(
                "Min aspect ratio",
                0.05,
                1.5,
                step=0.05,
                key=PARAM_KEYS["min_aspect_ratio"],
                help="Minimum width/height ratio to keep.",
            )
            max_aspect_ratio = st.slider(
                "Max aspect ratio",
                1.5,
                10.0,
                step=0.1,
                key=PARAM_KEYS["max_aspect_ratio"],
                help="Maximum width/height ratio to keep.",
            )
            must_be_in_floor_roi = st.checkbox(
                "BBox center must be in floor ROI",
                key=PARAM_KEYS["must_be_in_floor_roi"],
                help="Reject boxes whose center is outside the floor mask.",
            )

        apply_col, reset_col = st.columns(2)
        with apply_col:
            submitted_apply = st.form_submit_button("Apply settings")
        with reset_col:
            submitted_reset = st.form_submit_button("Reset settings")

    if submitted_reset:
        st.session_state.pop("cfg", None)
        for key in PARAM_KEYS.values():
            st.session_state.pop(key, None)
        cfg = PipelineConfig(show_debug_windows=False)
        st.session_state["cfg"] = cfg
        return cfg

    if submitted_apply:
        cfg = PipelineConfig(
            enable_resize=enable_resize,
            target_width=target_width,
            color_space=color_space,
            enable_clahe_on_luminance=enable_clahe,
            enable_floor_roi=enable_floor_roi,
            floor_roi_ratio=cfg.floor_roi_ratio,
            floor_seed_ratio=floor_seed_ratio,
            floor_k=floor_k,
            floor_seed_quantile=floor_seed_quantile,
            floor_texture_window=floor_texture_window,
            floor_texture_blur=floor_texture_blur,
            floor_w_l=floor_w_l,
            floor_w_ab=floor_w_ab,
            floor_w_tex=floor_w_tex,
            floor_clean_close_ksize=ensure_odd(floor_clean_close_ksize),
            floor_clean_open_ksize=ensure_odd(floor_clean_open_ksize),
            floor_keep_bottom_connected=floor_keep_bottom_connected,
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
            debug_scale=cfg.debug_scale,
        )
        st.session_state["cfg"] = cfg

    return st.session_state["cfg"]


def main() -> None:
    st.set_page_config(page_title="FloorShiftDetector", layout="wide")
    st.title("FloorShiftDetector")
    st.caption("Interactive visualization for each pipeline step.")

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
