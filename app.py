#!/usr/bin/env python3
"""Streamlit Web UI for Eyewear Localization & Brand Attribution on Kaggle GPU."""

import json
import os
import sys
import time
from pathlib import Path
from PIL import Image
import streamlit as st

# Ensure eyewear-localization package is in Python path
repo_root = Path(__file__).resolve().parent
app_dir = repo_root / "eyewear-localization"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from eyewear_localization.config import LocalizationConfig
from eyewear_localization.gazetteer import Gazetteer
from eyewear_localization.perception import (
    TextDetection,
    build_native_sam3_localizer,
    build_ocr_backend,
)
from eyewear_localization.visualization import annotate
from scripts.prototype_single_pass_ocr import run_single_pass_prototype, SAM3_PROFILES

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Eyewear Localization & Brand Attribution",
    page_icon="🕶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🕶️ Eyewear Localization & Brand Attribution Dashboard")
st.markdown("Interactive GPU-accelerated vision pipeline for retail display object counting and decoupled spatial brand attribution.")

# --- Helper: Model Caching ---
@st.cache_resource(show_spinner="Loading Native SAM3 (RapidOCR only, Florence off) onto GPU...")
def load_models(sam3_checkpoint_path: str, device: str, brand_file_path: str):
    brands = [line.strip() for line in Path(brand_file_path).read_text().splitlines() if line.strip()]
    gazetteer = Gazetteer(brands)
    
    localizer = build_native_sam3_localizer(sam3_checkpoint_path, device=device)
    ocr_backend = build_ocr_backend(
        "rapidocr+florence2",
        gpu=device,
        gazetteer=gazetteer,
        scale=1.0,
        florence_scene="off",
    )
    if hasattr(ocr_backend, "max_fallback_calls"):
        ocr_backend.max_fallback_calls = 99
        
    return localizer, ocr_backend, gazetteer, brands

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Pipeline Configuration")

# Image Selection Mode
sample_dir = repo_root / "item-count-image-samples"
sample_images = sorted(list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")))

# Pre-loaded samples are served from the mounted Kaggle dataset
# (faizankhan101/eyewear-test-samples) when running on Kaggle GPU.
kaggle_input_dir = Path("/kaggle/input/eyewear-test-samples")
if kaggle_input_dir.is_dir():
    sample_images += sorted(
        p for p in kaggle_input_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
seen: set[str] = set()
sample_images = [p for p in sample_images if not (str(p) in seen or seen.add(str(p)))]
sample_dict = {img.name: img for img in sample_images}

if sample_images:
    image_source = st.sidebar.radio(
        "Image Source",
        options=["Select Pre-loaded Sample", "Upload Custom Image"],
        index=0,
    )
else:
    st.sidebar.warning("No sample images found; using upload-only mode.")
    image_source = "Upload Custom Image"

selected_image_path = None
uploaded_image_file = None

if image_source == "Select Pre-loaded Sample":
    selected_sample_name = st.sidebar.selectbox(
        "Choose Retail Display Image",
        options=list(sample_dict.keys()),
        index=1 if "IMG_0278.jpg" in sample_dict else 0,
    )
    selected_image_path = sample_dict[selected_sample_name]
else:
    uploaded_image_file = st.sidebar.file_uploader(
        "Upload Image (.jpg, .png, .webp)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_image_file:
        temp_dir = repo_root / "output" / "temp_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        selected_image_path = temp_dir / uploaded_image_file.name
        selected_image_path.write_bytes(uploaded_image_file.getbuffer())

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Finalized Config")
st.sidebar.caption(
    "`fast/off/single@1.0` — SAM3 fast profile (3 class + 3 scene prompts), "
    "single full-frame RapidOCR pass @ 1.0x, no Florence. ~10s/image, "
    "7/9 GT accuracy."
)

c1_thresh = st.sidebar.slider(
    "Cascade C1 Threshold (t1)",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05,
    help="Min confidence for physical frame text direct assignment",
)

c2_thresh = st.sidebar.slider(
    "Cascade C2 Threshold (t2)",
    min_value=0.50,
    max_value=0.95,
    value=0.75,
    step=0.05,
    help="Min confidence for display zone header/base sign scope",
)

enable_highest_confidence_fallback = st.sidebar.checkbox(
    "Enable Tier 3 Highest-Confidence Fallback",
    value=True,
    help="Resolves low-margin C2 scopes instead of abstaining to unknown",
)

sam3_checkpoint_default = repo_root / "sam3-verbose-counting" / "checkpoints" / "sam3.pt"
brand_file_default = app_dir / "brands.txt"

# --- Main Interface Execution ---
if selected_image_path is None or not selected_image_path.exists():
    st.info("👈 Please select or upload an image from the sidebar to begin inference.")
else:
    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.path.exists("/dev/nvidia0") else "cpu"
    
    with st.spinner("Initializing models and running inference..."):
        try:
            localizer, ocr_backend, gazetteer, brands = load_models(
                str(sam3_checkpoint_default),
                device,
                str(brand_file_default),
            )
            
            config = LocalizationConfig(
                gazetteer=brands,
                confidence_threshold=c1_thresh,
                enable_highest_confidence_fallback=enable_highest_confidence_fallback,
            )
            
            # Run inference with the finalized fast/off/single@1.0 config
            fast = SAM3_PROFILES["fast"]
            result = run_single_pass_prototype(
                selected_image_path,
                localizer,
                ocr_backend,
                gazetteer,
                config,
                class_prompts=fast["class_prompts"],
                signage_prompts=fast["signage_prompts"],
                person_prompts=fast["person_prompts"],
                poster_prompts=fast["poster_prompts"],
                shelf_prompts=fast["shelf_prompts"],
                ocr_mode="single",
            )
            
            # --- KPI Dashboard Metrics ---
            t = result.get("timings", {})
            c = result.get("counts", {})
            outputs = result.get("outputs", [])
            signs = result.get("signs", [])
            
            brand_counts = {}
            for out in outputs:
                b = out.get("brand", "unknown")
                brand_counts[b] = brand_counts.get(b, 0) + 1

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Time", f"{t.get('total_pipeline_seconds', 0.0):.2f}s")
            col2.metric("SAM3 Time", f"{t.get('sam3_time_seconds', 0.0):.2f}s")
            col3.metric("Tiled OCR Time", f"{t.get('single_pass_ocr_seconds', 0.0):.2f}s")
            col4.metric("Frames Found", len(outputs))
            col5.metric("Signs Detected", len(signs))
            
            st.markdown(f"**Brand Breakdown**: `{json.dumps(brand_counts)}`")
            st.markdown("---")
            
            # --- Visual Result Overlay Columns ---
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.subheader("📷 Original Input Image")
                st.image(str(selected_image_path), use_container_width=True)
                
            with vcol2:
                st.subheader("🎯 Pipeline Bounding-Box Overlay")
                annotated_img = annotate(selected_image_path, result)
                st.image(annotated_img, use_container_width=True)
                
            st.markdown("---")
            
            # --- Detailed Data Tabs ---
            tab1, tab2, tab3 = st.tabs(["📋 Frame Attributions", "🪧 Detected Signage", "🔍 Raw JSON Contract"])
            
            with tab1:
                st.subheader("Localized Eyewear Frame Attributions")
                import pandas as pd
                df_out = pd.DataFrame(outputs)
                if not df_out.empty:
                    cols_to_show = [col for col in ["instance_id", "brand", "decision_path", "confidence", "product_brand", "zone_brand", "abstained"] if col in df_out.columns]
                    st.dataframe(df_out[cols_to_show], use_container_width=True)
                else:
                    st.write("No instances detected.")

            with tab2:
                st.subheader("Detected Signs & Placards")
                df_signs = pd.DataFrame(signs)
                if not df_signs.empty:
                    st.dataframe(df_signs, use_container_width=True)
                else:
                    st.write("No physical signs detected.")

            with tab3:
                st.subheader("Complete Output JSON Contract")
                st.json(result)
                
        except Exception as err:
            st.error(f"Inference error: {err}")
            st.exception(err)
