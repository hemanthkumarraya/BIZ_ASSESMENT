# ==================== CRITICAL FIXES FOR STREAMLIT CLOUD ====================
import os
# Disable file watcher that breaks PyTorch on Streamlit Cloud
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Monkey-patch to prevent torch.classes path inspection crash
import torch
if not hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []

# Optional: Suppress benign warnings
os.environ["KMP_SETTINGS"] = "false"
# ===========================================================================

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = 'streamlit_app/glove_1248_best_v1.pt'  # Make sure this path is correct in your repo
CONFIDENCE_THRESHOLD = 0.50
CLASS_NAMES = ['gloved_hand', 'bare_hand']

@st.cache_resource
def load_model():
    """Load YOLO model with error handling and caching."""
    try:
        st.info("Loading YOLOv8 model... (this takes a few seconds on first run)")
        model = YOLO(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error("Check if 'glove_1248_best_v1.pt' is in the 'streamlit_app/' folder.")
        st.stop()

def run_inference(model, image_bgr):
    """Run inference and return PIL image + detection counts."""
    try:
        # Convert BGR → RGB for display
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Run prediction (no plotting → we only need raw boxes)
        results = model.predict(
            source=image_bgr,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.45,
            verbose=False,
            stream=False,
            save=False,
            imgsz=1248
        )

        return pil_image, results[0] if results else None

    except Exception as e:
        st.error(f"Inference error: {e}")
        return None, None

# ============================= MAIN APP =============================
def main():
    st.set_page_config(
        page_title="Glove vs Bare Hand Detection",
        page_icon="gloves",
        layout="wide"
    )

    st.title("Hand Protection Compliance Detector")
    st.markdown("### Upload an image → Instantly detect gloved vs bare hands")

    st.markdown("""
    - 100% from-scratch dataset & model  
    - Trained on 503 manually annotated real-world images  
    - Live 24/7 demo · No public datasets used
    """)

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload image (JPG/PNG)",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Decode image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Invalid image file. Please upload a valid JPG/PNG.")
            return

        # Show preview
        st.sidebar.image(image_bgr, channels="BGR", caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image for hand protection compliance..."):
            pil_img, result = run_inference(model, image_bgr)

        if pil_img is None:
            st.error("Failed to process image.")
            return

        # Display results
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(pil_img, caption="Original Image", use_column_width=True)

        with col2:
            st.subheader("Compliance Report")

            if result is None or len(result.boxes) == 0:
                st.warning("No hands detected")
                st.info("No compliance issue found.")
            else:
                boxes = result.boxes
                cls = boxes.cls.cpu().numpy()
                conf = boxes.conf.cpu().numpy()

                gloved = sum(1 for c in cls if int(c) == 0)
                bare = sum(1 for c in cls if int(c) == 1)

                st.metric("Total Hands Detected", len(boxes))
                st.metric("Gloved Hands", gloved)
                st.metric("Bare Hands", bare)

                if bare > 0:
                    st.error(f"**COMPLIANCE FAILURE** – {bare} bare hand(s) detected!")
                else:
                    st.success("**FULL COMPLIANCE** – All hands are properly gloved")

            st.caption(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")

    else:
        st.info("👆 Upload an image to get started")

    st.markdown("---")
    st.markdown("Built 100% from scratch by **Hemanth Kumar** · Ready for production deployment")

if __name__ == "__main__":
    main()
