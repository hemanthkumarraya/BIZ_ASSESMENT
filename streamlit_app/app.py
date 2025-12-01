import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# --- Configuration ---
# Your model path (kept the same as provided)
MODEL_PATH = 'streamlit_app/glove_1248_best_v1.pt' 

st.set_page_config(
    page_title="Hand Safety Compliance Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
# Use st.cache_resource to load the model only once, making the app fast
@st.cache_resource
def load_detection_model(path):
    model = YOLO(path)
    return model

try:
    model = load_detection_model(MODEL_PATH)
except Exception as e:
    st.error(f"⚠️ Error loading model at {MODEL_PATH}. Ensure the path is correct and the file exists.")
    st.error(e)
    st.stop()

# --- Streamlit UI ---
st.title("🏭 Safety Compliance: Hand Detection")
st.subheader("Gloved vs. Bare Hand Monitoring")

st.sidebar.header("Configuration")
# Confidence slider for user control
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6)

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image (.jpg, .jpeg, .png) for analysis:", 
                                 type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert uploaded file to PIL Image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, caption=uploaded_file.name, use_column_width=True)

    if st.sidebar.button("Run Compliance Check"):
        with st.spinner('Running object detection...'):
            # --- Inference ---
            results = model.predict(image, conf=confidence, iou=0.7)
            
            # Get the annotated image
            res_plotted = results[0].plot() 
            annotated_image = Image.fromarray(res_plotted[..., ::-1])

        with col2:
            st.subheader("Detected Results")
            st.image(annotated_image, caption="Compliance Check", use_column_width=True)
            
        # --- Compliance and Counting Output ---
        st.markdown("---")
        
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            
            # 1. Get all detected class names
            detected_classes = [model.names[int(cls.item())] for cls in boxes.cls]
            class_counts = Counter(detected_classes)

            # 2. Extract counts for the specific classes
            glove_count = class_counts.get('GLOVE', 0)
            no_glove_count = class_counts.get('NO_GLOVE', 0)

            st.subheader("Hand Compliance Status & Count")
            
            # 3. Display Counts
            st.info(f"**Class: 0 (GLOVE)** Count: **{glove_count}**")
            st.warning(f"**Class: 1 (NO_GLOVE)** Count: **{no_glove_count}**")
            
            # 4. Display Compliance Status
            if no_glove_count > 0:
                st.error("🚨 **NON-COMPLIANT:** Bare hands detected! Immediate action required.")
            elif glove_count > 0:
                st.success("✅ **COMPLIANT:** Only gloved hands detected.")
            else:
                st.info("No hands detected above the set confidence threshold.")

            # (Optional: Keep the detailed output collapsed for the assessment)
            with st.expander("View Detailed Bounding Box and JSON Output"):
                detections = []
                for box in boxes:
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0].tolist()]
                    conf = round(box.conf[0].item(), 3)
                    label = model.names[int(box.cls[0].item())]
                    
                    detections.append({
                        "label": label, 
                        "confidence": conf, 
                        "bbox": [x1, y1, x2, y2]
                    })
                
                # Display the required JSON output
                st.json({
                    "filename": uploaded_file.name,
                    "detections": detections
                })
        else:
            st.warning("✅ No hands detected or all detections are below the confidence threshold.")
