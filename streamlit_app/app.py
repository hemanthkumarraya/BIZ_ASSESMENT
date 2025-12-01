import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- Configuration ---
# 1. Path to your trained YOLOv8 model (e.g., in a 'weights' folder)
MODEL_PATH = 'weights/best.pt' 

st.set_page_config(
    page_title="Hand Safety Compliance Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
# Use st.cache_resource to load the model only once, making the app fast
@st.cache_resource
def load_detection_model(path):
    # The YOLO model loads class names (GLOVE, NO_GLOVE) from the .pt file
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
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

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
            # iou=0.7 is a standard setting for Non-Maximum Suppression (NMS)
            results = model.predict(image, conf=confidence, iou=0.7)
            
            # Get the annotated image (YOLOv8 plotting returns a BGR NumPy array)
            res_plotted = results[0].plot() 
            
            # Convert BGR to RGB for Streamlit display
            annotated_image = Image.fromarray(res_plotted[..., ::-1])

        with col2:
            st.subheader("Detected Results")
            st.image(annotated_image, caption="Compliance Check", use_column_width=True)
            
        # --- Display Structured Data ---
        detections = []
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            st.markdown("---")
            st.subheader("Detailed Detection Log (JSON Format)")
            
            for box in boxes:
                # Bounding box in x1, y1, x2, y2 format (pixel coordinates)
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0].tolist()]
                
                # Confidence score
                conf = round(box.conf[0].item(), 3)
                
                # Class name (which should be 'GLOVE' or 'NO_GLOVE')
                label = model.names[int(box.cls[0].item())]
                
                detections.append({
                    "label": label, 
                    "confidence": conf, 
                    "bbox": [x1, y1, x2, y2]
                })

            # Display a summary table for quick review
            st.dataframe(detections, use_container_width=True)
            
            # Display the exact JSON output required in the assessment
            with st.expander("View Raw JSON Output"):
                st.json({
                    "filename": uploaded_file.name,
                    "detections": detections
                })
        else:
            st.warning("✅ No hands detected or all detections are below the confidence threshold.")
