
import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = 'streamlit_app/glove_1248_best_v1.pt'
CONFIDENCE_THRESHOLD = 0.50
CLASS_NAMES = ['bare_hand', 'gloved_hand']

@st.cache_resource
def load_model():
    """Loads the YOLO model only once and caches it."""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model {MODEL_PATH}: {e}")
        st.stop()

def run_inference(model, image_data):
    """Performs inference on the uploaded image."""
    try:
        # Predict on the image
        results = model.predict(
            source=image_data, 
            conf=CONFIDENCE_THRESHOLD, 
            iou=0.5, 
            save=False, 
            stream=False
        )
        
        # Get the annotated image (converted back to PIL for Streamlit display)
        if results and results[0] is not None:
            # The .plot() method returns an annotated NumPy array (BGR format)
            annotated_img_bgr = results[0].plot()
            
            # Convert BGR (OpenCV format) to RGB (PIL/Streamlit format)
            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert NumPy array to PIL Image object
            annotated_image = Image.fromarray(annotated_img_rgb)
            return annotated_image, results
        
        return Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)), []

    except Exception as e:
        st.error(f"An error occurred during inference: {e}")
        return None, []

def main():
    """Main Streamlit application function."""
    st.set_page_config(page_title="Glove/No-Glove Detection", layout="wide")

    st.title("🧤 Hand Protection Detection (YOLOv8)")
    st.markdown("""
        Upload an image below to run the object detection model (`glove_1248_best_v0.pt`). 
        The application will identify **gloved hands** and **bare hands**.
    """)

    # Load the model
    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Display the image and run inference
        
        # Read the uploaded file as a NumPy array (OpenCV format)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_data = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.sidebar.image(image_data, channels="BGR", caption="Original Image", use_column_width=True)
        st.sidebar.markdown("---")

        with st.spinner('Running detection...'):
            annotated_image, results = run_inference(model, image_data)

        if annotated_image:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Detection Results")
                st.image(annotated_image, caption="Annotated Image", use_column_width=True)

            with col2:
                st.subheader("Detection Summary")
                
                total_detections = 0
                gloved_count = 0
                bare_count = 0

                if results and results[0]:
                    boxes = results[0].boxes
                    total_detections = len(boxes)
                    
                    for box in boxes:
                        class_id = int(box.cls[0])
                        label = CLASS_NAMES[class_id]
                        
                        if label == 'gloved_hand':
                            gloved_count += 1
                        elif label == 'bare_hand':
                            bare_count += 1

                st.metric(label="Total Hands Detected", value=total_detections)
                st.metric(label="✅ Gloved Hands", value=gloved_count)
                st.metric(label="⚠️ Bare Hands", value=bare_count)
                
                if bare_count > 0:
                    st.warning("Immediate action required: Bare hands detected.")
                elif total_detections > 0:
                    st.success("Compliance Check: All detected hands are gloved.")
                
                st.markdown(f"---")
                st.caption(f"Confidence Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%")

if __name__ == '__main__':
    main()
