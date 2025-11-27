import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = 'streamlit_app/glove_1248_best_v1.pt'
CONFIDENCE_THRESHOLD = 0.50
CLASS_NAMES = ['gloved_hand','bare_hand']

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
    """
    Performs inference on the uploaded image.
    Returns the original image (as a PIL object) and the raw detection results.
    """
    try:
        # Convert BGR (OpenCV format) to RGB for PIL/Streamlit display
        original_img_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(original_img_rgb)
        
        # Predict on the image. We rely on the raw 'results' object for counts 
        # and do not use the .plot() method, which would draw annotations.
        results = model.predict(
            source=image_data, 
            conf=CONFIDENCE_THRESHOLD, 
            iou=0.5, 
            save=False, 
            stream=False,
        )
        
        # The results object contains the bounding box and class information
        return original_image, results

    except Exception as e:
        st.error(f"An error occurred during inference: {e}")
        return None, []

def main():
    """Main Streamlit application function."""
    st.set_page_config(page_title="Glove/No-Glove Detection", layout="wide")

    st.title("🧤 Hand Protection Detection (YOLOv8)")
    st.markdown("""
        Upload an image below to run the object detection model (`glove_1248_best_v0.pt`). 
        The application will identify **gloved hands** and **bare hands** and provide a compliance summary.
        
        **Note:** This application displays the original image without bounding box annotations, focusing only on the detection metrics.
    """)

    # Load the model
    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Read the uploaded file as a NumPy array (OpenCV format)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_data = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Show a small preview in the sidebar
        st.sidebar.image(image_data, channels="BGR", caption="Original Image", use_column_width=True)
        st.sidebar.markdown("---")

        with st.spinner('Running detection...'):
            # Run inference to get the original image and raw results
            original_image, results = run_inference(model, image_data)

        if original_image:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Image for Analysis")
                # Display the original image (without annotations)
                st.image(original_image, caption="Uploaded Image", use_column_width=True)

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
                        
                        if 0 <= class_id < len(CLASS_NAMES):
                            label = CLASS_NAMES[class_id]
                        
                            if label == 'gloved_hand':
                                gloved_count += 1
                            elif label == 'bare_hand':
                                bare_count += 1


                st.metric(label="Total Hands Detected", value=total_detections)
                st.metric(label="✅ Gloved Hands", value=gloved_count)
                st.metric(label="⚠️ Bare Hands", value=bare_count)
                
                # Compliance check feedback
                if bare_count > 0:
                    st.error("Immediate action required: Bare hands detected (Compliance Failure).")
                elif total_detections > 0:
                    st.success("Compliance Check: All detected hands are gloved (Success).")
                else:
                    st.info("No hands were detected in the image.")
                
                st.markdown(f"---")
                st.caption(f"Confidence Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%")

if __name__ == '__main__':
    main()
