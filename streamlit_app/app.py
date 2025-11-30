import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

# Page config
st.set_page_config(page_title="YOLOv8 Object Detector", layout="centered")

st.title("YOLOv8 Live Demo")
st.caption("Upload an image → get instant object detection (runs 100% on CPU)")

# Force CPU (important for Streamlit Cloud)
torch.cuda.is_available = lambda : False

# Load model once
@st.cache_resource
def load_model():
    # yolov8n = fastest & smallest, perfect for CPU demo
    return YOLO("yolov8n.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Running inference...")

    with st.spinner("Detecting objects..."):
        results = model(image, conf=0.25)[0]  # single image

    # Show results
    annotated = results.plot()  # returns numpy array with boxes & labels
    st.image(annotated, caption="Detection Results", use_column_width=True)

    # Show detected classes & confidence
    if len(results.boxes) > 0:
        st.success(f"Found {len(results.boxes)} object(s)!")
        for box in results.boxes:
            cls = results.names[int(box.cls)]
            conf = box.conf.item()
            st.write(f"• {cls}: {conf:.1%}")
    else:
        st.info("No objects detected.")
