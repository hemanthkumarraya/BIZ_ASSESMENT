import cv2
import time
import os
from ultralytics import YOLO

# --- CONFIGURATION ---

# 1. Path to your trained YOLOv8 model weights
MODEL_PATH = r"glove_1248_best_v1.pt"

# 2. Define your RTSP Stream Link or Webcam Index (0)
RTSP_URL = 'rtsp://admin:Dats%24121@192.168.1.70:554/video/live?channel=0&subtype=0' 

# 3. Define the folder to save images when a detection is triggered
OUTPUT_SAVE_DIR = 'output/triggered_images'

# 4. Custom label mapping for display (for consistency)
CLASS_MAPPING = {
    0: 'gloved_hand',   # Index 0 from your model output
    1: 'bare_hand'      # Index 1 from your model output
}

CONFIDENCE_THRESHOLD = 0.50

# --- Setup ---
os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)
print(f"✅ Output directory created for triggered images: {OUTPUT_SAVE_DIR}")

# --- Main Detection Function ---

def run_live_detection(rtsp_url, model_path, class_map, conf_thres, output_dir):
    """
    Loads the YOLOv8 model and runs live inference on a stream, saving frames
    when any detection is present.
    """
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"🎥 Connecting to stream: {rtsp_url}. Press 'q' on the video window to quit.")

    # ADDED: Test Connection Separately (OpenCV method)
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("❌ ERROR: Failed to open the RTSP stream. Check the URL, network connection, and camera credentials.")
        return
    cap.release()
    print("✅ RTSP stream connection test successful (pre-inference).")

    results_generator = model.predict(
        source=rtsp_url, 
        conf=conf_thres, 
        stream=True, 
        show=False, 
        # CRITICAL FIX: Use 'cpu' instead of 'device=0'
        device='cpu' 
    )

    frame_count = 0
    start_time = time.time()
    
    for result in results_generator:
        frame = result.orig_img
        frame_count += 1
        
        boxes_data = result.boxes.data
        # If there is at least one detection, save the frame
        detection_is_present = len(boxes_data) > 0
        
        # --- Frame Saving Logic ---
        if detection_is_present:
            # Create a unique filename using timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f"detected_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"  🚨 Hand detected! Saved frame to: {save_path}")
        
        # --- Custom Label Drawing ---
        for *box, conf, cls in boxes_data:
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            confidence = float(conf)
            
            label = class_map.get(class_id, f'Class {class_id}')
            display_text = f"{label} {confidence:.2f}"
            
            # Draw box and text
            color = (0, 255, 0) if label == 'gloved_hand' else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # --- Display the live feed with FPS ---
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Live YOLOv8 Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n👋 Live detection stopped.")

if __name__ == "__main__":
    run_live_detection(RTSP_URL, MODEL_PATH, CLASS_MAPPING, CONFIDENCE_THRESHOLD, OUTPUT_SAVE_DIR)