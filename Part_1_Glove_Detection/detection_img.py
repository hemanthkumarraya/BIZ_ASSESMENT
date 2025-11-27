import os
import json
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---

# 1. Path to your trained YOLOv8 model weights
# Using the model name directly requires it to be in the same folder as this script.
MODEL_PATH = 'glove_1248_best_v0.pt'

# 2. Input folder (where you place your test images)
INPUT_FOLDER = 'input'

# 3. Output folders (Matching the required directory structure)
OUTPUT_IMAGES_FOLDER = 'output' # Annotated images will be saved directly here
OUTPUT_LOGS_FOLDER = 'logs'    # JSON logs will be saved directly here


# --- CLASS NAME MAPPING ---
# Maps the internal model label (GLOVE/NO_GLOVE) to the desired final output label (gloved_hand/bare_hand).
CLASS_MAPPING = {
    "GLOVE": "gloved_hand",
    "NO_GLOVE": "bare_hand"
}

# --- Helper Functions ---

def setup_directories():
    """Creates the necessary input and output folders if they don't exist."""
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_LOGS_FOLDER, exist_ok=True)
    print(f"✅ Directory structure ensured:")
    print(f"  Input Images: {INPUT_FOLDER}")
    print(f"  Output Images: {OUTPUT_IMAGES_FOLDER}")
    print(f"  Output Logs: {OUTPUT_LOGS_FOLDER}")

def process_batch_detection(model, input_folder, output_images_folder, output_logs_folder):
    """
    Runs YOLOv8 detection, applies custom label mapping, saves annotated images, 
    and logs detections to a JSON file per image.
    """
    if not os.path.isdir(input_folder) or not any(os.listdir(input_folder)):
        print(f"❌ Error: Input folder is empty or not found at {input_folder}. Please add images.")
        return

    print(f"\n🚀 Starting batch detection on images in: {input_folder}")

    # 1. Run Detection 
    # Not specifying 'device' allows auto-selection (GPU if available, otherwise CPU).
    results = model.predict(
        source=input_folder, 
        conf=0.25, # Confidence threshold
        iou=0.7,   # NMS threshold
        save=False,
        verbose=False 
    )

    print("\n📝 Processing results and generating JSON logs...")
    
    processed_count = 0
    
    # 2. Iterate through Results (one result object per image)
    for result in results:
        
        if not result.path:
            continue
            
        original_filename = os.path.basename(result.path)
        base_name, ext = os.path.splitext(original_filename)
        
        # 2b. Prepare data for JSON log and frame annotation
        log_data = {
            "filename": original_filename,
            "detections": []
        }
        
        boxes = result.boxes
        annotated_frame = result.orig_img.copy() 
        
        for box in boxes:
            # --- Extract Data ---
            x1, y1, x2, y2 = box.xyxy[0].tolist() 
            confidence = box.conf[0].item() 
            class_id = int(box.cls[0].item())
            
            # Get the internal model label (e.g., "GLOVE")
            internal_label = model.names[class_id]
            
            # --- Map to Desired Output Label (e.g., "gloved_hand") ---
            output_label = CLASS_MAPPING.get(internal_label, internal_label) 

            # --- JSON Log Entry ---
            detection_entry = {
                "label": output_label,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            }
            log_data["detections"].append(detection_entry)

            # --- Frame Annotation (Using Custom Label) ---
            if confidence > 0.25:
                # Define color based on label for visual distinction
                color = (0, 255, 0) if output_label == 'gloved_hand' else (0, 165, 255) # Green vs Orange
                
                # Draw the bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Create label text: "gloved_hand 0.92"
                label_text = f"{output_label} {confidence:.2f}"
                
                # Calculate text size and draw background rectangle for label
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (int(x1), int(y1) - text_height - 10), (int(x1) + text_width, int(y1)), color, -1)
                
                # Put text (Black text on colored background)
                cv2.putText(annotated_frame, label_text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


        # 2c. Save the JSON log file to the 'logs/' folder
        json_filename = base_name + '.json'
        json_path = os.path.join(output_logs_folder, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=4)
            
        # 2d. Save the Annotated Image to the 'output/' folder
        annotated_dest_path = os.path.join(output_images_folder, original_filename)
        cv2.imwrite(annotated_dest_path, annotated_frame)
        
        processed_count += 1
        
    print(f"\n🎉 Batch detection complete. {processed_count} images processed.")
    print(f"Annotated images saved to: {output_images_folder}")
    print(f"JSON logs saved to: {output_logs_folder}")
    
# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    
    # 1. Setup
    setup_directories()
    
    # 2. Load the Model
    try:
        model = YOLO(MODEL_PATH)
        print("✅ YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"❌ Error: Failed to load model from {MODEL_PATH}. Ensure the path is correct.")
        print(f"Details: {e}")
        exit()
        
    # 3. Run the Pipeline

    process_batch_detection(model, INPUT_FOLDER, OUTPUT_IMAGES_FOLDER, OUTPUT_LOGS_FOLDER)
