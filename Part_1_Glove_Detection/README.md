
# Gloved vs Bare Hand Detection – Biz-Tech Analytics Technical Assessment

## Problem Statement
Build a robust object detection system capable of distinguishing **gloved_hand** from **bare_hand** in real-world factory environments for safety compliance monitoring.

## Solution Overview
- **Model:** YOLOv8n (nano) – custom fine-tuned  
- **Framework:** Ultralytics (PyTorch)  
- **Input resolution:** 1248 × 1248 (training & inference)  
- **Final validation performance:** **mAP@0.5 = 0.980**, **mAP@0.5:0.95 = 0.820**  
- Two inference scripts provided: batch image processing (with CLI) + real-time RTSP/webcam streaming

## 1. Dataset Name & Source
- **Custom-collected dataset** (503 high-quality images)  
- Recorded in multiple real-world scenarios: different lighting conditions, skin tones, glove colors/types (nitrile, leather, fabric), backgrounds, and hand poses  
- Annotated using Roboflow (bounding boxes + two classes: `GLOVE`, `NO_GLOVE`)  
- Split: 402 train → 101 validation  
- Heavy online augmentations enabled by default in YOLOv8 (Mosaic, MixUp, HSV, Flip, Rotation, Scale 0.5–2.0), effectively increasing diversity 10–20× and preventing overfitting despite modest raw image count

## 2. Model & Training Details
| Hyperparameter          | Value                          |
|-------------------------|--------------------------------|
| Model                   | YOLOv8n (3.2M params)          |
| Pretrained weights      | COCO (yolov8n.pt)              |
| Epochs                  | 250                            |
| Optimizer               | AdamW (initial lr = 0.001)     |
| Scheduler               | Cosine Annealing               |
| Image size              | 1248 × 1248                    |
| Batch size              | 16 (auto-scaled)               |
| Hardware                | NVIDIA Tesla T4 (Google Colab / Kaggle) |

**Final Validation Metrics (epoch 250)**
| Metric               | Value   |
|----------------------|---------|
| Precision            | 0.952   |
| Recall               | 0.941   |
| mAP@0.5              | **0.980** |
| mAP@0.5:0.95         | **0.820** |

## 3. What Worked Well
- Extremely fast convergence (<50 epochs to >0.95 mAP@0.5)
- Excellent generalization on unseen factory-like images
- Tight, accurate bounding boxes even with partial occlusions and motion blur
- Real-time capable (>80 FPS on Tesla T4, >30 FPS on CPU)

## 4. Limitations & Future Improvements
- Skin-colored gloves remain the hardest edge case (rare but possible confusion)
- Very low-light or heavy motion blur can drop confidence (can be mitigated with test-time brightness augmentation)
- Adding 2–3k more images + hard-negative mining would push mAP50-95 > 0.88

## 5. How to Run

### Environment Setup
```bash
pip install ultralytics opencv-python

Part_1_Glove_Detection/
├── detection_script_img.py      # Batch processing (CLI)
├── detection_script_live.py     # Real-time RTSP/webcam
├── glove_1248_best_v0.pt        # Trained weights
├── input_images/                # Put your test .jpgs here
├── output/                      # Annotated images appear here
├── logs/                        # JSON logs per image
└── README.md/                        


# Basic usage (uses defaults)
python detection_script_img.py

# Full control
python detection_script_img.py \
  --input input_images \
  --output output \
  --logs logs \
  --model glove_1248_best_v0.pt \
  --conf 0.35


python detection_script_live.py

```
