
# Part 2: Reasoning-Based Questions

### Q1: Choosing the Right Approach  
**Task:** Detect whether a product is missing its label on an assembly line (products are visually very similar except for the label).

I would start with **image classification** (binary: label_present vs label_missing). It is dramatically faster to train, cheaper to annotate (one label per image), requires far less compute, and is easier to deploy on edge devices or low-cost cameras. Classification works perfectly when the label position/size is relatively consistent.  

If early experiments show many false positives/negatives due to varying label positions, partial occlusions, or multiple labels per product, I would immediately switch to **object detection** (single class: “label”) and threshold on the number of detections (≥1 = OK, 0 = missing). As a final fallback, if labels contain readable text or barcodes, I would combine detection + lightweight OCR (EasyOCR/Tesseract) to confirm presence.

### Q2: Debugging a Poorly Performing Model  
Model trained on 1,000 clean images but performs badly on new factory images.

My systematic debugging checklist:  
1. Collect and visually inspect 50–100 failure cases from the factory line.  
2. Compute confusion matrix and per-class calibration curves on factory data.  
3. Check for obvious domain shift: lighting, camera angle, background clutter, new product variants, motion blur, or compression artifacts.  
4. Extract embeddings (e.g., from the backbone) and visualize with t-SNE/UMAP — if factory samples form a separate cluster, domain shift is confirmed.  
5. Randomly audit labels on a small factory subset for annotation mistakes.  
6. Quick fixes to try: test-time augmentation (brightness/contrast), fine-tune on just 100–200 labeled factory images, or add a simple domain-adaptation layer (e.g., BatchNorm tuning).  
This process usually identifies the root cause within 1–2 days.

### Q3: Accuracy vs Real Risk  
Model shows 98% accuracy but still misses 1 out of 10 defective products.

Accuracy is almost meaningless here because defects are typically rare (e.g., 1–2% of products). 98% accuracy can easily coexist with only 90% recall on the defective class — which is unacceptable in safety or quality-control settings where missing a defect is 10–100× costlier than a false alarm.  

I would switch to **recall at high precision** (e.g., recall@95%precision or recall@99%precision) or define a custom cost-based metric:  
`score = (cost_of_missed_defect × missed_defects) + (cost_of_false_alarm × false_alarms)`  
PR-AUC or a precision-recall curve is far more informative than overall accuracy in such imbalanced scenarios.

### Q4: Annotation Edge Cases  
Many images contain blurry or partially visible objects during labeling.

I keep most of these cases but follow strict rules:  
- If the object is still recognizable to a human → keep it (blurry images improve motion-blur robustness).  
- If the object is completely unidentifiable → discard.  
- For partially visible objects → annotate tight bounding boxes and (if the tool supports) mark as “truncated” or “occluded”.  

Removing too many hard examples creates a clean-data bias; the model will fail exactly when deployed in the real factory. The trade-off is slightly higher annotation time and minor label noise versus significantly better real-world generalization. In practice, keeping ~80–90% of blurry/partial images yields the best production performance.
