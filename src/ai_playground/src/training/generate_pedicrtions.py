import json
import os
import shutil
import random
import torch
import uuid
from pathlib import Path
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

# --- CONFIGURATION ---
EXPORT_JSON_PATH = "src/ai_playground/data/export.json"  # The latest export from Label Studio
ALL_IMAGES_DIR = "src/ai_playground/data/images"              # Where all 270 images live
MODEL_PATH = "src/ai_playground/models/custom_v2/final"                 # Your best model
BASE_MODEL_PATH = "src/ai_playground/models/base_model"
BATCH_SIZE = 20                                     # How many to label next?
OUTPUT_DIR = "src/ai_playground/data/batch_upload"                         # Temporary folder for drag-and-drop

def get_completed_filenames(json_path):
    """Reads Label Studio export to find what we have already done."""
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: {json_path} not found. Assuming 0 images done.")
        return set()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    done_files = set()
    for item in data:
        # Label Studio stores as "/data/upload/filename.png" -> extract "filename.png"
        # Also handles URL encoded names if needed
        filename = Path(item['image']).name
        # Simple decode if needed, or rely on endswith match later
        done_files.add(filename)
    
    return done_files

def pixel_to_percent(box, width, height):
    """Converts pixel [x1, y1, x2, y2] to Label Studio % [x, y, w, h]"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return {
        "x": (x1 / width) * 100,
        "y": (y1 / height) * 100,
        "width": (w / width) * 100,
        "height": (h / height) * 100,
        "rotation": 0
    }

def merge_boxes_bio(boxes, labels, scores):
    """
    Merges tokens based on BIO tags (B- / I-).
    This prevents 'Row 1' and 'Row 2' from merging even if they are close.
    """
    merged_results = []
    if not boxes: return merged_results

    curr_box = None
    curr_label = None
    curr_scores = []

    for box, label, score in zip(boxes, labels, scores):
        if label == "O":
            # If we hit background, close the current box
            if curr_box:
                avg_score = sum(curr_scores) / len(curr_scores)
                merged_results.append((curr_box, curr_label, avg_score))
                curr_box = None
            continue

        # Split "B-Test_Name" into "B" and "Test_Name"
        prefix = label[0] # "B" or "I"
        core_label = label[2:] # "Test_Name"

        # LOGIC: Start a new box if:
        # 1. We don't have an open box
        # 2. The label type changed (Name -> Value)
        # 3. We hit a "B-" tag (Explicit start of new entity)
        if curr_box is None or core_label != curr_label or prefix == "B":
            
            # Close previous box first
            if curr_box:
                avg_score = sum(curr_scores) / len(curr_scores)
                merged_results.append((curr_box, curr_label, avg_score))
            
            # Start new box
            curr_box = list(box)
            curr_label = core_label
            curr_scores = [score]
            
        elif prefix == "I" and core_label == curr_label:
            # EXTEND current box (Merge)
            curr_box[0] = min(curr_box[0], box[0]) # min x
            curr_box[1] = min(curr_box[1], box[1]) # min y
            curr_box[2] = max(curr_box[2], box[2]) # max x
            curr_box[3] = max(curr_box[3], box[3]) # max y
            curr_scores.append(score)

    # Append the final straggler
    if curr_box:
        avg_score = sum(curr_scores) / len(curr_scores)
        merged_results.append((curr_box, curr_label, avg_score))

    return merged_results

def main():
    # 1. clean previous batch
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 2. Identify "ToDo" list
    completed_files = get_completed_filenames(EXPORT_JSON_PATH)
    all_files = [f for f in os.listdir(ALL_IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Logic: Filter out files that "end with" any string in the completed list
    # (Handles the random UUID prefix Label Studio adds: "abcde-my_image.png")
    todo_files = []
    for f in all_files:
        is_done = False
        for done_name in completed_files:
            if done_name.endswith(f) or f.endswith(done_name):
                is_done = True
                break
        if not is_done:
            todo_files.append(f)

    print(f"📊 Status: {len(completed_files)} Done | {len(todo_files)} Remaining")
    
    if not todo_files:
        print("🎉 All images are annotated! No new batch needed.")
        return

    # 3. Select Batch
    # Select fewer if we are near the end
    current_batch_size = min(BATCH_SIZE, len(todo_files))
    batch_files = random.sample(todo_files, current_batch_size)
    
    print(f"🚀 Preparing Batch of {current_batch_size} images...")
    
    # 4. Load Model for Pre-Labeling
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_PATH)
    id2label = model.config.id2label

    ls_tasks = []

    for filename in batch_files:
        src_path = os.path.join(ALL_IMAGES_DIR, filename)
        dst_path = os.path.join(OUTPUT_DIR, filename)
        
        # Copy image to upload folder
        shutil.copy(src_path, dst_path)
        
        # --- RUN INFERENCE (Simplified for brevity) ---
        image = Image.open(src_path).convert("RGB")
        inputs = processor(image, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # ... (Insert your pixel_to_percent and merge_boxes logic here from previous message) ...
        # For now, let's assume you have the `final_detections` list from the previous script
        
        # Placeholder logic to ensure script runs:
        # You MUST paste the `merge_boxes` and `pixel_to_percent` functions here!
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        probs = outputs.logits.softmax(-1).max(-1).values.squeeze().tolist()
        token_boxes = inputs.bbox.squeeze().tolist()
        width, height = image.size

        pixel_boxes = []
        labels = []
        for box, pred_id in zip(token_boxes, predictions):
            pixel_boxes.append([
                box[0] * width / 1000,
                box[1] * height / 1000,
                box[2] * width / 1000,
                box[3] * height / 1000
            ])
            labels.append(id2label[pred_id])

        # Merge for UI
        final_detections = merge_boxes_bio(pixel_boxes, labels, probs)
        
        results = []
        # Basic loop to create results (Replace with your merge logic)
        for box, label, score in final_detections:
            if score < 0.40: continue
            if label == "O": continue
            
            # Convert [0-1000] to [0-100%]
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            
            results.append({
                "id": str(uuid.uuid4())[:8],
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x/10, "y": y/10, "width": w/10, "height": h/10, # Simple /10 conversion
                    "rectanglelabels": [label]
                },
                "score": score
            })

        # Add to JSON
        ls_tasks.append({
            "data": { "image": filename }, 
            "predictions": [{
                "model_version": "custom_v2",
                "result": results
            }]
        })

    # 5. Save JSON to the batch folder
    json_output_path = os.path.join(OUTPUT_DIR, "predictions.json")
    with open(json_output_path, "w") as f:
        json.dump(ls_tasks, f, indent=2)

    print(f"✅ Batch ready in '{OUTPUT_DIR}'")
    print(f"👉 Action: Drag the contents of '{OUTPUT_DIR}' into Label Studio.")

if __name__ == "__main__":
    main()