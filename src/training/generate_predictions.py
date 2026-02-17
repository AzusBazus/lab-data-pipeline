import json
import os
import shutil
import random
import torch
import uuid
from pathlib import Path
import urllib.parse
import unicodedata
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from src.config import  JSON_MIN_PATH, IMAGES_PATH, MODEL_PATH, BASE_MODEL_PATH, MODEL_VERSION

BATCH_SIZE = 20
OUTPUT_DIR = "./data/batch_upload"

def get_completed_filenames(json_path):
    """Reads Label Studio export and correctly decodes filenames."""
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: {json_path} not found. Assuming 0 images done.")
        return set()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    done_files = set()
    for item in data:
        raw_path = item.get('image') or item.get('data', {}).get('image')
        
        if not raw_path: continue
        
        decoded_path = urllib.parse.unquote(raw_path)
        
        full_name = Path(decoded_path).name
        
        norm_name = unicodedata.normalize('NFC', full_name)

        done_files.add(norm_name)
    
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
    Merges tokens based on BIO tags AND geometric proximity.
    """
    merged_results = []
    if not boxes: return merged_results

    curr_box = None
    curr_label = None
    curr_scores = []

    # Threshold: If next token is >15 pixels below current box, break it.
    Y_BREAK_THRESHOLD = 15.0 

    for box, label, score in zip(boxes, labels, scores):
        if label == "O":
            if curr_box:
                avg_score = sum(curr_scores) / len(curr_scores)
                merged_results.append((curr_box, curr_label, avg_score))
                curr_box = None
            continue

        prefix = label[0] # "B" or "I"
        core_label = label[2:] if len(label) > 2 else label

        # Calculate gap
        is_vertical_break = False
        if curr_box:
            # gap = current_top - previous_bottom
            gap = box[1] - curr_box[3] 
            if gap > Y_BREAK_THRESHOLD:
                is_vertical_break = True

        if (curr_box is None or 
            core_label != curr_label or 
            prefix == "B" or 
            is_vertical_break):
            
            # Close previous
            if curr_box:
                avg_score = sum(curr_scores) / len(curr_scores)
                merged_results.append((curr_box, curr_label, avg_score))
            
            # Start new
            curr_box = list(box)
            curr_label = core_label
            curr_scores = [score]
            
        elif prefix == "I" and core_label == curr_label:
            # MERGE
            curr_box[0] = min(curr_box[0], box[0])
            curr_box[1] = min(curr_box[1], box[1])
            curr_box[2] = max(curr_box[2], box[2])
            curr_box[3] = max(curr_box[3], box[3])
            curr_scores.append(score)

    # Close final
    if curr_box:
        avg_score = sum(curr_scores) / len(curr_scores)
        merged_results.append((curr_box, curr_label, avg_score))

    return merged_results

def main():
    # 1. clean previous batch
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    completed_files = get_completed_filenames(JSON_MIN_PATH)
    all_files = [f for f in os.listdir(IMAGES_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    todo_files = []
    for f in all_files:
        # Normalize local filename too
        f_norm = unicodedata.normalize('NFC', f)
        
        is_done = False
        for done_name in completed_files:
            # Check if the Label Studio name ENDS with our local filename
            # Ex: "uuid-image.png".endswith("image.png") -> True
            if done_name.endswith(f_norm):
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
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH + "/final")
    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_PATH)
    id2label = model.config.id2label

    ls_tasks = []

    for filename in batch_files:
        src_path = os.path.join(IMAGES_PATH, filename)
        dst_path = os.path.join(OUTPUT_DIR, filename)
        
        # Copy image to upload folder
        shutil.copy(src_path, dst_path)
        
        image = Image.open(src_path).convert("RGB")
        inputs = processor(image, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
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

        # 2. Merge Tokens into Entities (e.g. "Test" + "Name" -> "Test Name")
        merge_detections = merge_boxes_bio(pixel_boxes, labels, probs) 
        
        results = []
        # Basic loop to create results (Replace with your merge logic)
        for box, label, score in merge_detections:
            if score < 0.40 or label == "O": 
                continue
            
            x1, y1, x2, y2 = box
    
            # 3. CONVERT TO PERCENTAGES (0-100)
            # Formula: (Pixel / Total_Pixels) * 100
            x = (x1 / width) * 100
            y = (y1 / height) * 100
            w = ((x2 - x1) / width) * 100
            h = ((y2 - y1) / height) * 100

            # Add this print debugging line
            print(f"Label: {label} | Pixels: {x1:.0f} of {width} | Percent: {x:.2f}%")
            
            results.append({
                "id": str(uuid.uuid4())[:8],
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x,
                    "y": y, 
                    "width": w, 
                    "height": h, 
                    "rotation": 0,
                    "rectanglelabels": [label]
                },
                "score": float(score) # Ensure JSON serializable
            })

        # Add to JSON
        ls_tasks.append({
            "data": { "image": filename }, 
            "predictions": [{
                "model_version": f"{MODEL_VERSION}",
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