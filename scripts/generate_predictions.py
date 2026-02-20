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
    # 1. Clean previous batch
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    completed_files = get_completed_filenames(JSON_MIN_PATH)
    all_files = [f for f in os.listdir(IMAGES_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 2. Filter ToDo List
    todo_files = []
    for f in all_files:
        f_norm = unicodedata.normalize('NFC', f)
        is_done = False
        for done_name in completed_files:
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
    current_batch_size = min(BATCH_SIZE, len(todo_files))
    batch_files = random.sample(todo_files, current_batch_size)
    
    print(f"🚀 Preparing Batch of {current_batch_size} images...")
    
    # 4. Load Model
    model = LayoutLMv3ForTokenClassification.from_pretrained(CUSTOM_MODEL_PATH)
    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_PATH)
    id2label = model.config.id2label

    ls_tasks = []

    for filename in batch_files:
        src_path = os.path.join(IMAGES_PATH, filename)
        dst_path = os.path.join(OUTPUT_DIR, filename)
        
        # Copy image
        shutil.copy(src_path, dst_path)
        
        image = Image.open(src_path).convert("RGB")
        width, height = image.size

        inputs = processor(
            image, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding="max_length",
            return_overflowing_tokens=True, # Enable chunking
            stride=128                      # Enable overlap
        )

        pixel_values = inputs['pixel_values']
        
        # 1. Convert List to Tensor if needed
        if isinstance(pixel_values, list):
            if len(pixel_values) > 0 and isinstance(pixel_values[0], torch.Tensor):
                pixel_values = torch.stack(pixel_values) # Stack list of tensors
            else:
                pixel_values = torch.tensor(pixel_values) # Convert list of floats
        
        # 2. Ensure Batch Dimension (1, 3, 224, 224)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
            
        # 3. Repeat Image to match Number of Text Chunks
        num_chunks = inputs['input_ids'].shape[0]
        if pixel_values.shape[0] != num_chunks:
            # Example: Expand (1, 3, 224, 224) -> (5, 3, 224, 224)
            pixel_values = pixel_values.repeat(num_chunks, 1, 1, 1)
            
        # 4. Update Inputs
        inputs['pixel_values'] = pixel_values
        
        # "inputs" now contains [num_chunks, 512, ...]
        # The model handles this as a standard "batch" automatically
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract predictions for ALL chunks
        # Shape: [num_chunks, 512]
        chunk_predictions = outputs.logits.argmax(-1)
        chunk_probs = outputs.logits.softmax(-1).max(-1).values
        chunk_boxes = inputs.bbox # 1000-scale boxes

        # --- 🧩 STITCHING LOGIC ---
        final_pixel_boxes = []
        final_labels = []
        final_probs = []
        
        # We use a Set to prevent duplicates in the overlap regions
        # (Tokens 384-512 appear in both chunks; we only want them once)
        seen_boxes = set()

        num_chunks = len(chunk_predictions)
        
        for i in range(num_chunks):
            # Extract lists for this specific chunk
            preds = chunk_predictions[i].tolist()
            probs = chunk_probs[i].tolist()
            boxes = chunk_boxes[i].tolist()

            for k, (pred_id, prob, box) in enumerate(zip(preds, probs, boxes)):
                # Skip Padding tokens (0,0,0,0)
                if box == [0, 0, 0, 0]:
                    continue
                
                # Deduplication Check
                # Box coordinates are absolute (0-1000), so duplicates are identical
                box_tuple = tuple(box)
                if box_tuple in seen_boxes:
                    continue
                seen_boxes.add(box_tuple)
                
                # Add valid token to final list
                final_pixel_boxes.append([
                    box[0] * width / 1000,
                    box[1] * height / 1000,
                    box[2] * width / 1000,
                    box[3] * height / 1000
                ])
                final_labels.append(id2label[pred_id])
                final_probs.append(prob)

        # 5. Merge (Now using the full page of tokens)
        merge_detections = merge_boxes_bio(final_pixel_boxes, final_labels, final_probs) 
        
        results = []
        for box, label, score in merge_detections:
            if score < 0.40 or label == "O": 
                continue
            
            x1, y1, x2, y2 = box
    
            # Convert to Percentages
            x = (x1 / width) * 100
            y = (y1 / height) * 100
            w = ((x2 - x1) / width) * 100
            h = ((y2 - y1) / height) * 100

            results.append({
                "id": str(uuid.uuid4())[:8],
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x, "y": y, "width": w, "height": h, 
                    "rotation": 0,
                    "rectanglelabels": [label]
                },
                "score": float(score)
            })

        # Add to JSON
        ls_tasks.append({
            "data": { "image": filename }, 
            "predictions": [{
                "model_version": f"{MODEL_VERSION}",
                "result": results
            }]
        })

    # Save JSON
    json_output_path = os.path.join(OUTPUT_DIR, "predictions.json")
    with open(json_output_path, "w") as f:
        json.dump(ls_tasks, f, indent=2)

    print(f"✅ Batch ready in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()