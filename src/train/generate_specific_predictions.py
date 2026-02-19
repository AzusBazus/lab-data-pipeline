import json
import os
import shutil
import torch
import uuid
import unicodedata
from pathlib import Path
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from src.config import JSON_MIN_PATH, IMAGES_PATH, MODEL_PATH, BASE_MODEL_PATH, PRIORITY_FOLDER
from src.train.generate_predictions import merge_boxes_bio

OUTPUT_DIR = "./data/batch_upload"

def get_completed_filenames(json_path):
    """ Reads Label Studio export to find what we have already done. """
    if not os.path.exists(json_path):
        return set()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    done_files = set()
    for item in data:
        # Decode URL and Normalize
        raw_path = item.get('image') or item.get('data', {}).get('image')
        if not raw_path: continue
        
        decoded_path = unicodedata.normalize('NFC', Path(raw_path).name)
        # We store the *suffix* to match robustly
        # e.g., "82381-my_image.png" -> matches "my_image.png"
        done_files.add(decoded_path)
    
    return done_files

def main():
    # 1. Setup
    if not os.path.exists(PRIORITY_FOLDER):
        print(f"❌ Error: Priority folder '{PRIORITY_FOLDER}' does not exist.")
        print("   -> Create it and drag your 'Edge Case' images there.")
        return

    # Clean output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 2. Check what is already done
    completed_files_set = get_completed_filenames(JSON_MIN_PATH)
    
    # 3. Scan Priority Folder
    priority_files = [f for f in os.listdir(PRIORITY_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    todo_files = []
    
    print(f"🔍 Scanning '{PRIORITY_FOLDER}'...")
    
    for f in priority_files:
        f_norm = unicodedata.normalize('NFC', f)
        
        # Check against completed set
        is_done = False
        for done_name in completed_files_set:
            if done_name.endswith(f_norm): # Robust check
                is_done = True
                break
        
        if is_done:
            print(f"   Skipping (Already Done): {f}")
        else:
            todo_files.append(f)

    if not todo_files:
        print("🎉 No new priority images to process!")
        return

    print(f"🚀 Processing {len(todo_files)} Priority Images...")

    # 4. Load Model
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH + "/final")
    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_PATH)
    id2label = model.config.id2label

    ls_tasks = []

    for filename in todo_files:
        src_path = os.path.join(PRIORITY_FOLDER, filename)
        dst_path = os.path.join(OUTPUT_DIR, filename)
        
        # Copy to upload folder
        shutil.copy(src_path, dst_path)
        
        image = Image.open(src_path).convert("RGB")
        width, height = image.size

        # --- SLIDING WINDOW INFERENCE ---
        inputs = processor(
            image, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding="max_length",
            return_overflowing_tokens=True, 
            stride=128
        )
        
        # Fix Pixel Values (Replicate Image for batches)
        pixel_values = inputs['pixel_values']
        if isinstance(pixel_values, list):
            if len(pixel_values) > 0 and isinstance(pixel_values[0], torch.Tensor):
                pixel_values = torch.stack(pixel_values)
            else:
                pixel_values = torch.tensor(pixel_values)
        
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
            
        num_chunks = inputs['input_ids'].shape[0]
        if pixel_values.shape[0] != num_chunks:
            pixel_values = pixel_values.repeat(num_chunks, 1, 1, 1)
            
        inputs['pixel_values'] = pixel_values

        with torch.no_grad():
            outputs = model(**inputs)
        
        # --- STITCHING LOGIC ---
        chunk_predictions = outputs.logits.argmax(-1)
        chunk_probs = outputs.logits.softmax(-1).max(-1).values
        chunk_boxes = inputs.bbox 

        final_pixel_boxes = []
        final_labels = []
        final_probs = []
        seen_boxes = set()

        for i in range(len(chunk_predictions)):
            preds = chunk_predictions[i].tolist()
            probs = chunk_probs[i].tolist()
            boxes = chunk_boxes[i].tolist()

            for pred_id, prob, box in zip(preds, probs, boxes):
                if box == [0, 0, 0, 0]: continue
                
                box_tuple = tuple(box)
                if box_tuple in seen_boxes: continue
                seen_boxes.add(box_tuple)
                
                final_pixel_boxes.append([
                    box[0] * width / 1000,
                    box[1] * height / 1000,
                    box[2] * width / 1000,
                    box[3] * height / 1000
                ])
                final_labels.append(id2label[pred_id])
                final_probs.append(prob)

        merge_detections = merge_boxes_bio(final_pixel_boxes, final_labels, final_probs) 

        results = []
        for box, label, score in merge_detections:
            if score < 0.40 or label == "O": continue
            x1, y1, x2, y2 = box
            
            results.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": (x1 / width) * 100, "y": (y1 / height) * 100, 
                    "width": ((x2 - x1) / width) * 100, "height": ((y2 - y1) / height) * 100,
                    "rotation": 0, "rectanglelabels": [label]
                },
                "score": float(score)
            })

        ls_tasks.append({
            "data": { "image": filename }, 
            "predictions": [{
                "model_version": "priority_v1",
                "result": results
            }]
        })

    # Save
    with open(os.path.join(OUTPUT_DIR, "predictions.json"), "w") as f:
        json.dump(ls_tasks, f, indent=2)

    print(f"✅ Priority Batch ready in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()