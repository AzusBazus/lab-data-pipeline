import random
import json
import os
from pathlib import Path
from urllib.parse import unquote
import time
import shutil
from PIL import Image
from transformers import LayoutLMv3Processor
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import numpy as np
from collections import defaultdict 
from src.config import LABELS, BASE_MODEL_PATH, JSON_MIN_PATH, IMAGES_PATH, DATASET_PATH, CRITICAL_LABELS

id2label = {k: v for k, v in enumerate(LABELS)}
label2id = {v: k for k, v in enumerate(LABELS)}

def smart_find_file(json_image_path, local_image_dir):
    """
    Matches Label Studio's weird filename to your local file.
    1. Decodes URL (turns %D0%A3 into 'У')
    2. Ignores the random UUID prefix (e50fb879-)
    """
    # 1. Decode URL characters (Cyrillic fix)
    clean_name = unquote(Path(json_image_path).name) 
    
    # 2. List all local files
    local_files = [f.name for f in Path(local_image_dir).iterdir()]
    
    # 3. Find the match
    # Logic: The Label Studio name (clean_name) will END with the local filename
    # Example: "e50fb879-1-Усманова.png" ends with "1-Усманова.png"
    for local_name in local_files:
        if clean_name.endswith(local_name):
            return Path(local_image_dir) / local_name
            
    return None

def is_inside(center, box):
    # Check if a point (x,y) is inside a box [x1, y1, x2, y2]
    x, y = center
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]

def generate_examples(json_path=JSON_MIN_PATH):
    print(f"📂 Loading annotations from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_PATH)
    
    total_valid_samples = 0

    for item_idx, item in enumerate(data):
        filename = Path(item['image']).name
        image_path = smart_find_file(item['image'], IMAGES_PATH)
        
        if not image_path or not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # 1. LOAD USER BOXES (Same as before)
        user_boxes = []
        user_labels = []
        if 'label' in item:
            temp_boxes = []
            for annotation in item['label']:
                x = annotation['x'] / 100 * width
                y = annotation['y'] / 100 * height
                w = annotation['width'] / 100 * width
                h = annotation['height'] / 100 * height
                area = w * h
                box = [x, y, x + w, y + h]
                label_name = annotation['rectanglelabels'][0]
                temp_boxes.append({"box": box, "label": label_name, "area": area})
            
            temp_boxes.sort(key=lambda x: x["area"])
            for t in temp_boxes:
                user_boxes.append(t["box"])
                user_labels.append(t["label"])

        # --- 2. RUN OCR WITH SLIDING WINDOW (The Fix) ---
        try:
            # We enable sliding window here
            encoding = processor(
                image, 
                return_offsets_mapping=True, 
                truncation=True, 
                max_length=512, 
                padding="max_length", 
                return_tensors="pt",
                # NEW PARAMETERS:
                return_overflowing_tokens=True, # Return the extra pieces
                stride=128 # Overlap amount (ensures no context is lost at the cut)
            )
        except Exception as e:
            print(f"❌ [File {filename}] OCR Failed: {e}")
            continue

        # encoding.input_ids is now shape [num_chunks, 512]
        num_chunks = len(encoding.input_ids)
        
        # Loop through EACH CHUNK (e.g., Top Half, Bottom Half)
        for chunk_idx in range(num_chunks):
            
            # Extract specific boxes for this chunk
            ocr_boxes_1000 = encoding.bbox[chunk_idx]
            token_labels = []
            stats = defaultdict(int)

            for i, ocr_box in enumerate(ocr_boxes_1000):
                if ocr_box.tolist() == [0, 0, 0, 0]:
                    token_labels.append(label2id["O"])
                    continue

                x1 = ocr_box[0] * width / 1000
                y1 = ocr_box[1] * height / 1000
                x2 = ocr_box[2] * width / 1000
                y2 = ocr_box[3] * height / 1000
                
                center_x = x1 + (x2 - x1) / 2
                center_y = y1 + (y2 - y1) / 2

                best_label = "O"
                for u_box, u_label in zip(user_boxes, user_labels):
                    if is_inside((center_x, center_y), u_box):
                        best_label = u_label
                        break 

                final_tag = "O"
                if best_label != "O":
                    if i > 0 and token_labels[-1] in [label2id[f"B-{best_label}"], label2id[f"I-{best_label}"]]:
                         final_tag = f"I-{best_label}"
                    else:
                         final_tag = f"B-{best_label}"
                    stats[best_label] += 1
                    
                token_labels.append(label2id[final_tag])

            if sum(stats.values()) > 0:
                 # Clean up print to show which part of the page this is
                 part_str = f"(Part {chunk_idx+1}/{num_chunks})"
                 print(f"✅ {image_path} {part_str} | Found: {dict(stats)}")

            chunk_label_names = [id2label[l] for l in token_labels]
            
            # 1. Check for CRITICAL labels (Never skip these)
            has_critical = any(l in CRITICAL_LABELS for l in chunk_label_names)
            
            # 2. Check for ANY entities (for the 10% keep rule)
            has_any_entities = any(l != "O" for l in chunk_label_names)

            if has_critical:
                pass 
            elif not has_any_entities:
                if random.random() > 0.1:
                    continue

            total_valid_samples += 1

            # Yield THIS specific chunk
            yield {
                "id": f"{filename}_{chunk_idx}", # Unique ID for each chunk
                "input_ids": encoding.input_ids[chunk_idx].tolist(),
                "attention_mask": encoding.attention_mask[chunk_idx].tolist(),
                "bbox": encoding.bbox[chunk_idx].tolist(),
                "pixel_values": encoding.pixel_values[0].tolist(), 
                "labels": token_labels
            }


# --- MAIN EXECUTION ---
print("🚀 Parsing Label Studio Data...")

if os.path.exists(DATASET_PATH):
    shutil.rmtree(DATASET_PATH)

def gen(json_path=JSON_MIN_PATH, **kwargs):
    return generate_examples(json_path)

# Define Schema
features = Features({
    "id": Value("string"),
    "input_ids": Sequence(Value("int64")),
    "attention_mask": Sequence(Value("int64")),
    "bbox": Array2D(dtype="int64", shape=(512, 4)),
    "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
    "labels": Sequence(ClassLabel(names=LABELS))
})

ds = Dataset.from_generator(
    gen, 
    gen_kwargs={
        "json_path": JSON_MIN_PATH,
        "anti_cache_timestamp": time.time()
    }, 
    features=features
)

ds.save_to_disk(DATASET_PATH)

print(f"✅ Dataset processed and saved to {DATASET_PATH}")
print(f"📊 Total Samples: {len(ds)}")