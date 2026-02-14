import json
import os
from pathlib import Path
from urllib.parse import unquote
from PIL import Image
from transformers import LayoutLMv3Processor
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D, DownloadMode
import numpy as np
from collections import defaultdict 

JSON_MIN_PATH = "src/ai_playground/data/export.json"

IMAGE_DIR = "src/ai_playground/data/images"

DATASET_PATH = "src/ai_playground/data/dataset_processed"

BASE_MODEL_ID = "./src/ai_playground/models/base_model" 

LABELS = [
    "O", 
    "B-Section_Header", "I-Section_Header",
    "B-Test_Context_Name", "I-Test_Context_Name",
    "B-Test_Name", "I-Test_Name",
    "B-Test_Value", "I-Test_Value",
    "B-Test_Unit", "I-Test_Unit",
    "B-Test_Norm", "I-Test_Norm",
    "B-Patient_Name", "I-Patient_Name",
    "B-Patient_DOB", "I-Patient_DOB",
    "B-Patient_Weight", "I-Patient_Weight",
    "B-Patient_Height", "I-Patient_Height",
]

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

    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_ID)
    
    total_files = 0
    total_valid_samples = 0

    for item_idx, item in enumerate(data):
        filename = Path(item['image']).name
        image_path = smart_find_file(item['image'], IMAGE_DIR)
        
        if not image_path or not image_path.exists():
            print(f"⚠️  [File {item_idx}] Image not found: {filename}")
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # --- 1. LOAD & SORT USER BOXES (The Fix) ---
        user_boxes = []
        user_labels = []
        
        if 'label' in item:
            # First, extract all raw boxes
            temp_boxes = []
            for annotation in item['label']:
                x = annotation['x'] / 100 * width
                y = annotation['y'] / 100 * height
                w = annotation['width'] / 100 * width
                h = annotation['height'] / 100 * height
                
                # Store Area for sorting
                area = w * h
                box = [x, y, x + w, y + h]
                label_name = annotation['rectanglelabels'][0]
                
                temp_boxes.append({"box": box, "label": label_name, "area": area})
            
            # SORT: Smallest Area First! 
            # This prevents a big "Table" box from stealing the label from a small "Value" box.
            temp_boxes.sort(key=lambda x: x["area"])
            
            for t in temp_boxes:
                user_boxes.append(t["box"])
                user_labels.append(t["label"])

        # --- 2. RUN OCR ---
        try:
            encoding = processor(
                image, 
                return_offsets_mapping=True, 
                truncation=True, 
                max_length=512, 
                padding="max_length", 
                return_tensors="pt"
            )
        except Exception as e:
            print(f"❌ [File {filename}] OCR Failed: {e}")
            continue

        ocr_boxes_1000 = encoding.bbox[0]
        token_labels = []
        
        # LOGGING COUNTER
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
            
            # Check user boxes (Now sorted small -> large)
            for u_box, u_label in zip(user_boxes, user_labels):
                if is_inside((center_x, center_y), u_box):
                    best_label = u_label
                    break 

            # BIO Logic
            final_tag = "O"
            if best_label != "O":
                if i > 0 and token_labels[-1] in [label2id[f"B-{best_label}"], label2id[f"I-{best_label}"]]:
                     final_tag = f"I-{best_label}"
                else:
                     final_tag = f"B-{best_label}"
                
                # Count it for the logs
                stats[best_label] += 1
                
            token_labels.append(label2id[final_tag])

        # --- 3. PRINT REPORT ---
        # Only print if we found meaningful labels to avoid spam
        if sum(stats.values()) > 0:
            print(f"✅ {image_path} | Found: ", end="")
            # Format: Name: 12, Value: 5, Unit: 5
            log_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
            print(log_str, "\n")
        else:
            print(f"⚠️  {image_path} | NO LABELS MATCHED (Check Box Alignment!)")

        # Padding logic...
        if len(token_labels) < 512:
            token_labels += [label2id["O"]] * (512 - len(token_labels))
        else:
            token_labels = token_labels[:512]

        total_valid_samples += 1

        yield {
            "id": filename,
            "input_ids": encoding.input_ids[0].tolist(),
            "attention_mask": encoding.attention_mask[0].tolist(),
            "bbox": ocr_boxes_1000,
            "pixel_values": encoding.pixel_values[0].tolist(),
            "labels": token_labels
        }
    
    print(f"\n📊 Summary: Processed {total_valid_samples} valid images.")


# --- MAIN EXECUTION ---
print("🚀 Parsing Label Studio Data...")

# Create Generator
def gen(json_path=JSON_MIN_PATH):
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

# Build Dataset
ds = Dataset.from_generator(gen, gen_kwargs={"json_path": JSON_MIN_PATH}, features=features)
ds.save_to_disk(DATASET_PATH)

print(f"✅ Dataset processed and saved to {DATASET_PATH}")
print(f"📊 Total Samples: {len(ds)}")