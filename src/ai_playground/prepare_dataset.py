import json
import os
from pathlib import Path
from urllib.parse import unquote
from PIL import Image
from transformers import LayoutLMv3Processor
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import numpy as np

# --- CONFIGURATION ---
JSON_PATH = "src/ai_playground/export.json"
IMAGE_DIR = "src/ai_playground/archive" # Where your actual .png files are
OUTPUT_PATH = "src/ai_playground/dataset_processed"
MODEL_ID = "./src/ai_playground/model_cache" 


# Define your Label Map (MUST MATCH YOUR XML EXACTLY)
# "O" means "Outside" (background text)
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

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def iou(boxA, boxB):
    # Calculate Intersection over Union to find which box a word belongs to
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    return interArea / float(boxAArea + 1e-5)

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

def generate_examples():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # Load processor (handles OCR automatically)
    processor = LayoutLMv3Processor.from_pretrained(MODEL_ID)

    for item in data:
        # 1. Load Image
        # Label studio usually stores filename like "/data/upload/filename.png"
        # We need to extract just the filename
        image_path = smart_find_file(item['image'], IMAGE_DIR)
        
        if not image_path.exists():
            print(f"⚠️ Image not found: {image_path}, skipping...")
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # 2. Get User Labels (The Ground Truth)
        user_boxes = []
        user_labels = []
        
        if 'label' in item:
            for annotation in item['label']:
                # Label Studio uses 0-100 percentages. Convert to pixels.
                x = annotation['x'] / 100 * width
                y = annotation['y'] / 100 * height
                w = annotation['width'] / 100 * width
                h = annotation['height'] / 100 * height
                
                # Convert to [x1, y1, x2, y2]
                box = [x, y, x + w, y + h]
                label_name = annotation['rectanglelabels'][0]
                
                user_boxes.append(box)
                user_labels.append(label_name)

        # 3. Run OCR to get words (Tokens)
        encoding = processor(image, return_offsets_mapping=True, truncation=True, max_length=512)
        offset_mapping = encoding.offset_mapping
        
        # 4. Align Tokens to User Boxes
        # We need to assign a label to every token found by the OCR
        token_labels = []
        
        # Determine actual boxes of tokens using the processor's OCR results
        # Note: LayoutLM processor returns 'bbox' in [0-1000] scale
        ocr_boxes_1000 = encoding.bbox[0]
        
        for i, ocr_box in enumerate(ocr_boxes_1000):
            # Skip special tokens (CLS, SEP) which have [0,0,0,0] box
            if ocr_box == [0, 0, 0, 0]:
                token_labels.append(label2id["O"])
                continue

            # Convert 1000-scale back to pixel scale for comparison
            ocr_box_pixels = [
                ocr_box[0] * width / 1000,
                ocr_box[1] * height / 1000,
                ocr_box[2] * width / 1000,
                ocr_box[3] * height / 1000,
            ]

            # Find best matching user box
            best_iou = 0
            best_label = "O"
            
            for u_box, u_label in zip(user_boxes, user_labels):
                score = iou(ocr_box_pixels, u_box)
                if score > 0.5: # If word is >50% inside the box
                    best_iou = score
                    best_label = u_label
            
            # BIO Tagging Logic (Beginning / Inside)
            # Simplified: We just use 'B-' for now for everything to be safe, 
            # or map B/I if you want strict sequence logic.
            # Let's simple-map everything to "B-" for the first token of a word, 
            # but since we are iterating tokens, simpler is just Direct Mapping.
            
            final_tag = "O"
            if best_label != "O":
                final_tag = f"B-{best_label}" # Simplified approach
                
            token_labels.append(label2id[final_tag])

        yield {
            "id": filename,
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "bbox": encoding.bbox,
            "pixel_values": encoding.pixel_values,
            "labels": token_labels
        }

# --- MAIN EXECUTION ---
print("🚀 Parsing Label Studio Data...")

# Create Generator
def gen():
    return generate_examples()

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
ds = Dataset.from_generator(gen, features=features)
ds.save_to_disk(OUTPUT_PATH)

print(f"✅ Dataset processed and saved to {OUTPUT_PATH}")
print(f"📊 Total Samples: {len(ds)}")