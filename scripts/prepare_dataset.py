import random
import json
import os
from pathlib import Path
from urllib.parse import unquote
import time
import shutil
from PIL import Image
import torch
from transformers import LayoutLMv3Processor
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import numpy as np
from collections import defaultdict 
from src.extraction.document import MedicalDocument
from src.extraction.converter import DocumentConverter
from src.extraction.ocr import TextExtractor
from src.config import LABELS, BASE_MODEL_PATH, JSON_MIN_PATH, IMAGES_PATH, DATASET_PATH, CRITICAL_LABELS

id2label = {k: v for k, v in enumerate(LABELS)}
label2id = {v: k for k, v in enumerate(LABELS)}

DIAGNOSTICS = {
    "total_user_boxes": 0,
    "matched_boxes": 0,
    "missed_boxes": 0,
    "missed_labels_breakdown": defaultdict(int)
}

def smart_find_file(json_image_path, local_image_dir):
    clean_name = unquote(Path(json_image_path).name) 
    local_files = [f.name for f in Path(local_image_dir).iterdir()]
    for local_name in local_files:
        if clean_name.endswith(local_name):
            return Path(local_image_dir) / local_name
    return None

def is_inside(center, box):
    x, y = center
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]

def generate_examples(json_path=JSON_MIN_PATH):
    print(f"📂 Loading annotations from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_PATH, apply_ocr=False)
    extractor = TextExtractor()
    
    total_valid_samples = 0

    for item_idx, item in enumerate(data):
        filename = Path(item['image']).name
        image_path = smart_find_file(item['image'], IMAGES_PATH)
        
        if not image_path or not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # 1. LOAD USER BOXES
        user_boxes = []
        user_labels = []
        if 'label' in item:
            for annotation in item['label']:
                x = annotation['x'] / 100 * width
                y = annotation['y'] / 100 * height
                w = annotation['width'] / 100 * width
                h = annotation['height'] / 100 * height
                user_boxes.append([x, y, x + w, y + h])
                user_labels.append(annotation['rectanglelabels'][0])

        DIAGNOSTICS["total_user_boxes"] += len(user_boxes)
        
        # Track which user boxes get matched in this specific document
        matched_user_indices = set()

        # 2. RUN OCR
        doc = MedicalDocument(str(image_path))
        DocumentConverter.convert_to_images(doc)
        extractor.extract(doc)
        
        if not doc.extracted_data or not doc.extracted_data[0]:
            print(f"⚠️ [File {filename}] OCR found NO text.")
            DIAGNOSTICS["missed_boxes"] += len(user_boxes)
            for lbl in user_labels:
                DIAGNOSTICS["missed_labels_breakdown"][lbl] += 1
            continue
            
        tokens = doc.extracted_data[0]
        words = [t['text'] for t in tokens]
        boxes = [t['bbox'] for t in tokens]

        # 3. PROCESS
        try:
            encoding = processor(
                image, words, boxes=boxes, truncation=True, 
                max_length=512, padding="max_length", return_tensors="pt",
                return_overflowing_tokens=True, stride=128 
            )
        except Exception as e:
            print(f"❌ [File {filename}] Processor Failed: {e}")
            continue

        num_chunks = len(encoding.input_ids)
        
        pv = encoding.pixel_values
        if isinstance(pv, list):
            if len(pv) > 0 and isinstance(pv[0], torch.Tensor): pv = torch.stack(pv)
            else: pv = torch.tensor(pv)
        if pv.dim() == 3: pv = pv.unsqueeze(0)
        if pv.shape[0] == 1 and num_chunks > 1: pv = pv.repeat(num_chunks, 1, 1, 1)

        for chunk_idx in range(num_chunks):
            ocr_boxes_1000 = encoding.bbox[chunk_idx]
            token_labels = []
            
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
                for u_idx, (u_box, u_label) in enumerate(zip(user_boxes, user_labels)):
                    if is_inside((center_x, center_y), u_box):
                        best_label = u_label
                        matched_user_indices.add(u_idx) # Mark as successfully found!
                        break 

                final_tag = "O"
                if best_label != "O":
                    final_tag = f"B-{best_label}" # Simplified BIO for training stability
                    
                token_labels.append(label2id[final_tag])

            chunk_label_names = [id2label[l] for l in token_labels]
            has_critical = any(l in CRITICAL_LABELS for l in chunk_label_names)
            has_any_entities = any(l != "O" for l in chunk_label_names)

            if has_critical or has_any_entities or random.random() <= 0.1:
                total_valid_samples += 1
                yield {
                    "id": f"{filename}_{chunk_idx}",
                    "input_ids": encoding.input_ids[chunk_idx].tolist(),
                    "attention_mask": encoding.attention_mask[chunk_idx].tolist(),
                    "bbox": encoding.bbox[chunk_idx].tolist(),
                    "pixel_values": pv[chunk_idx].tolist(), 
                    "labels": token_labels
                }

        # --- 🔍 FILE-LEVEL DIAGNOSTICS LOGGING ---
        missed_this_file = []
        for u_idx, u_label in enumerate(user_labels):
            if u_idx not in matched_user_indices:
                missed_this_file.append(u_label)
                DIAGNOSTICS["missed_labels_breakdown"][u_label] += 1
        
        DIAGNOSTICS["matched_boxes"] += len(matched_user_indices)
        DIAGNOSTICS["missed_boxes"] += len(missed_this_file)

        if missed_this_file:
            print(f"   ⚠️ {image_path.name} | OCR missed your annotations for: {missed_this_file}")
        else:
            print(f"   ✅ {image_path.name} | Perfect match. All {len(user_labels)} annotations found.")


# --- MAIN EXECUTION ---
print("🚀 Parsing Label Studio Data with OOP Engine...")

if os.path.exists(DATASET_PATH): shutil.rmtree(DATASET_PATH)

features = Features({
    "id": Value("string"),
    "input_ids": Sequence(Value("int64")),
    "attention_mask": Sequence(Value("int64")),
    "bbox": Array2D(dtype="int64", shape=(512, 4)),
    "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
    "labels": Sequence(ClassLabel(names=LABELS))
})

ds = Dataset.from_generator(
    generate_examples, 
    gen_kwargs={"json_path": JSON_MIN_PATH}, 
    features=features
)

ds.save_to_disk(DATASET_PATH)

print("\n" + "="*50)
print("📊 DATASET PREPARATION DIAGNOSTICS REPORT")
print("="*50)
print(f"Total Annotations Drawn by You : {DIAGNOSTICS['total_user_boxes']}")
print(f"Successfully Matched to OCR    : {DIAGNOSTICS['matched_boxes']} ({(DIAGNOSTICS['matched_boxes']/max(1, DIAGNOSTICS['total_user_boxes']))*100:.1f}%)")
print(f"Lost/Missed by OCR             : {DIAGNOSTICS['missed_boxes']} ({(DIAGNOSTICS['missed_boxes']/max(1, DIAGNOSTICS['total_user_boxes']))*100:.1f}%)")

if DIAGNOSTICS["missed_labels_breakdown"]:
    print("\n⚠️ Breakdown of Missed Labels (OCR found NO text inside these boxes):")
    for label, count in sorted(DIAGNOSTICS["missed_labels_breakdown"].items(), key=lambda x: x[1], reverse=True):
        print(f"   - {label}: {count} missed")
print("="*50 + "\n")