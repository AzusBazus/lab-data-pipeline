import json
import os
import shutil
import random
import torch
import uuid
from pathlib import Path
import urllib.parse
import unicodedata

from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

# --- IMPORT YOUR OOP PIPELINE ---
from src.extraction.document import MedicalDocument
from src.extraction.converter import DocumentConverter
from src.extraction.ocr import TextExtractor
from src.config import CUSTOM_MODEL_PATH, BASE_MODEL_PATH, JSON_MIN_PATH, IMAGES_PATH, MODEL_VERSION

BATCH_SIZE = 10
OUTPUT_DIR = "./data/batch_upload"

def get_completed_filenames(json_path):
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

def merge_boxes_bio(boxes, labels, scores):
    """Merges tokens based on BIO tags AND geometric proximity."""
    merged_results = []
    if not boxes: return merged_results

    curr_box = None
    curr_label = None
    curr_scores = []
    Y_BREAK_THRESHOLD = 15.0 

    for box, label, score in zip(boxes, labels, scores):
        if label == "O":
            if curr_box:
                avg_score = sum(curr_scores) / len(curr_scores)
                merged_results.append((curr_box, curr_label, avg_score))
                curr_box = None
            continue

        prefix = label[0] 
        core_label = label[2:] if len(label) > 2 else label

        is_vertical_break = False
        if curr_box:
            gap = box[1] - curr_box[3] 
            if gap > Y_BREAK_THRESHOLD:
                is_vertical_break = True

        if (curr_box is None or core_label != curr_label or prefix == "B" or is_vertical_break):
            if curr_box:
                avg_score = sum(curr_scores) / len(curr_scores)
                merged_results.append((curr_box, curr_label, avg_score))
            
            curr_box = list(box)
            curr_label = core_label
            curr_scores = [score]
            
        elif prefix == "I" and core_label == curr_label:
            curr_box[0] = min(curr_box[0], box[0])
            curr_box[1] = min(curr_box[1], box[1])
            curr_box[2] = max(curr_box[2], box[2])
            curr_box[3] = max(curr_box[3], box[3])
            curr_scores.append(score)

    if curr_box:
        avg_score = sum(curr_scores) / len(curr_scores)
        merged_results.append((curr_box, curr_label, avg_score))

    return merged_results

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    completed_files = get_completed_filenames(JSON_MIN_PATH)
    all_files = [f for f in os.listdir(IMAGES_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    todo_files = []
    for f in all_files:
        f_norm = unicodedata.normalize('NFC', f)
        is_done = any(done_name.endswith(f_norm) for done_name in completed_files)
        if not is_done:
            todo_files.append(f)

    print(f"📊 Status: {len(completed_files)} Done | {len(todo_files)} Remaining")
    
    if not todo_files:
        print("🎉 All images are annotated! No new batch needed.")
        return

    current_batch_size = min(BATCH_SIZE, len(todo_files))
    batch_files = random.sample(todo_files, current_batch_size)
    
    print(f"🚀 Preparing Batch of {current_batch_size} images...")
    
    # --- INITIALIZE OOP PIPELINE & MODEL ---
    print("⏳ Loading Models & Extractors...")
    extractor = TextExtractor()
    model = LayoutLMv3ForTokenClassification.from_pretrained(CUSTOM_MODEL_PATH)
    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_PATH, apply_ocr=False) # MUST BE FALSE
    id2label = model.config.id2label
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    ls_tasks = []

    for filename in batch_files:
        src_path = os.path.join(IMAGES_PATH, filename)
        dst_path = os.path.join(OUTPUT_DIR, filename)
        shutil.copy(src_path, dst_path)
        
        print(f"Processing: {filename}")

        # --- 1. OOP EXTRACTION ---
        doc = MedicalDocument(src_path)
        DocumentConverter.convert_to_images(doc)
        extractor.extract(doc)
        
        if not doc.extracted_data or not doc.extracted_data[0]:
            print(f"  ⚠️ No text found by OCR.")
            continue
            
        image = doc.pages[0]
        width, height = image.size
        tokens = doc.extracted_data[0]
        words = [t['text'] for t in tokens]
        boxes = [t['bbox'] for t in tokens]

        # --- 2. PREDICTION (With Sliding Window) ---
        encoding = processor(
            image, 
            words, 
            boxes=boxes, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length",
            max_length=512,
            return_overflowing_tokens=True,
            stride=128
        )

        num_chunks = encoding['input_ids'].shape[0]
        chunk_word_ids = [encoding.word_ids(batch_index=i) for i in range(num_chunks)]

        pixel_values = encoding['pixel_values']
        if isinstance(pixel_values, list):
            if len(pixel_values) > 0 and isinstance(pixel_values[0], torch.Tensor):
                pixel_values = torch.stack(pixel_values)
            else:
                pixel_values = torch.tensor(pixel_values)
        if pixel_values.dim() == 3: pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.shape[0] == 1 and num_chunks > 1:
            pixel_values = pixel_values.repeat(num_chunks, 1, 1, 1)

        encoding['pixel_values'] = pixel_values
        encoding.pop("overflow_to_sample_mapping", None)

        encoding_on_device = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding_on_device)

        chunk_preds = outputs.logits.argmax(-1)
        chunk_probs = torch.softmax(outputs.logits, dim=-1).max(-1).values

        # --- 3. MERGE OVERLAPPING CHUNKS (Max Confidence) ---
        best_predictions = {} 
        for chunk_idx in range(num_chunks):
            word_ids = chunk_word_ids[chunk_idx]
            for seq_idx, word_idx in enumerate(word_ids):
                if word_idx is None: continue 

                label_id = chunk_preds[chunk_idx][seq_idx].item()
                label_name = id2label[label_id]
                confidence = chunk_probs[chunk_idx][seq_idx].item()

                if word_idx in best_predictions:
                    if confidence > best_predictions[word_idx][1]:
                        best_predictions[word_idx] = (label_name, confidence)
                else:
                    best_predictions[word_idx] = (label_name, confidence)

        # --- 4. PREPARE RESULTS FOR LABEL STUDIO ---
        final_pixel_boxes = []
        final_labels = []
        final_probs = []

        # We iterate through the original tokens we got from EasyOCR
        for word_idx, token in enumerate(tokens):
            if word_idx not in best_predictions: continue
            
            label, conf = best_predictions[word_idx]
            
            # Convert 0-1000 scale back to actual pixel scale
            b = token['bbox']
            final_pixel_boxes.append([
                b[0] * width / 1000,
                b[1] * height / 1000,
                b[2] * width / 1000,
                b[3] * height / 1000
            ])
            final_labels.append(label)
            final_probs.append(conf)

        # Merge BIO tags into solid boxes
        merge_detections = merge_boxes_bio(final_pixel_boxes, final_labels, final_probs) 
        
        results = []
        for box, label, score in merge_detections:
            if score < 0.40 or label == "O": continue
            
            x1, y1, x2, y2 = box
            results.append({
                "id": str(uuid.uuid4())[:8],
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": (x1 / width) * 100, 
                    "y": (y1 / height) * 100, 
                    "width": ((x2 - x1) / width) * 100, 
                    "height": ((y2 - y1) / height) * 100, 
                    "rotation": 0,
                    "rectanglelabels": [label]
                },
                "score": float(score)
            })

        ls_tasks.append({
            "data": { "image": filename }, 
            "predictions": [{"model_version": MODEL_VERSION, "result": results}]
        })

    json_output_path = os.path.join(OUTPUT_DIR, "predictions.json")
    with open(json_output_path, "w") as f:
        json.dump(ls_tasks, f, indent=2)

    print(f"✅ Batch ready in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()