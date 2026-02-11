from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

# 1. Load the "Base" Model (Pre-trained by Microsoft)
# Note: This is a generic model. It knows what a "word" is, but maybe not your specific "Antibiotic Result".
# We use 'microsoft/layoutlmv3-base' for the architecture, 
# but for extraction tasks, we usually start with a fine-tuned version like 'nielsr/layoutlmv3-finetuned-funsd' 
# just to see bounding boxes in action.
MODEL_ID = "src/ai_playground/model_cache" 

print(f"⏳ Loading Model: {MODEL_ID}...")
processor = LayoutLMv3Processor.from_pretrained(MODEL_ID)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_ID)

# 2. Load an Image (You need to convert one page of your PDF to an image first)
# For this test, just put a screenshot of your table here named 'test_table.png'
image_path = "src/ai_playground/test_table.png" 
image = Image.open(image_path).convert("RGB")

# 3. Prepare Inputs for the Model
# LayoutLMv3 needs the image AND the text words (OCR). 
# The processor usually handles basic OCR if tesseract is installed, 
# or we pass image directly for visual features.
print("⚙️  Processing image...")
inputs = processor(
    image, 
    return_tensors="pt", 
    truncation=True, 
    max_length=512
)

# 4. Run Inference (The "Magic")
print("🧠  Thinking...")
with torch.no_grad():
    outputs = model(**inputs)

# 5. Decode Results
# The model gives us "logits" (raw scores). We turn them into class labels.
predictions = outputs.logits.argmax(-1).squeeze().tolist()
token_boxes = inputs.bbox.squeeze().tolist()
normalized_boxes = []

# Map IDs to actual Label Names (e.g., "B-HEADER", "B-QUESTION", "B-ANSWER")
id2label = model.config.id2label

print("\n--- 🔍 Detections ---")
for box, label_id in zip(token_boxes, predictions):
    label = id2label[label_id]
    if label != "O": # "O" means "Outside" / Irrelevant text
        print(f"Box: {box} | Label: {label}")

print("\n✅ Done! If you see 'B-HEADER' or 'B-ANSWER', it works.")