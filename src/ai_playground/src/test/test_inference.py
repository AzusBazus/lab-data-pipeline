import os
import random
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

# --- CONFIGURATION ---
MODEL_PATH = "src/ai_playground/model_output/final" # Load YOUR trained model
IMAGE_DIR = "src/ai_playground/archive"
CONFIDENCE_THRESHOLD = 0.5 # Only show boxes with >50% confidence

# Define colors for your labels for easy visualization
COLORS = {
    "Table_Context": "darkblue",
    "Section_Header": "red",
    "Test_Name": "green",
    "Test_Value": "orange",
    "Test_Unit": "cyan",
    "Patient_Name": "purple",
    "Patient_ID": "grey"
}

def main():
    # 1. Load Model & Processor
    print(f"⏳ Loading model from {MODEL_PATH}...")
    try:
        model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
        # Use the base processor (it handles image resizing/OCR)
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Pick a Random Image
    all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
    if not all_images:
        print("❌ No PNG images found in archive.")
        return
    
    filename = random.choice(all_images)
    image_path = os.path.join(IMAGE_DIR, filename)
    print(f"📸 Testing on: {filename}")

    image = Image.open(image_path).convert("RGB")

    # 3. Run Inference
    # We must use the same truncation logic as training
    inputs = processor(
        image, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Decode Output
    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    # Get probabilities to filter low-confidence guesses
    probs = logits.softmax(-1).max(-1).values.squeeze().tolist()
    
    token_boxes = inputs.bbox.squeeze().tolist()
    id2label = model.config.id2label # The model remembers your labels!

    # 5. Draw on Image
    draw = ImageDraw.Draw(image)
    # Optional: Load a font, or use default
    # font = ImageFont.truetype("arial.ttf", 15) 

    print("\n--- 🔍 Detections ---")
    found_something = False
    
    width, height = image.size

    for box, label_id, prob in zip(token_boxes, predictions, probs):
        label = id2label[label_id]
        
        # Skip "Outside" or Low Confidence
        if label == "O" or prob < CONFIDENCE_THRESHOLD:
            continue

        found_something = True
        
        # Un-normalize box (0-1000 -> pixels)
        unnorm_box = [
            box[0] * width / 1000,
            box[1] * height / 1000,
            box[2] * width / 1000,
            box[3] * height / 1000
        ]

        # Draw Box
        # Strip "B-" or "I-" to get the core name for color lookup
        core_label = label.replace("B-", "").replace("I-", "")
        color = COLORS.get(core_label, "black")
        
        draw.rectangle(unnorm_box, outline=color, width=2)
        draw.text((unnorm_box[0], unnorm_box[1] - 10), f"{label} ({prob:.2f})", fill=color)
        
        print(f"Found {label} : {prob:.2%} confidence")

    if found_something:
        # Save the result
        output_filename = f"prediction_{filename}"
        image.save(output_filename)
        print(f"\n✅ Saved visualized result to: {output_filename}")
        print("Go open that file and check if the boxes are correct!")
    else:
        print("⚠️ No labels detected. The model might need more training or the image is empty.")

if __name__ == "__main__":
    main()