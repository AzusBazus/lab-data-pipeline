import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from src.extraction.document import MedicalDocument
from src.config import CUSTOM_MODEL_PATH, BASE_MODEL_PATH

class LayoutLMPredictor:
    def __init__(self):
        print(f"🧠 Loading LayoutLMv3 Model from {CUSTOM_MODEL_PATH}...")
        
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(CUSTOM_MODEL_PATH)
        self.processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_PATH, apply_ocr=False)
        self.id2label = self.model.config.id2label
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, doc: MedicalDocument):
        print(f"🔮 Predicting labels for: {doc.filename}")

        for i, (image, tokens) in enumerate(zip(doc.pages, doc.extracted_data)):
            if not tokens:
                continue

            words = [t['text'] for t in tokens]
            boxes = [t['bbox'] for t in tokens]

            # 1. Enable Sliding Window (Chunking)
            encoding = self.processor(
                image, 
                words, 
                boxes=boxes, 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length",
                max_length=512,
                return_overflowing_tokens=True, # Create chunks
                stride=128                      # Overlap by 128 tokens
            )

            # 2. Fix Pixel Values for Batched Chunks
            num_chunks = encoding['input_ids'].shape[0]
            pixel_values = encoding['pixel_values']
            chunk_word_ids = [encoding.word_ids(batch_index=i) for i in range(num_chunks)]
            
            # Catch the Hugging Face quirk: convert list to PyTorch Tensor
            if isinstance(pixel_values, list):
                if len(pixel_values) > 0 and isinstance(pixel_values[0], torch.Tensor):
                    pixel_values = torch.stack(pixel_values)
                else:
                    pixel_values = torch.tensor(pixel_values)

            # Ensure it has a batch dimension (Shape must be: [batch, channels, height, width])
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            
            # If we only have 1 image but 3 text chunks, duplicate the image 3 times
            if pixel_values.shape[0] == 1 and num_chunks > 1:
                pixel_values = pixel_values.repeat(num_chunks, 1, 1, 1)

            encoding['pixel_values'] = pixel_values

            # Remove keys that the model doesn't expect in its forward pass
            encoding.pop("overflow_to_sample_mapping", None)

            # Move inputs to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # 3. Forward Pass (Batch process all chunks at once)
            with torch.no_grad():
                outputs = self.model(**encoding)

            logits = outputs.logits # Shape: [num_chunks, 512, num_labels]
            chunk_preds = logits.argmax(-1)
            chunk_probs = torch.softmax(logits, dim=-1).max(-1).values

            # 4. Merge Chunks using "Max Confidence"
            best_predictions = {} # Dictionary to store: word_idx -> (label, confidence)

            for chunk_idx in range(num_chunks):
                word_ids = chunk_word_ids[chunk_idx]
                
                for seq_idx, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        continue # Skip special tokens like [CLS] and [SEP]

                    label_id = chunk_preds[chunk_idx][seq_idx].item()
                    label_name = self.id2label[label_id]
                    confidence = chunk_probs[chunk_idx][seq_idx].item()

                    if label_name == "O":
                        continue # We don't care about background

                    # If we've seen this word in a previous chunk, only overwrite if confidence is higher
                    if word_idx in best_predictions:
                        if confidence > best_predictions[word_idx][1]:
                            best_predictions[word_idx] = (label_name, confidence)
                    else:
                        best_predictions[word_idx] = (label_name, confidence)

            # 5. Apply the winning predictions back to our document object
            for word_idx, (label, conf) in best_predictions.items():
                tokens[word_idx]["label"] = label
                tokens[word_idx]["confidence"] = conf
            
            print(f"   📄 Page {i+1}: Classified {len(best_predictions)} entities across {num_chunks} chunks.")