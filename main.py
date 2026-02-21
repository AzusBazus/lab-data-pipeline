import os
import shutil
from src.extraction.document import MedicalDocument
from src.extraction.converter import DocumentConverter
from src.extraction.ocr import TextExtractor
from src.model.inference import LayoutLMPredictor
from src.config import DATA_INPUT_PATH, DATA_OUTPUT_PATH, DATA_FAILED_PATH

def main():
    extractor = TextExtractor()
    predictor = LayoutLMPredictor() 

    os.makedirs(DATA_FAILED_PATH, exist_ok=True)
    
    incoming_files = [f for f in os.listdir(DATA_INPUT_PATH) if not f.startswith('.')]
    if not incoming_files:
        print("Inbox is empty. Nothing to process.")
        return

    for filename in incoming_files:
        file_path = os.path.join(DATA_INPUT_PATH, filename)
        
        print(f"\n--- Processing: {filename} ---")
        
        doc = MedicalDocument(file_path)
        DocumentConverter.convert_to_images(doc)
        extractor.extract(doc)
        predictor.predict(doc)

        for page in doc.extracted_data:
            for token in page:
                if "label" in token:
                    print(f"Found: {token['text']} -> {token['label']} ({token['confidence']:.2f})")
                else:
                    print(f"Found: {token['text']}, no label")
        
        # processed_path = os.path.join(DATA_OUTPUT_PATH, doc.filename)
        # shutil.move(doc.original_path, processed_path)
        # print(f"✅ Successfully processed and archived: {doc.filename}")
            
        
if __name__ == "__main__":
    main()