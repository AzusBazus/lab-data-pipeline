import os
import shutil
from src.extraction.document import MedicalDocument
from src.extraction.converter import DocumentConverter
from src.extraction.ocr import TextExtractor
from src.config import DATA_INPUT_PATH, DATA_OUTPUT_PATH

def main():
    # 1. Load the heavy AI models into memory once
    extractor = TextExtractor() 
    
    # 2. Check the inbox
    incoming_files = os.listdir(DATA_INPUT_PATH)
    if not incoming_files:
        print("Inbox is empty. Nothing to process.")
        return

    for filename in incoming_files:
        file_path = os.path.join(DATA_INPUT_PATH, filename)
        
        # A. Create the Document Object
        doc = MedicalDocument(file_path)
        
        # B. Convert to Images
        DocumentConverter.convert_to_images(doc)
        
        # C. Extract the Text and Boxes
        extractor.extract(doc)

        print("Digital: ", doc.is_digital)
        print("Extension: ", doc.file_ext)
        print("PDF Path: ", doc.pdf_path)
        print("Pages: ", len(doc.pages))
        
        # processed_path = os.path.join(DATA_OUTPUT_PATH, filename)
        # shutil.move(file_path, processed_path)
        # print(f"✅ Successfully processed and archived: {filename}")

if __name__ == "__main__":
    main()