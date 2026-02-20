from src.extraction.document import MedicalDocument

class TextExtractor:
    def __init__(self):
        print("Loading OCR Models into memory...")
        # Initialize PaddleOCR or other models here ONCE
        
    def extract(self, document: MedicalDocument):
        # 1. Run your digital check (pdfplumber)
        # 2. If digital -> document.is_digital = True, extract exact text/boxes
        # 3. If scanned -> run PaddleOCR on document.pages
        # 4. Save results back into document.extracted_data
        pass