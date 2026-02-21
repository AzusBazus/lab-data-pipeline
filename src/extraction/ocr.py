import easyocr
import pdfplumber
import numpy as np
from src.extraction.document import MedicalDocument

class TextExtractor:
    def __init__(self):
        # Initialize EasyOCR once. 
        print("\n⏳ Initializing EasyOCR (Russian/English)...")
        self.ocr_engine = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        # gpu=False ensures stability on Mac if MPS isn't perfectly configured.
        
    def extract(self, doc: MedicalDocument):
        """
        Main Router:
        Digital PDF -> pdfplumber (No-Loss)
        Scanned/Image -> EasyOCR (AI Extraction)
        """
        if doc.is_digital:
            self._extract_digital(doc)
        else:
            self._extract_scanned(doc)

    def _extract_digital(self, doc: MedicalDocument):
        print(f"\n💎 Track A: Digital Extraction on {doc.filename}")
        all_pages_data = []
        with pdfplumber.open(doc.pdf_path) as pdf:
            for page in pdf.pages:
                width, height = float(page.width), float(page.height)
                words = page.extract_words()
                page_tokens = []
                for w in words:
                    page_tokens.append({
                        "text": w['text'],
                        "bbox": [
                            int((w['x0'] / width) * 1000),
                            int((w['top'] / height) * 1000),
                            int((w['x1'] / width) * 1000),
                            int((w['bottom'] / height) * 1000)
                        ]
                    })
                all_pages_data.append(page_tokens)
        doc.extracted_data = all_pages_data

    def _extract_scanned(self, doc: MedicalDocument):
        print(f"\n📸 Track B: AI OCR Extraction (EasyOCR) on {doc.filename}")
        all_pages_data = []
        
        for img in doc.pages:
            img_np = np.array(img)
            
            # EasyOCR returns: [ ([[x0,y0], [x1,y0], [x1,y1], [x0,y1]], 'Text', confidence), ... ]
            results = self.ocr_engine.readtext(img_np)
            
            page_tokens = []
            width, height = img.size
            
            for bbox, text, confidence in results:
                # Extract min/max to get [x0, y0, x1, y1]
                x0 = min([pt[0] for pt in bbox])
                y0 = min([pt[1] for pt in bbox])
                x1 = max([pt[0] for pt in bbox])
                y1 = max([pt[1] for pt in bbox])
                
                # Normalize to 0-1000
                page_tokens.append({
                    "text": text,
                    "bbox": [
                        int((x0 / width) * 1000),
                        int((y0 / height) * 1000),
                        int((x1 / width) * 1000),
                        int((y1 / height) * 1000)
                    ]
                })
            all_pages_data.append(page_tokens)
        
        doc.extracted_data = all_pages_data