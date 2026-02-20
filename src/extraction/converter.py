import os
import subprocess
from pdf2image import convert_from_path
from PIL import Image
from src.extraction.document import MedicalDocument

# Define what image formats we accept natively
SUPPORTED_IMAGES = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

class DocumentConverter:
    @staticmethod
    def convert_to_images(doc: MedicalDocument):
        
        # --- Route 1: Word Documents ---
        if doc.file_ext == '.docx':
            print(f"Converting Word Document to PDF: {doc.filename}")
            doc.pdf_path = doc.original_path.replace('.docx', '.pdf')
            
            subprocess.run([
                'soffice', 
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', os.path.dirname(doc.pdf_path),
                doc.original_path
            ], check=True)
            
            print(f"🗑️ Deleting original .docx: {doc.original_path}")
            os.remove(doc.original_path)
            
            doc.original_path = doc.pdf_path
            doc.filename = os.path.basename(doc.pdf_path)
            doc.file_ext = '.pdf'
            
            print(f"Extracting images from: {doc.pdf_path}")
            doc.pages = convert_from_path(doc.pdf_path)

        # --- Route 2: Native PDFs ---
        elif doc.file_ext == '.pdf':
            doc.pdf_path = doc.original_path
            print(f"Extracting images from: {doc.pdf_path}")
            doc.pages = convert_from_path(doc.pdf_path)
            
        # --- Route 3: Native Images ---
        elif doc.file_ext in SUPPORTED_IMAGES:
            print(f"Loading image directly: {doc.original_path}")
            # Load the image and wrap it in a list so doc.pages is consistently a list
            img = Image.open(doc.original_path).convert("RGB")
            doc.pages = [img]
            
        # --- Route 4: Unsupported Formats ---
        else:
            # We raise an Error here to immediately stop processing this specific file
            raise ValueError(f"Unsupported file format '{doc.file_ext}'")