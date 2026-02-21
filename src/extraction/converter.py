import os
import subprocess
from pdf2image import convert_from_path
from PIL import Image
from src.extraction.document import MedicalDocument
from src.config import SUPPORTED_IMAGES

class DocumentConverter:
    @staticmethod
    def convert_to_images(doc: MedicalDocument):
        
        if doc.file_ext == '.docx':
            print(f"\nConverting Word Document to PDF: {doc.filename}")
            doc.pdf_path = doc.original_path.replace('.docx', '.pdf')
            
            subprocess.run([
                'soffice', 
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', os.path.dirname(doc.pdf_path),
                doc.original_path
            ], check=True)
            
            print(f"\n🗑️ Deleting original .docx: {doc.original_path}")
            os.remove(doc.original_path)
            
            doc.original_path = doc.pdf_path
            doc.filename = os.path.basename(doc.pdf_path)
            doc.file_ext = '.pdf'
            
            print(f"\nExtracting images from: {doc.pdf_path}")
            doc.pages = convert_from_path(doc.pdf_path)

        elif doc.file_ext == '.pdf':
            doc.pdf_path = doc.original_path
            print(f"\nExtracting images from: {doc.pdf_path}")
            doc.pages = convert_from_path(doc.pdf_path)
            
        elif doc.file_ext in SUPPORTED_IMAGES:
            print(f"\nLoading image directly: {doc.original_path}")
            img = Image.open(doc.original_path).convert("RGB")
            doc.pages = [img]
            
        else:
            raise ValueError(f"\nUnsupported file format '{doc.file_ext}'")