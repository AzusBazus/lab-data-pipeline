import os
import subprocess
from pdf2image import convert_from_path
from src.extraction.document import MedicalDocument

class DocumentConverter:
    @staticmethod
    def convert_to_images(doc: MedicalDocument):
        
        if doc.file_ext == '.docx':
            print(f"Converting Word Document to PDF: {doc.filename}")
            doc.pdf_path = doc.original_path.replace('.docx', '.pdf')
            
            try:
                subprocess.run([
                    'soffice', 
                    '--headless',
                    '--convert-to', 'pdf',
                    '--outdir', os.path.dirname(doc.pdf_path),
                    doc.original_path
                ], check=True)
            
                os.remove(doc.original_path)
                
                doc.original_path = doc.pdf_path
                doc.filename = os.path.basename(doc.pdf_path)
                doc.file_ext = '.pdf'
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to convert {doc.filename}: {e}")
                return
        else:
            doc.pdf_path = doc.original_path

        print(f"Extracting images from: {doc.pdf_path}")
        doc.pages = convert_from_path(doc.pdf_path)