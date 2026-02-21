import os
import pdfplumber

class MedicalDocument:
    def __init__(self, file_path):
        self.original_path = file_path
        self.filename = os.path.basename(file_path)
        self.file_ext = os.path.splitext(file_path)[1].lower()
        self.is_digital = self._check_if_digital()
        self.pdf_path = None     
        self.pages = []          
        self.extracted_data = [] 

    def _check_if_digital(self):
        """Determines if the file has a readable text layer."""
        if self.file_ext == '.docx':
            return True  # Word docs are always digital
            
        elif self.file_ext == '.pdf':
            # Peek inside the PDF to see if it has selectable text
            try:
                with pdfplumber.open(self.original_path) as pdf:
                    if len(pdf.pages) > 0:
                        first_page_text = pdf.pages[0].extract_text()
                        # If we find actual text characters, it's digital
                        if first_page_text and len(first_page_text.strip()) > 10:
                            return True
            except Exception as e:
                print(f"Warning: Could not read {self.filename} to check digital status.")
                return False
                
        return False