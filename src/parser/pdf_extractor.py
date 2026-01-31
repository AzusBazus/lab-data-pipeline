import pdfplumber
import re
from src.utils.text_matching import is_fuzzy_match

class MedicalLabParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.split("/")[-1] # For logging

    def parse(self):
        """
        Main method to extract data.
        Returns: (patient_info_dict, list_of_results)
        """
        extracted_results = []
        patient_info = {}
        
        current_context = "Unknown Category" # State variable

        print(f"--- Parsing {self.filename} ---")

        with pdfplumber.open(self.filepath) as pdf:
            # --- STEP 1: Extract Patient Info (Page 1) ---
            if len(pdf.pages) > 0:
                patient_info = self._extract_patient_info(pdf.pages[0])

            # --- STEP 2: Iterate all pages for Tables ---
            for page_num, page in enumerate(pdf.pages):
                # Get text lines for header detection
                text_lines = page.extract_text_lines()
                
                # Get tables
                tables = page.find_tables()
                # Sort top-to-bottom
                tables.sort(key=lambda t: t.bbox[1])

                for table in tables:
                    # A. Find the Label (The "Y-Axis Truth")
                    label = self._find_header_above_table(table.bbox, text_lines)
                    
                    if label:
                        current_context = label
                    # If no label found, we implicitly keep the old current_context
                    
                    # B. Extract Data
                    data = table.extract()
                    if not data: continue

                    # C. Clean and Append Data
                    # Skip if it's the patient info table (contains "Ф.И.О.")
                    if self._is_patient_table(data):
                        continue

                    cleaned_rows = self._process_table_rows(data, current_context, page_num)
                    extracted_results.extend(cleaned_rows)

        return patient_info, extracted_results

    def _extract_patient_info(self, page):
        """Simple keyword search on the first page for patient details"""
        text = page.extract_text()
        info = {}
        
        # Regex to find Name after "Ф.И.О."
        # Looks for: Ф.И.О. -> optional space -> Capture Name -> End of line or specific keyword
        name_match = re.search(r"Ф\.И\.О\.[\s\n]*([А-Яа-яЁё\s]+)", text)
        if name_match:
            info['name'] = name_match.group(1).strip()
            
        # Regex for DOB (DD.MM.YYYY)
        dob_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
        if dob_match:
            info['dob'] = dob_match.group(1)
            
        return info

    def _find_header_above_table(self, table_bbox, text_lines):
        """Finds text physically located just above the table."""
        table_top = table_bbox[1]
        
        # Look for text in the 100px zone above the table
        candidates = [
            line for line in text_lines 
            if line['bottom'] < table_top and (table_top - line['bottom']) < 100
        ]
        
        if not candidates:
            return None
            
        # The closest text is the header
        return candidates[-1]['text'].strip()

    def _is_patient_table(self, data):
        """
        Robust check: Looks for 'Ф.И.О.' or 'Patient Name' anywhere in the table.
        """
        # Flatten the table into a set of unique strings to search faster
        unique_words = set()
        for row in data:
            for cell in row:
                if cell:
                    unique_words.add(str(cell))
    
        # Check our known keywords
        anchors = ["ф.и.о.", "фамилия", "patient", "name"]
    
        for word in unique_words:
            for anchor in anchors:
                if is_fuzzy_match(word, anchor, threshold=90):
                    return True
        return False

    def _process_table_rows(self, data, context, page_num):
        """Converts raw list-of-lists into nice dictionaries"""
        results = []
    
        # 1. Identify the Header Row using Fuzzy Matching
        header_idx = -1
    
        # List of possible names for the Result column
        target_headers = ["результат", "result", "value", "значение"]
    
        for i, row in enumerate(data):
            # Check every cell in this row
            for cell in row:
                if not cell: continue
            
                # Check if ANY target header matches this cell
                match_found = any(is_fuzzy_match(str(cell), t) for t in target_headers)
            
                if match_found:
                    header_idx = i
                    break
            if header_idx != -1: break
    
        # Fallback: If fuzzy match failed, assume row 0
        if header_idx == -1:
            header_idx = 0

        # Iterate rows AFTER the header
        for row in data[header_idx+1:]:
            # Skip empty rows or single-column garbage
            if not row or len(row) < 2: continue
            
            # Simple Mapping (You can improve this later based on column position)
            # Assuming: [Test Name, Result, Norm, Unit] structure roughly
            clean_row = {
                "category": context,
                "test_name": row[0], 
                "value": row[1] if len(row) > 1 else None,
                "norm": row[2] if len(row) > 2 else None,
                "unit": row[3] if len(row) > 3 else None,
                "page": page_num + 1
            }
            results.append(clean_row)
            
        return results