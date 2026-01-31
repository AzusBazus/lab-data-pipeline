import re
import pdfplumber
from src.utils.text_matching import is_fuzzy_match
from src.config import COLUMN_KEYWORDS, NOISE_PATTERNS

class MedicalLabParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.split("/")[-1] # For logging
        self.metadata = {}

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

                # Extract Metadata (Creation Date)
                self._extract_metadata_from_noise(text_lines)
                
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

    def _extract_metadata_from_noise(self, text_lines):
        """
        Scans page text for the 'Print page' timestamp to capture creation date.
        Only needs to find it once per document.
        """
        # If we already found it, skip (assuming it's the same on every page)
        if self.metadata.get('creation_date'): return

        for line in text_lines:
            text = line['text']
            # Look for DD.MM.YYYY, HH:MM
            match = re.search(r'(\d{2}\.\d{2}\.\d{4}),\s+(\d{2}:\d{2})', text)
            if match:
                self.metadata['creation_date'] = f"{match.group(1)} {match.group(2)}"
                # Don't break loop, we just grab the first one we see

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
        table_top = table_bbox[1]
        
        candidates = []
        
        for line in text_lines:
            # 1. Spatial Check (Is it immediately above?)
            is_above = line['bottom'] < table_top and (table_top - line['bottom']) < 100
            if not is_above:
                continue

            text = line['text'].strip()
            
            # 2. Noise Check (Is it a printer artifact?)
            if self._is_noise(text):
                continue
                
            candidates.append(line)
        
        if not candidates:
            return None
            
        return candidates[-1]['text'].strip()

    def _is_noise(self, text):
        """Returns True if text matches any of our known noise patterns."""
        return any(pattern.search(text) for pattern in NOISE_PATTERNS)

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

    def _find_header_row_index(self, data):
        """Finds the row index that contains column headers."""
        # We look for the "Result" column as the strongest anchor
        target_headers = COLUMN_KEYWORDS['result']
        
        for i, row in enumerate(data):
            for cell in row:
                if not cell: continue
                if any(is_fuzzy_match(str(cell), t) for t in target_headers):
                    return i
        return 0 # Fallback to first row

    def _map_header_indices(self, header_row):
        """
        Analyzes a single header row and figures out which index is which.
        Returns: dict { 'test_name': 0, 'result': 2, ... }
        """
        col_map = {}
        
        # Iterate over our known concepts (test_name, result, etc.)
        for col_type, keywords in COLUMN_KEYWORDS.items():
            
            # Check every cell in the header row
            for idx, cell in enumerate(header_row):
                if not cell: continue
                
                # Check fuzzy match against ALL keywords for this concept
                # We use a higher threshold (90) here because headers are usually clear
                if any(is_fuzzy_match(str(cell), k, threshold=88) for k in keywords):
                    col_map[col_type] = idx
                    break # Stop looking for this column once found
        
        # FAILSAFE: If fuzzy matching failed completely (e.g. empty header), 
        # revert to the "standard" structure you observed: [Name, Result, Norm, Unit]
        if 'result' not in col_map:
            # You can log a warning here if you want
            print(f"⚠️ Warning: Could not auto-map columns. Using default structure.")
            return {'test_name': 0, 'result': 1, 'norm': 2, 'unit': 3}
            
        return col_map

    def _process_table_rows(self, data, context, page_num):
        results = []
        if not data: return results

        # 1. Identify the Header Row
        header_idx = self._find_header_row_index(data)
        
        # 2. Map Columns (The Upgrade)
        # This returns something like: {'test_name': 0, 'result': 2, 'unit': 3}
        header_row = data[header_idx]
        col_map = self._map_header_indices(header_row)

        # 3. Iterate rows AFTER the header
        for row in data[header_idx+1:]:
            if not row or len(row) < 2: continue
            
            # Use the map to pluck values safely
            clean_row = {
                "category": context,
                "page": page_num + 1,
                
                # Fetch using the mapped index, or None if that column wasn't found
                "test_name": row[col_map['test_name']] if col_map.get('test_name') is not None and len(row) > col_map['test_name'] else None,
                "value":     row[col_map['result']]    if col_map.get('result')    is not None and len(row) > col_map['result']    else None,
                "norm":      row[col_map['norm']]      if col_map.get('norm')      is not None and len(row) > col_map['norm']      else None,
                "unit":      row[col_map['unit']]      if col_map.get('unit')      is not None and len(row) > col_map['unit']      else None,
            }
            
            # Integrity Check: If we didn't find a Result or a Name, this row is likely garbage
            if clean_row['test_name'] and clean_row['value']:
                results.append(clean_row)
            
        return results