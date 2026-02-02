import re
import pdfplumber
from src.utils.text_matching import is_fuzzy_match
from src.parser.cleaner import expand_composite_rows, infer_missing_units
from src.config import COLUMN_KEYWORDS, NOISE_PATTERNS, PATIENT_FIELDS, DATE_PATTERN

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

                for i, table in enumerate(tables):
                    # A. Find the Label (The "Y-Axis Truth")
                    label = self._find_header_above_table(table.bbox, text_lines)
                    
                    # If no label found, we implicitly keep the old current_context
                    if label:
                        current_context = label
                    
                    # B. Extract Data
                    data = table.extract()

                    if not data:
                        continue

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
        """
        Extracts patient info by looking for a specific table structure.
        """
        info = {}
        
        # 1. Try extracting from TABLES (Most Robust)
        tables = page.extract_tables()
        
        for table in tables:
            # Flatten table to checking if this is the "Patient Table"
            # We check if ANY cell matches ANY "name" keyword
            if self._is_patient_info_table(table):
                info = self._parse_patient_table(table)
                break
        
        # 2. Fallback: Text-based (only if table extraction failed completely)
        if not info.get('name'):
            print("⚠️ Table extraction failed for patient info. Using fallback.")
            info = self._extract_patient_info_fallback(page.extract_text())
            
        return info

    def _is_patient_info_table(self, table_data):
        """Returns True if the table contains 'Name' or 'DOB' keywords."""
        for row in table_data:
            for cell in row:
                if cell and any(is_fuzzy_match(str(cell), k) for k in PATIENT_FIELDS['name']):
                    return True
        return False

    def _parse_patient_table(self, table_data):
        """
        Iterates through the table cells to find Name, DOB, and Report Date.
        Logic: Find a keyword cell -> The value is likely in the NEXT non-empty cell.
        """
        info = {}
        
        # Iterate through every cell
        for r_idx, row in enumerate(table_data):
            for c_idx, cell in enumerate(row):
                if not cell: continue
                
                cell_text = str(cell).strip()
                
                # --- CHECK FOR NAME ---
                if not info.get('name'):
                    if any(is_fuzzy_match(cell_text, k) for k in PATIENT_FIELDS['name']):
                        # Found label! Get the value.
                        val = self._get_next_cell_value(table_data, r_idx, c_idx)
                        if val: info['name'] = val

                # --- CHECK FOR DOB ---
                if not info.get('dob'):
                    if any(is_fuzzy_match(cell_text, k) for k in PATIENT_FIELDS['dob']):
                        val = self._get_next_cell_value(table_data, r_idx, c_idx)
                        # Extract just the date part if there's extra text
                        date_str = self._extract_date_string(val)
                        if date_str: info['dob'] = date_str

                # --- CHECK FOR REPORT DATE ---
                # We want to distinguish "Date" (Report) from "Date of Birth"
                # Simple logic: If it matches "Date" but NOT "DOB"
                is_dob_label = any(is_fuzzy_match(cell_text, k) for k in PATIENT_FIELDS['dob'])
                if not is_dob_label and any(is_fuzzy_match(cell_text, k) for k in PATIENT_FIELDS['report_date']):
                     val = self._get_next_cell_value(table_data, r_idx, c_idx)
                     date_str = self._extract_date_string(val)
                     if date_str: 
                         self.metadata['creation_date'] = date_str # Save to metadata
        
        return info

    def _get_next_cell_value(self, table, r, c):
        """
        Looks for the value associated with a label at [r, c].
        Priority:
        1. Immediate right cell [r, c+1]
        2. Cell two steps right [r, c+2] (common in merged layouts)
        3. Immediate cell below [r+1, c]
        """
        # Try Right (Row, Col+1)
        if c + 1 < len(table[r]):
            val = table[r][c+1]
            if val and str(val).strip(): return str(val).strip()

        # Try Right + 1 (Row, Col+2) - Handles empty separator columns
        if c + 2 < len(table[r]):
            val = table[r][c+2]
            if val and str(val).strip(): return str(val).strip()
            
        # Try Below (Row+1, Col)
        if r + 1 < len(table):
            val = table[r+1][c]
            if val and str(val).strip(): return str(val).strip()
            
        return None

    def _extract_date_string(self, text):
        """Finds DD.MM.YYYY, DD-MM-YYYY, or DD/MM/YYYY and normalizes to DD.MM.YYYY"""
        if not text: return None
        
        match = DATE_PATTERN.search(text)
        if match:
            # Normalize to dots: 15-10-1983 -> 15.10.1983
            return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
        return None

    def _extract_patient_info_fallback(self, text):
        """
        Regex fallback if table detection fails completely.
        Updated to stop capturing name when it hits a "Stop Word".
        """
        info = {}
        
        # 1. Name: Look for anchor, grab text, STOP at a newline or another keyword
        # (?i) = case insensitive
        # (.*?) = non-greedy capture
        # (?=...) = Lookahead (stop before you hit this)
        stop_words = "Дата|Date|Born|Рожд|Результат|Result|Analys"
        
        name_pattern = fr"(?:Ф\.?И\.?О\.?|Name|Patient)[\s:.]*([^\n]+?)(?=\s*(?:{stop_words}|\n|$))"
        name_match = re.search(name_pattern, text, re.IGNORECASE)
        
        if name_match:
            info['name'] = name_match.group(1).strip()

        # 2. DOB: Look for DOB keywords
        dob_pattern = fr"(?:Дата\s*рождения|DOB|Born)[\s:.]*([\d\.\-\/]+)"
        dob_match = re.search(dob_pattern, text, re.IGNORECASE)
        
        if dob_match:
            info['dob'] = self._extract_date_string(dob_match.group(1))
            
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
        extracted_rows = []
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
                extracted_rows.append(clean_row)
                
        expanded_rows = expand_composite_rows(extracted_rows)
        final_results = infer_missing_units(expanded_rows)

        return final_results