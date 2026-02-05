import re
import pdfplumber
from src.parser.utils.text_matching import is_fuzzy_match
from src.parser.processors.table_handler import TableHandler
from src.parser.processors.interpreter import Interpreter
from src.config import NOISE_PATTERNS, PATIENT_FIELDS, DATE_PATTERN

class MedicalLabParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.split("/")[-1] # For logging
        self.metadata = {}

    def parse(self):
        extracted_results = []
        patient_info = {}
        current_context = "Unknown Category"
        current_col_map = None 
        current_parent_cat = None 

        print(f"--- Parsing {self.filename} ---")

        with pdfplumber.open(self.filepath) as pdf:
            if len(pdf.pages) > 0:
                patient_info = self._extract_patient_info(pdf.pages[0])

            for page_num, page in enumerate(pdf.pages):
                text_lines = page.extract_text_lines()
                self._extract_metadata_from_noise(text_lines)
                
                tables = page.find_tables()
                tables.sort(key=lambda t: t.bbox[1])

                for i, table in enumerate(tables):
                    label = self._find_header_above_table(table.bbox, text_lines)
                    
                    # LOGIC: New Label = New Context AND New Structure
                    if label:
                        current_context = label
                        current_col_map = None
                        current_parent_cat = None 
                    
                    data = table.extract()
                    if not data: continue
                    if Interpreter.is_patient_table(data): continue

                    data = TableHandler.clean_and_normalize(data)

                    if TableHandler.is_table_hierarchical(data) or (current_parent_cat is not None):
                        # Flatten it! Pass the state from previous page/table
                        data, current_parent_cat = TableHandler.flatten_hierarchical_table(
                            data, 
                            inherited_parent=current_parent_cat
                        )
                    else:
                        current_parent_cat = None

                    # Pass the inherited map, receive the updated map
                    cleaned_rows, used_map = Interpreter.process_table_data(
                        data, 
                        current_context, 
                        page_num, 
                        inherited_map=current_col_map
                    )
                    
                    # Persist the map for the next loop/page
                    if used_map:
                        current_col_map = used_map
                        
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