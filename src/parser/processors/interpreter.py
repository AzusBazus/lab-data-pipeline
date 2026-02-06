import re
import pandas as pd
from src.config import COLUMN_KEYWORDS, PATIENT_FIELDS, DATE_PATTERN, YEAR_PATTERN, HIDDEN_RESULT_KEYWORDS
from src.parser.utils.text_matching import is_fuzzy_match
from src.parser.processors.value_handler import ValueHandler

class Interpreter:
    @staticmethod
    def process_table_data(data, context, page_num, inherited_map=None):
        """
        Main logic to convert a clean 2D grid into structured Lab Result Dictionaries.
        """
        extracted_rows = []
        if not data: return [], inherited_map

        # 1. Determine Structure (Header Mapping)
        header_idx = Interpreter._find_header_row_index(data)
        col_map = None
        start_row_idx = 0
        
        if header_idx is not None:
            col_map = Interpreter._map_header_indices(data[header_idx])
            start_row_idx = header_idx + 1
        elif inherited_map is not None:
            col_map = inherited_map
            start_row_idx = 0
        else:
            # Fallback
            col_map = Interpreter._map_header_indices(data[0])
            start_row_idx = 1
            
        # 2. Extract Basic Dictionaries
        for row in data[start_row_idx:]:
            if not row or len(row) < 2: continue
            
            def get_col(name):
                idx = col_map.get(name)
                if idx is not None and len(row) > idx:
                    return row[idx]
                return None

            clean_row = {
                "category": context,
                "page": page_num + 1,
                "test_name": get_col('test_name'),
                "value":     get_col('result'),
                "text_value": get_col('result'), 
                "norm":      get_col('norm'),
                "unit":      get_col('unit'),
            }
            
            if clean_row['test_name'] and clean_row['value']:
                extracted_rows.append(clean_row)

        # 3. Pipeline Processing (Value Cleaning)
        # Note: We call ValueEngine here to ensure the output is fully ready
        expanded_rows = ValueHandler.expand_composite_rows(extracted_rows)
        results_with_units = ValueHandler.infer_missing_units(expanded_rows)
        final_results = ValueHandler.normalize_result_values(results_with_units)

        return final_results, col_map

    @staticmethod
    def is_patient_table(table_data):
        """
        Universal check: Does this table contain metadata labels (FIO, DOB)?
        """
        # Optimization: Flatten once, lower-case once
        unique_words = set()
        for row in table_data:
            for cell in row:
                if cell:
                    # Clean the cell content: remove numbers, punctuation, extra spaces
                    # This helps isolated words stand out
                    cleaned_cell = str(cell).lower().strip()
                    unique_words.add(cleaned_cell)

        # Check against our config
        identifying_anchors = PATIENT_FIELDS['name'] + PATIENT_FIELDS['dob']
        
        for word in unique_words:
            # FIX 1: Ignore short garbage tokens (like 'r', 'S', 'R', '-', '1')
            if len(word) < 3: 
                continue

            for anchor in identifying_anchors:
                # FIX 2: Ensure anchor is also reasonable length (sanity check)
                if len(anchor) < 3:
                    continue
                
                # FIX 3: Strict matching
                # We check two things:
                # A. Fuzzy match is high
                # B. The length difference isn't massive (prevents "r" matching "born")
                if is_fuzzy_match(word, anchor, threshold=92):
                    
                    # Double Check: Is it a ridiculous substring match?
                    # e.g. matching "r" inside "born" is wrong.
                    len_diff = abs(len(word) - len(anchor))
                    if len_diff > 3: 
                        continue

                    # print(f"Patient table match: '{word}' matches anchor '{anchor}'")
                    return True
        return False

    @staticmethod
    def _find_header_row_index(data):
        target_headers = COLUMN_KEYWORDS['result']
        for i, row in enumerate(data):
            for cell in row:
                if not cell: continue
                if any(is_fuzzy_match(str(cell), t) for t in target_headers):
                    return i
        return None

    @staticmethod
    def _map_header_indices(header_row):
        col_map = {}
        for col_type, keywords in COLUMN_KEYWORDS.items():
            for idx, cell in enumerate(header_row):
                if not cell: continue
                if any(is_fuzzy_match(str(cell), k, threshold=88) for k in keywords):
                    col_map[col_type] = idx
                    break
        
        if 'result' not in col_map:
            return {'test_name': 0, 'result': 1, 'norm': 2, 'unit': 3}
        return col_map

    @staticmethod
    def extract_patient_info(tables):
        """
        Universal strategy to find patient info in a list of tables (PDF or DOCX).
        Returns: (patient_info_dict, list_of_extra_results)
        """
        info = {}
        extra_results = []

        for table in tables:
            # Check if this table looks like a patient table
            if Interpreter.is_patient_table(table):
                info, new_results = Interpreter._parse_patient_table(table)
                # Merge found info
                extra_results.extend(new_results)
                # If we found the name, we assume this was the correct table and stop?
                # Or keep looking for more info? Usually stop is safe if name found.
                if info.get('name'):
                    break
        
        return info, extra_results

    @staticmethod
    def _parse_patient_table(table_data):
        """
        Scans cells for metadata (Name, DOB) AND specific result keywords.
        """
        info = {}
        results = []
        
        # Keywords for the "Hidden Result" in the patient header
        # "В посеве обнаружено" -> "Detected in culture"
        

        for r_idx, row in enumerate(table_data):
            for c_idx, cell in enumerate(row):
                if not cell: continue
                cell_text = str(cell).strip()
                cell_lower = cell_text.lower()
                
                # 1. Check for Name
                if not info.get('name') and any(is_fuzzy_match(cell_lower, k) for k in PATIENT_FIELDS['name']):
                    val = Interpreter._get_next_cell_value(table_data, r_idx, c_idx)
                    if val: info['name'] = val
                    continue

                # 2. Check for DOB
                if not info.get('dob') and any(is_fuzzy_match(cell_lower, k) for k in PATIENT_FIELDS['dob']):
                    val = Interpreter._get_next_cell_value(table_data, r_idx, c_idx)
                    date_str = Interpreter._extract_date_string(val)
                    if date_str: info['dob'] = date_str
                    continue
                
                # 3. Check for Report Date
                if any(is_fuzzy_match(cell_lower, k) for k in PATIENT_FIELDS['report_date']):
                    # Ensure it's not actually the DOB field (fuzzy match overlap)
                    if not any(is_fuzzy_match(cell_lower, k) for k in PATIENT_FIELDS['dob']):
                        val = Interpreter._get_next_cell_value(table_data, r_idx, c_idx)
                        date_str = Interpreter._extract_date_string(val)
                        if date_str: info['report_date'] = date_str
                    continue

                # --- NEW: HEIGHT ---
                if not info.get('height') and any(k in cell_lower for k in PATIENT_FIELDS['height']):
                    # Strategy A: Check inside CURRENT cell (e.g. "Рост: 180см")
                    val = Interpreter._standardize_height(cell_text)
                    
                    # Strategy B: If not found, check NEXT cell
                    if not val:
                        next_cell = Interpreter._get_next_cell_value(table_data, r_idx, c_idx)
                        val = Interpreter._standardize_height(next_cell)
                    
                    if val: info['height_cm'] = val # Store as explicit unit

                # --- NEW: WEIGHT ---
                if not info.get('weight') and any(k in cell_lower for k in PATIENT_FIELDS['weight']):
                    # Strategy A: Check inside CURRENT cell
                    val = Interpreter._standardize_weight(cell_text)
                    
                    # Strategy B: Check NEXT cell
                    if not val:
                        next_cell = Interpreter._get_next_cell_value(table_data, r_idx, c_idx)
                        val = Interpreter._standardize_weight(next_cell)
                        
                    if val: info['weight_kg'] = val

                # 4. THE HIDDEN RESULT ("В посеве обнаружено")
                if any(is_fuzzy_match(cell_text, k) for k in HIDDEN_RESULT_KEYWORDS):
                    val = Interpreter._get_next_cell_value(table_data, r_idx, c_idx)
                    if val:
                        # Create a result object immediately
                        results.append({
                            "category": "Microbiology",
                            "test_name": "Culture Result", # Standardized name
                            "value": None, # It's text, not a number
                            "text_value": val, # "Escherichia coli 105 KOE"
                            "norm": None,
                            "unit": None
                        })

        return info, results

    @staticmethod
    def _get_next_cell_value(table, r, c):
        """Priority: Right -> Right+1 -> Below"""
        # Try Right
        if c + 1 < len(table[r]):
            val = table[r][c+1]
            if val and str(val).strip(): return str(val).strip()
        # Try Right + 1 (Merged cells)
        if c + 2 < len(table[r]):
            val = table[r][c+2]
            if val and str(val).strip(): return str(val).strip()
        # Try Below (rare but possible)
        if r + 1 < len(table):
            val = table[r+1][c]
            if val and str(val).strip(): return str(val).strip()
        return None

    @staticmethod
    def _extract_date_string(text):
        if not text: return None
        
        match = DATE_PATTERN.search(text)
        if match:
             return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"

        year_match = YEAR_PATTERN.search(text)
        if year_match:
            return f"00.00.{year_match.group(0)}"
            
        return None

    @staticmethod
    def _standardize_height(text):
        """
        Parses height string and returns Float in CM.
        Handles: "1м72см", "172 см", "1.72 м", "185см"
        """
        if not text: return None
        text = text.lower().replace(',', '.').replace(' ', '')
        
        # Pattern 1: Composite (e.g., 1м72, 1m72cm)
        # Looks for digit + 'м' + digit
        composite_match = re.search(r'(\d+)[мm](\d+)', text)
        if composite_match:
            meters = float(composite_match.group(1))
            cm = float(composite_match.group(2))
            return round(meters * 100 + cm, 1)

        # Pattern 2: Centimeters (e.g., 185см, 185cm)
        cm_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:см|cm)', text)
        if cm_match:
            return float(cm_match.group(1))
            
        # Pattern 3: Meters (e.g., 1.72м, 1.72m)
        m_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:м|m)\b', text)
        if m_match:
            return round(float(m_match.group(1)) * 100, 1)
            
        return None

    @staticmethod
    def _standardize_weight(text):
        """
        Parses weight string and returns Float in KG.
        Handles: "127кг", "127.5 kg"
        """
        if not text: return None
        text = text.lower().replace(',', '.').replace(' ', '')
        
        # Simple Number + Unit
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:кг|kg)', text)
        if match:
            return float(match.group(1))
            
        return None