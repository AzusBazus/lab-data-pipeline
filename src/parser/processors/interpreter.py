from src.config import COLUMN_KEYWORDS, PATIENT_FIELDS, DATE_PATTERN, YEAR_PATTERN
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
    def is_patient_table(data):
        """Checks if table contains patient metadata keywords."""
        unique_words = set(str(cell) for row in data for cell in row if cell)
        anchors = ["ф.и.о.", "фамилия", "patient", "name"]
        for word in unique_words:
            for anchor in anchors:
                if is_fuzzy_match(word, anchor, threshold=90):
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
            if Interpreter._is_patient_info_table(table):
                # Parse it
                info, new_results = Interpreter._parse_patient_table(table)
                # Merge found info
                extra_results.extend(new_results)
                # If we found the name, we assume this was the correct table and stop?
                # Or keep looking for more info? Usually stop is safe if name found.
                if info.get('name'):
                    break
        
        return info, extra_results

    @staticmethod
    def _is_patient_info_table(table_data):
        """Returns True if the table contains 'Name' or 'DOB' keywords."""
        for row in table_data:
            for cell in row:
                if cell and any(is_fuzzy_match(str(cell), k) for k in PATIENT_FIELDS['name']):
                    return True
        return False

    @staticmethod
    def _parse_patient_table(table_data):
        """
        Scans cells for metadata (Name, DOB) AND specific result keywords.
        """
        info = {}
        results = []
        
        # Keywords for the "Hidden Result" in the patient header
        # "В посеве обнаружено" -> "Detected in culture"
        HIDDEN_RESULT_KEYWORDS = ["посеве обнаружено", "рост микрофлоры", "detected"]

        for r_idx, row in enumerate(table_data):
            for c_idx, cell in enumerate(row):
                if not cell: continue
                cell_text = str(cell).strip()
                
                # 1. Check for Name
                if not info.get('name') and any(is_fuzzy_match(cell_text, k) for k in PATIENT_FIELDS['name']):
                    val = Interpreter._get_next_cell_value(table_data, r_idx, c_idx)
                    if val: info['name'] = val
                    continue

                # 2. Check for DOB
                if not info.get('dob') and any(is_fuzzy_match(cell_text, k) for k in PATIENT_FIELDS['dob']):
                    val = Interpreter._get_next_cell_value(table_data, r_idx, c_idx)
                    date_str = Interpreter._extract_date_string(val)
                    if date_str: info['dob'] = date_str
                    continue
                
                # 3. Check for Report Date
                if any(is_fuzzy_match(cell_text, k) for k in PATIENT_FIELDS['report_date']):
                    # Ensure it's not actually the DOB field (fuzzy match overlap)
                    if not any(is_fuzzy_match(cell_text, k) for k in PATIENT_FIELDS['dob']):
                        val = Interpreter._get_next_cell_value(table_data, r_idx, c_idx)
                        date_str = Interpreter._extract_date_string(val)
                        if date_str: info['report_date'] = date_str
                    continue

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