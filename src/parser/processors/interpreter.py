from src.config import COLUMN_KEYWORDS
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