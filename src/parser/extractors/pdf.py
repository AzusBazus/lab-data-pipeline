import re
import pdfplumber
from src.parser.utils.text_matching import is_fuzzy_match
from src.parser.processors.table_handler import TableHandler
from src.parser.processors.interpreter import Interpreter
from src.config import NOISE_PATTERNS, LABEL_KEYWORDS, LABEL_NOISE_KEYWORDS

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
                p1_tables = pdf.pages[0].extract_tables()
                patient_info, extra_res = Interpreter.extract_patient_info(p1_tables)
                extracted_results.extend(extra_res)

            for page_num, page in enumerate(pdf.pages):
                text_lines = page.extract_text_lines()
                self._extract_metadata_from_noise(text_lines)
                
                tables = page.find_tables()
                tables.sort(key=lambda t: t.bbox[1])
                all_table_bboxes = [t.bbox for t in tables]

                for i, table in enumerate(tables):
                    label = self._find_header_above_table(table.bbox, text_lines, exclusion_bboxes=all_table_bboxes)
                    
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

    def _find_header_above_table(self, table_bbox, text_lines, exclusion_bboxes=None):
        table_top = table_bbox[1]
        candidates = []
        
        for line in text_lines:
            # 1. Spatial Check (Is it immediately above?)
            # We look 100pts up.
            is_above = line['bottom'] < table_top and (table_top - line['bottom']) < 100
            if not is_above:
                continue

            # --- NEW: GEOMETRIC EXCLUSION ---
            # If this text line is strictly inside ANY table's bbox, ignore it.
            # This kills "РЕЗУЛЬТАТЫ" because it sits inside the Patient Info Table.
            if exclusion_bboxes:
                if self._is_inside_any_table(line, exclusion_bboxes):
                    continue

            text = line['text'].strip()
            
            # 2. Noise Check
            if self._is_noise(text):
                continue
                
            candidates.append(text)
        
        if not candidates:
            return None

        best_candidate = None
        best_score = -1

        for text in candidates:
            print("\n" + "Text: ")
            print(text + "\n")

            score = 0
            
            # Criterion 1: UPPERCASE
            if text.isupper() and len(text) > 5: score += 50
                
            # Criterion 2: Keywords
            if any(k in text.lower() for k in LABEL_KEYWORDS): score += 50
                
            # Criterion 3: Noise Penalty
            if any(n in text.lower() for n in LABEL_NOISE_KEYWORDS): score -= 100
            
            # Criterion 4: Length Sanity
            if len(text) < 3: score -= 20
            if len(text) > 100: score -= 20

            if score >= best_score and score > 0:
                best_score = score
                best_candidate = text


        return best_candidate.strip() if best_candidate else None

    def _is_inside_any_table(self, line_obj, table_bboxes):
        # Calculate center of the text line
        line_x = (line_obj['x0'] + line_obj['x1']) / 2
        line_y = (line_obj['top'] + line_obj['bottom']) / 2
        
        for bbox in table_bboxes:
            b_x0, b_top, b_x1, b_bottom = bbox
            
            # Check if center is strictly inside
            if (b_x0 < line_x < b_x1) and (b_top < line_y < b_bottom):
                return True
        return False

    def _is_noise(self, text):
        """Returns True if text matches any of our known noise patterns."""
        return any(pattern.search(text) for pattern in NOISE_PATTERNS)