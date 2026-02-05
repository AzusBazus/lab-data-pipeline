import re
import pdfplumber
from src.parser.utils.text_matching import is_fuzzy_match
from src.parser.processors.table_handler import TableHandler
from src.parser.processors.interpreter import Interpreter
from src.config import NOISE_PATTERNS

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