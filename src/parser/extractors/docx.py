from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import pandas as pd
import os
from src.config import LABEL_KEYWORDS, LABEL_NOISE_KEYWORDS
from src.parser.processors.interpreter import Interpreter
from src.parser.processors.table_handler import TableHandler
from src.parser.processors.value_handler import ValueHandler

class DocxMedicalParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

    def parse(self):
        print(f"--- Parsing DOCX: {self.filename} ---")
        if not os.path.exists(self.filepath): return {}, []

        document = Document(self.filepath)
        all_results = []
        
        # We need to maintain context as we scroll through the document
        current_context = "Unknown"
        
        # Buffer to hold recent paragraphs (candidates for labels)
        recent_paragraphs = []

        # ITERATE THROUGH DOCUMENT BODY (Order matters!)
        for element in document.element.body:
            
            # CASE A: It's a PARAGRAPH
            if isinstance(element, CT_P):
                para = Paragraph(element, document._body)
                text = para.text.strip()
                if text:
                    recent_paragraphs.append(text)
                    # Keep buffer small (last 5 lines is enough context)
                    if len(recent_paragraphs) > 5:
                        recent_paragraphs.pop(0)

            # CASE B: It's a TABLE
            elif isinstance(element, CT_Tbl):
                table = Table(element, document._body)
                
                # 1. Convert to Grid
                grid = [[cell.text.strip().replace('\n', ' ') for cell in row.cells] for row in table.rows]
                if not grid: continue

                # 2. Check if it's the Patient Info Table
                if Interpreter.is_patient_table(grid):
                    # We can assume patient info is handled elsewhere or extracting here
                    continue
                
                # 3. Determine Context (Label)
                # We look at recent paragraphs to find the best label
                label = self._find_best_label(recent_paragraphs)
                if label:
                    current_context = label
                    # Clear buffer so we don't reuse this label for the next table incorrectly
                    recent_paragraphs = [] 
                
                # 4. Process Table
                df = pd.DataFrame(grid)
                
                # Demultiplex (Split Left/Right)
                split_dfs = TableHandler.demultiplex(df)
                
                for sub_df in split_dfs:
                    sub_grid = sub_df.where(pd.notnull(sub_df), None).values.tolist()
                    
                    # Interpret
                    extracted_rows, _ = Interpreter.process_table_data(
                        sub_grid, 
                        context=current_context,
                        page_num=1
                    )
                    
                    # Extra Clean for Antibiotics
                    # (This ensures "Пенициллины: 1) Бензилпенициллин" -> "Бензилпенициллин")
                    if "антибиоти" in current_context.lower():
                         for row in extracted_rows:
                             if row.get('test_name'):
                                 row['test_name'] = ValueHandler.clean_antibiotic_name(row['test_name'])

                    all_results.extend(extracted_rows)

        # Separate Patient Info extraction (could be done via a pre-scan or integrated above)
        # For simplicity, let's do a quick pre-scan for patient info using the existing method
        # logic if needed, or rely on the fact that Interpreter.extract_patient_info 
        # is robust enough to run on raw_tables list if we collected them.
        
        # (Re-opening simply to get patient info from the whole doc structure is safer 
        # if we want to reuse the existing `extract_patient_info` logic exactly)
        patient_info, _ = Interpreter.extract_patient_info(self._get_all_tables_as_grids(document))
        
        return patient_info, all_results

    def _get_all_tables_as_grids(self, document):
        grids = []
        for table in document.tables:
            grid = [[cell.text.strip().replace('\n', ' ') for cell in row.cells] for row in table.rows]
            grids.append(grid)
        return grids

    def _find_best_label(self, paragraphs):
        """
        Scans the last few paragraphs to find the most likely table header.
        Prioritizes UPPERCASE and keywords. Excludes 'Version', 'EUCAST'.
        """
        if not paragraphs: return None
        
        best_candidate = None
        best_score = -1
        
        # Look at the last 3 paragraphs (closest to table is last)
        candidates = paragraphs[-3:]
        
        for text in candidates:
            score = 0
            upper_text = text.upper()
            
            # Criterion 1: UPPERCASE (Strong signal)
            if text.isupper() and len(text) > 5:
                score += 50
                
            # Criterion 2: Keywords
            if any(k in upper_text for k in LABEL_KEYWORDS):
                score += 50
                
            # Criterion 3: Noise Penalty
            if any(n in upper_text for n in LABEL_NOISE_KEYWORDS):
                score -= 100
            
            # Criterion 4: Length (Too short is bad, too long is paragraph text)
            if len(text) < 3: score -= 20
            if len(text) > 100: score -= 20

            if score > best_score and score > 0:
                best_score = score
                best_candidate = text
                
        return best_candidate