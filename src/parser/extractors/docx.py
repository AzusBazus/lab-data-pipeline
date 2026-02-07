from docx import Document
import os
import pandas as pd
from src.parser.processors.interpreter import Interpreter
from src.parser.processors.table_handler import TableHandler

class DocxMedicalParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

    def parse(self):
        print(f"--- Parsing DOCX: {self.filename} ---")
        if not os.path.exists(self.filepath): return {}, []

        document = Document(self.filepath)
        all_results = []
        
        # 1. Get all tables as raw lists of lists
        raw_tables = []
        for table in document.tables:
            grid = [[cell.text.strip().replace('\n', ' ') for cell in row.cells] for row in table.rows]
            raw_tables.append(grid)
            
        # 2. Extract Patient Info & The "Hidden Result"
        patient_info, extra_results = Interpreter.extract_patient_info(raw_tables)
        all_results.extend(extra_results)
        
        # 3. Process the Result Tables
        for grid in raw_tables:
            if Interpreter.is_patient_table(grid): 
                continue
                
            # Convert to DataFrame for handling
            df = pd.DataFrame(grid)
            
            # A. Demultiplex (Splits 1 table into 2 if needed)
            split_dfs = TableHandler.demultiplex(df)
            
            for sub_df in split_dfs:
                print("Table:")
                print(sub_df)
                print("\n")

        return patient_info, all_results

    
        

    