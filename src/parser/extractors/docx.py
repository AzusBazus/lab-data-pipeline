import os
from docx import Document

class DocxMedicalParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

    def parse(self):
        """
        Main entry point for DOCX files.
        Currently just dumps content for inspection.
        """
        print(f"--- Parsing DOCX: {self.filename} ---")
        
        if not os.path.exists(self.filepath):
            print(f"Error: File not found {self.filepath}")
            return {}, []

        document = Document(self.filepath)
        
        # 1. Inspect Paragraphs (Text outside tables)
        print("\n--- Raw Paragraphs ---")
        for i, para in enumerate(document.paragraphs):
            text = para.text.strip()
            if text:
                print(f"Para {i}: {text}")

        # 2. Inspect Tables
        print("\n--- Raw Tables ---")
        for i, table in enumerate(document.tables):
            print(f"Table {i} (Rows: {len(table.rows)}, Cols: {len(table.columns)})")
            for row in table.rows:
                # Extract text from each cell
                row_data = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                print(row_data)

        # Return empty dummy data for now so main.py doesn't crash
        return {}, []