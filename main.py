import os
import json
from src.config import RAW_DATA_DIR
from src.parser.extractors.pdf import MedicalLabParser
from src.parser.extractors.docx import DocxMedicalParser

# Define directory constants if not already imported
RAW_DATA_DIR = "data/raw" 

def main():
    # 1. Get files (PDF or DOCX)
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: Directory not found: {RAW_DATA_DIR}")
        return

    # Updated filter to accept tuple of extensions
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(('.pdf', '.docx'))]
    
    if not files:
        print("⚠️  No compatible files (PDF/DOCX) found in data/raw/")
        return

    print(f"📂 Found {len(files)} files. Starting extraction...\n")

    # 2. Process each file
    for filename in files:
        file_path = os.path.join(RAW_DATA_DIR, filename)
        
        # --- SELECT PARSER BASED ON EXTENSION ---
        if filename.lower().endswith('.pdf'):
            parser = MedicalLabParser(file_path)
        elif filename.lower().endswith('.docx'):
            parser = DocxMedicalParser(file_path)
        else:
            continue

        try:
            patient, results = parser.parse()
            
            # --- LOGGING THE OUTPUT ---
            print("="*60)
            print(f"📄 RESULTS FOR: {filename}")
            print(f"👤 Patient: {patient}")
            print("-" * 20)
            
            for res in results:
                # Safety check in case results are empty or malformed during dev
                if isinstance(res, dict):
                    print(f"   🔹 [Category:{res.get('category', 'N/A')}]")
                    print(f"     Test Name: {res.get('test_name')}")
                    print(f"     Value: {res.get('value')} (Text: {res.get('text_value')})")
                    print(f"     Norm: {res.get('norm')}")
                    print(f"     Unit: {res.get('unit')}\n")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()