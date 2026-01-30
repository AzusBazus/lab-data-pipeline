import os
import json
from src.config import RAW_DATA_DIR
from src.parser.pdf_extractor import MedicalLabParser

def main():
    # 1. Get files
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: Directory not found: {RAW_DATA_DIR}")
        return

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith('.pdf')]
    
    if not files:
        print("⚠️  No PDF files found in data/raw/")
        return

    print(f"📂 Found {len(files)} files. Starting extraction...\n")

    # 2. Process each file
    for filename in files:
        file_path = os.path.join(RAW_DATA_DIR, filename)
        
        parser = MedicalLabParser(file_path)
        patient, results = parser.parse()
        
        # --- LOGGING THE OUTPUT ---
        print("="*60)
        print(f"📄 RESULTS FOR: {filename}")
        print(f"👤 Patient: {patient}")
        print("-" * 20)
        
        # Print first 5 results to keep console clean, or all if you prefer
        for i, res in enumerate(results):
            if i < 5: 
                print(f"   🔹 [{res['category']}] {res['test_name']}: {res['value']} (Unit: {res['unit']})")
            else:
                print(f"   ... and {len(results) - 5} more rows.")
                break
        print("="*60 + "\n")

if __name__ == "__main__":
    main()