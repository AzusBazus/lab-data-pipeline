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
        
        for res in results:
            print(f"   🔹 [{res['category']}] {res['test_name']}: Value: {res['value']}, Norm: {res['norm']}, Unit: {res['unit']}")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()