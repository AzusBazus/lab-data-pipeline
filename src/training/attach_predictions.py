import json
import os
import urllib.parse
from config import TASKS_JSON_PATH

PREDICTIONS_JSON = "data/batch_upload/predictions.json"
OUTPUT_JSON = "data/batch_upload/ready_to_import.json"

def main():
    if not os.path.exists(TASKS_JSON_PATH):
        print(f"❌ Error: Could not find {TASKS_JSON_PATH}")
        return

    print(f"⏳ Loading huge export: {TASKS_JSON_PATH}...")
    with open(TASKS_JSON_PATH, 'r') as f:
        all_tasks = json.load(f)
    
    print(f"⏳ Loading local predictions: {PREDICTIONS_JSON}...")
    with open(PREDICTIONS_JSON, 'r') as f:
        preds_raw = json.load(f)

    # 1. Build a map of our 20 new predictions
    # Key: "page1.png" -> Value: [Prediction Object]
    pred_map = {}
    for item in preds_raw:
        # Normalize filename (handle paths if necessary)
        fname = os.path.basename(item['data']['image']) 
        pred_map[fname] = item['predictions']

    print(f"   Found {len(all_tasks)} total tasks in export.")
    print(f"   Looking for {len(pred_map)} new predictions.")

    tasks_to_update = []
    
    # 2. Filter the Huge List
    for task in all_tasks:
        # Label Studio path: "/data/upload/1/8d9bc659-%D0%9C%D0%B0...png"
        raw_path = task['data']['image']
        
        # 1. DECODE the URL (Turn %D0%9C into Cyrillic characters)
        # This converts "/data/.../%D0%9C.png" -> "/data/.../Махмудова.png"
        decoded_path = urllib.parse.unquote(raw_path)
        
        # Check if this task matches any of our new files
        match_found = False
        for fname, prediction in pred_map.items():
            # Now we compare "Махмудова.png" with "Махмудова.png"
            # We check both raw and decoded just to be safe
            if decoded_path.endswith(fname) or raw_path.endswith(fname):
                
                # MATCH! This is one of the 20 new files.
                # Attach the prediction
                task['predictions'] = prediction
                
                # Add to our "Update List"
                tasks_to_update.append(task)
                
                # Remove from map so we don't double-match
                del pred_map[fname]
                match_found = True
                break
        
        if len(pred_map) == 0:
            break

    # 3. Save ONLY the updated tasks
    if not tasks_to_update:
        print("❌ CRITICAL: No matching tasks found. Did you upload the images first?")
        return

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(tasks_to_update, f, indent=2)

    print(f"\n✅ Success! Extracted and linked {len(tasks_to_update)} tasks.")
    print(f"👉 Import '{OUTPUT_JSON}' into Label Studio.")
    print("   (This file is small and safe—it only updates the new images).")

if __name__ == "__main__":
    main()