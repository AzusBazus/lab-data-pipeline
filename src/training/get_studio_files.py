import requests
import json

# --- CONFIG ---
from config import LABEL_STUDIO_URL, PROJECT_ID, SESSION_ID

def main():
    print(f"⏳ Connecting to Project {PROJECT_ID} via Session Cookie...")
    
    # Define cookies dictionary
    cookies = {
        "sessionid": SESSION_ID
    }
    
    # URL to export tasks
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/export?exportType=JSON&download_all_tasks=true"
    
    try:
        # Note: We use 'cookies=' instead of 'headers='
        response = requests.get(url, cookies=cookies)
        
        # Check for errors
        if response.status_code == 403:
             print("❌ 403 Forbidden: Your cookie might be expired. Refresh the page and copy 'sessionid' again.")
             return
        if response.status_code == 401:
             print("❌ 401 Unauthorized: The cookie is invalid.")
             return
             
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Success! Downloaded {len(data)} tasks.")
        
        # Save file
        with open("data/project_tasks.json", "w") as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()