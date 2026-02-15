import requests
import json

# --- CONFIG ---
LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3ODI5NTU5MCwiaWF0IjoxNzcxMDk1NTkwLCJqdGkiOiI4OTg2YjVjZGU1ZjA0ODdiOGM4M2M5YjRiNjQxOWQ4MiIsInVzZXJfaWQiOiIxIn0.iU8hDAvvkM0N8BgKEuALmiA9Spm5qc2LoS646WJ4CJg"
PROJECT_ID = 1

def main():
    # URL to list projects (This is a simpler check than export)
    test_url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/"
    
    headers = {
        "Authorization": f"Token {API_KEY}"
    }
    
    print(f"🕵️ Testing connection to: {test_url}")
    print(f"🔑 Using Header: 'Authorization: Token {API_KEY[:5]}...{API_KEY[-5:]}'")

    try:
        # First, just check if we can see the project
        r = requests.get(test_url, headers=headers)
        
        if r.status_code == 401:
            print("\n❌ 401 Unauthorized! The server rejected your key.")
            print("   - Did you include the word 'Token ' in the header?")
            print("   - Did you copy a whitespace character by mistake?")
            return
            
        if r.status_code == 404:
            print(f"\n❌ 404 Not Found! Project ID {PROJECT_ID} does not exist.")
            print("   - Check your browser URL: /projects/ID/")
            return

        r.raise_for_status()
        print("✅ Connection Successful! Project found.")
        
        # NOW run the export
        print("⏳ Downloading tasks...")
        export_url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/export?exportType=JSON&download_all_tasks=true"
        r = requests.get(export_url, headers=headers)
        
        data = r.json()
        print(f"✅ Downloaded {len(data)} tasks.")
        
        with open("src/ai_playground/project_tasks.json", "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"❌ Crash: {e}")

if __name__ == "__main__":
    main()