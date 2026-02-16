import requests
import json

# --- CONFIG ---
LABEL_STUDIO_URL = "http://localhost:8080" # Ensure this matches your browser URL exactly
PROJECT_ID = 1

# 🔴 PASTE YOUR COOKIE VALUE HERE (No "Bearer", no "Token", just the string)
SESSION_ID = ".eJxVT8mOhCAQ_RfOSopFkD7Ofb7BFFAo0wY6osksmX8fnfSlLy-pt6Z-2JEjuzFQGIagqHcORa8VDP0YEXptQCd7QlCadaxuM5b8jXuuZXrc2U10bMW2T2udczlPa4UYrNSWSyk0qLFjEx77Mh2Ntul_SrAXzmO4U7mE-IFlrjzUsm_Z88vCn2rj7zXS-vb0vhQs2JYzrSyS09E4HyMoI0abggaHEVIakyKM5KSF5E3UQpGLYNSZSMYLGcjLq7RRa9dn9PnI2xe7ycFJAA6_f0h9XAo:1vrgw6:MUAfTybahFEekljf0ibgan0LL5GBIUjMOEf6OjP98yI" 

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
        with open("src/data/project_tasks.json", "w") as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()