import os
import requests

# --- CONFIGURATION ---
# Lichess high-quality SVG pieces
BASE_URL = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett"
TARGET_DIR = "assets/pieces"
PIECES = [
    "wP", "wN", "wB", "wR", "wQ", "wK", # White
    "bP", "bN", "bB", "bR", "bQ", "bK"  # Black
]

def download_assets():
    # 1. Create directory if it doesn't exist
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📂 Created directory: {TARGET_DIR}")

    print("⬇️  Downloading Chess Assets (Offline Mode)...")
    
    # 2. Download each piece
    for p in PIECES:
        filename = f"{p}.svg"
        file_path = os.path.join(TARGET_DIR, filename)
        
        # Skip if already exists (Prevents re-downloading)
        if os.path.exists(file_path):
            print(f"   ✅ {filename} already exists.")
            continue
            
        try:
            url = f"{BASE_URL}/{filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"   📥 Downloaded {filename}")
            else:
                print(f"   ❌ Failed to download {filename} (Status: {response.status_code})")
        except Exception as e:
            print(f"   ⚠️ Error {filename}: {e}")

    print("\n🎉 ASSETS READY. You can now build the app offline.")

if __name__ == "__main__":
    download_assets()