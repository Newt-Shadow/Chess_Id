import os
import time
from roboflow import Roboflow

# --- CONFIGURATION ---
RAW_DIR = "datasets/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def download_titan_v3():
    print("\n🚀 PHASE 1: DOWNLOADING TITAN v3 (13+ Sources)\n")

    # ---------------------------------------------------------
    # PART A: ROBOFLOW (Reliable + New User Datasets)
    # ---------------------------------------------------------
    print("--- 📦 ROBOFLOW DATASETS ---")
    try:
        # Check environment or ask for key
        key = os.environ.get("ROBOFLOW_KEY") 
        if not key:
            key = input("🔑 Enter Roboflow API Key: ")
        
        rf = Roboflow(api_key=key)

        # 1. THE FOUNDATION (Previous "Titan" Datasets)
        foundation_sources = [
            # (Workspace, Project, Version, Local_Folder_Name)
            ("joseph-nelson", "chess-pieces-new", 24, "rf_standard"),
            ("roboflow-100", "chess-pieces-mjzgj", 2, "rf_benchmark"),
            ("chess-ojn0h", "chess_merged", 1, "rf_merged"),
        ]

        # 2. THE NEW EXPANSION (From your links)
        # We assume Version 1 for these. If v1 fails, the script will skip safely.
        new_sources = [
            ("chess-9jfdk", "chess-0rofm", 1, "rf_new_0rofm"),
            ("chess-5tjmt", "chess-piece-na374", 1, "rf_new_na374"),
            ("somework", "chess-2-ha2gu", 1, "rf_new_ha2gu"),
            ("michael-th53m", "chess-llr39", 1, "rf_new_llr39"),
            ("chess-xezgz", "chess-hndwj", 1, "rf_new_hndwj"),
            ("budai", "chess-1zpfy", 1, "rf_new_1zpfy"),
        ]

        all_roboflow = foundation_sources + new_sources

        for workspace, project, version, folder in all_roboflow:
            download_rf_dataset(rf, workspace, project, version, folder)

    except Exception as e:
        print(f"   ❌ Roboflow Error: {e}")

    # ---------------------------------------------------------
    # PART B: KAGGLE (Existing Collection)
    # ---------------------------------------------------------
    print("\n--- 📦 KAGGLE COLLECTION ---")
    
    kaggle_sources = {
        "kg_anshul": "anshulmehtakaggl/chess-pieces-detection-images-dataset",
        "kg_tanner": "tannergi/chess-piece-detection",
        "kg_majid": "majidalsant/chess-pieces-detection-dataset"
    }

    for name, slug in kaggle_sources.items():
        dest = f"{RAW_DIR}/{name}"
        if os.path.exists(dest) and len(list(os.scandir(dest))) > 10:
            print(f"   ✅ {name} seems ready (skipping).")
        else:
            print(f"   ⬇️  Downloading {name}...")
            # --force to fix any broken previous downloads
            res = os.system(f"kaggle datasets download -d {slug} -p {dest} --unzip --force")
            if res != 0:
                print(f"      ❌ Failed to download {name}. Check Kaggle API.")
            else:
                print(f"      ✅ {name} Complete.")

    print("\n🎉 TITAN v3 DOWNLOAD COMPLETE.")
    print("   Proceed to Step 2 (Processing).")

def download_rf_dataset(rf_client, workspace, project, version, folder_name):
    """Helper to safely download RF datasets"""
    dest = f"{RAW_DIR}/{folder_name}"
    
    # Skip if already exists and looks populated
    if os.path.exists(dest) and len(list(os.scandir(dest))) > 2:
        print(f"   ✅ {project} (v{version}) already exists.")
        return

    print(f"   ⬇️  Downloading: {project} (v{version})...")
    try:
        rf_client.workspace(workspace).project(project).version(version).download("yolov8", location=dest)
        time.sleep(1) # Rate limit protection
    except Exception as e:
        print(f"       ⚠️ Skipped {project}: {e}")
        # Sometimes v1 doesn't exist, you could try v2 manually if needed, 
        # but usually v1 is the default for user projects.

if __name__ == "__main__":
    download_titan_v3()