"""This module implements util functions (TURBO MODE)."""
import zipfile
import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

def download_chessred(dataroot: str) -> None:
    """Downloads the ChessReD dataset using ARIA2C (Turbo Speed)."""
    
    # Load the YAML config
    with open('cfg/chessred.yaml', 'r') as f:
        chessred_yaml = yaml.safe_load(f)

    print("🚀 TURBO MODE: Downloading ChessReD with 16 connections...")
    
    # Create directory if it doesn't exist
    Path(dataroot).mkdir(parents=True, exist_ok=True)

    # 1. Download Annotations

    url_images = chessred_yaml['images']['url']
    dest_zip = Path(dataroot, 'images.zip')
    
    if not dest_zip.exists():
        print(f"⬇️  Downloading Images Zip...")
        # The aria2c magic command
        cmd = f"aria2c -x 16 -s 16 -o images.zip -d {dataroot} \"{url_images}\""
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            print("\n❌ Aria2c failed! The server might be blocking 16 connections.")
            print("   Trying safe mode (4 connections)...")
            os.system(f"aria2c -x 4 -s 4 -o images.zip -d {dataroot} \"{url_images}\"")
            
    else:
        print("✅ Images Zip already exists.")


    

    # 2. Download Images (The big file)
    
    




    url_json = chessred_yaml['annotations']['url']
    dest_json = Path(dataroot, 'annotations.json')
    if not dest_json.exists():
        print(f"⬇️  Downloading Annotations...")
        # -x 16: 16 connections
        # -s 16: Split file into 16 parts
        # -o: Output filename
        cmd = f"aria2c -x 16 -s 16 -o annotations.json -d {dataroot} \"{url_json}\""
        os.system(cmd)
    else:
        print("✅ Annotations already exist.")

    print("\n🎉 Download completed.")

def extract_zipfile(zip_file: str, output_directory: str) -> None:
    """Extracts `zip_file` to `output_directory`."""
    print(f"📦 Extracting ChessReD images to {output_directory}...")
    
    # Check if zip exists
    if not Path(zip_file).exists():
        print(f"❌ Error: {zip_file} not found!")
        return

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in tqdm(file_list, desc="Extracting", unit="file"):
            zip_ref.extract(file_name, output_directory)

    print("✅ Extraction completed.")