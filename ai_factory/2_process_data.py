import shutil
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import difflib
import xml.etree.ElementTree as ET
import json
import cv2
import numpy as np

# --- CONFIGURATION ---
BASE = Path("datasets")
RAW = BASE / "raw"
OUT = BASE / "processed"

CLASSES = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR', 'corner']

# --- DICTIONARY ---
KNOWN_MAP = {
    'bb': 0, 'b_bishop': 0, 'black-bishop': 0, 'bishop_black': 0, 'blackbishop': 0, 'b-bishop': 0, 'black_bishop': 0, 'bishop-black': 0, 'b.bishop': 0, 'b bishop': 0, 'black bishop': 0, 'black_b': 0, 'bishop_b': 0, 'b_b': 0, 'b-b': 0,
    'bk': 1, 'b_king': 1, 'black-king': 1, 'king_black': 1, 'blackking': 1, 'b-king': 1, 'black_king': 1, 'king-black': 1, 'b.king': 1, 'b king': 1, 'black king': 1, 'black_k': 1, 'king_b': 1, 'b_k': 1, 'b-k': 1,
    'bn': 2, 'b_knight': 2, 'black-knight': 2, 'knight_black': 2, 'b_horse': 2, 'black_horse': 2, 'b-knight': 2, 'black_knight': 2, 'knight-black': 2, 'b.knight': 2, 'black_night': 2, 'b knight': 2, 'black knight': 2, 'black_n': 2, 'knight_b': 2, 'b_n': 2, 'b-n': 2, 'black night': 2, 'b night': 2,
    'bp': 3, 'b_pawn': 3, 'black-pawn': 3, 'pawn_black': 3, 'blackpawn': 3, 'b-pawn': 3, 'black_pawn': 3, 'pawn-black': 3, 'b.pawn': 3, 'b pawn': 3, 'black pawn': 3, 'black_p': 3, 'pawn_b': 3, 'b_p': 3, 'b-p': 3,
    'bq': 4, 'b_queen': 4, 'black-queen': 4, 'queen_black': 4, 'blackqueen': 4, 'b-queen': 4, 'black_queen': 4, 'queen-black': 4, 'b.queen': 4, 'b queen': 4, 'black queen': 4, 'black_q': 4, 'queen_b': 4, 'b_q': 4, 'b-q': 4,
    'br': 5, 'b_rook': 5, 'black-rook': 5, 'rook_black': 5, 'blackrook': 5, 'b_castle': 5, 'b-rook': 5, 'black_rook': 5, 'rook-black': 5, 'b.rook': 5, 'black_castle': 5, 'b rook': 5, 'black rook': 5, 'black_r': 5, 'rook_b': 5, 'b_r': 5, 'b-r': 5,
    'wb': 6, 'w_bishop': 6, 'white-bishop': 6, 'bishop_white': 6, 'whitebishop': 6, 'w-bishop': 6, 'white_bishop': 6, 'bishop-white': 6, 'w.bishop': 6, 'w bishop': 6, 'white bishop': 6, 'white_b': 6, 'bishop_w': 6, 'w_b': 6, 'w-b': 6,
    'wk': 7, 'w_king': 7, 'white-king': 7, 'king_white': 7, 'whiteking': 7, 'w-king': 7, 'white_king': 7, 'king-white': 7, 'w.king': 7, 'w king': 7, 'white king': 7, 'white_k': 7, 'king_w': 7, 'w_k': 7, 'w-k': 7,
    'wn': 8, 'w_knight': 8, 'white-knight': 8, 'knight_white': 8, 'w_horse': 8, 'white_horse': 8, 'w-knight': 8, 'white_knight': 8, 'knight-white': 8, 'w.knight': 8, 'white_night': 8, 'w knight': 8, 'white knight': 8, 'white_n': 8, 'knight_w': 8, 'w_n': 8, 'w-n': 8, 'white night': 8, 'w night': 8,
    'wp': 9, 'w_pawn': 9, 'white-pawn': 9, 'pawn_white': 9, 'whitepawn': 9, 'w-pawn': 9, 'white_pawn': 9, 'pawn-white': 9, 'w.pawn': 9, 'w pawn': 9, 'white pawn': 9, 'white_p': 9, 'pawn_w': 9, 'w_p': 9, 'w-p': 9,
    'wq': 10, 'w_queen': 10, 'white-queen': 10, 'queen_white': 10, 'whitequeen': 10, 'w-queen': 10, 'white_queen': 10, 'queen-white': 10, 'w.queen': 10, 'w queen': 10, 'white queen': 10, 'white_q': 10, 'queen_w': 10, 'w_q': 10, 'w-q': 10,
    'wr': 11, 'w_rook': 11, 'white-rook': 11, 'rook_white': 11, 'whiterook': 11, 'w_castle': 11, 'w-rook': 11, 'white_rook': 11, 'rook-white': 11, 'w.rook': 11, 'white_castle': 11, 'w rook': 11, 'white rook': 11, 'white_r': 11, 'rook_w': 11, 'w_r': 11, 'w-r': 11,
    'corner': 12, 'board_corner': 12, 'chess_board_corner': 12, 'corners': 12, 'board': 12, 'background': None
}

def intelligent_map(raw_name):
    clean = str(raw_name).lower().strip().replace('-', '_')
    if clean in KNOWN_MAP: return KNOWN_MAP[clean]
    matches = difflib.get_close_matches(clean, list(KNOWN_MAP.keys()), n=1, cutoff=0.7)
    return KNOWN_MAP[matches[0]] if matches else None

def normalize(box, w, h):
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(w, box[2]), min(h, box[3])
    return [((x1+x2)/2)/w, ((y1+y2)/2)/h, (x2-x1)/w, (y2-y1)/h]

# --- PROCESSORS ---

def process_chessred(src):
    if not (src/"annotations.json").exists(): return
    print(f"🔨 Processing ChessReD...")
    with open(src/"annotations.json") as f: data = json.load(f)
    piece_map = {'b':0,'k':1,'n':2,'p':3,'q':4,'r':5,'B':6,'K':7,'N':8,'P':9,'Q':10,'R':11}
    image_paths = {p.name: p for p in src.rglob("*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg']}

    for entry in tqdm(data.get('annotations', []), desc="ChessReD"):
        img_id = entry['image_id']
        path = None
        for k, v in image_paths.items():
            if img_id in k: 
                path = v
                break
        if not path: continue
        
        img = cv2.imread(str(path))
        if img is None: continue
        h, w = img.shape[:2]
        
        corners = np.array(entry['corners'], dtype=np.float32)
        if len(corners) != 4: continue
        M = cv2.getPerspectiveTransform(corners, np.array([[0,0],[800,0],[800,800],[0,800]], dtype=np.float32))
        M_inv = np.linalg.inv(M)
        
        labels = []
        for c in corners: labels.append([12] + normalize([c[0]-20, c[1]-20, c[0]+20, c[1]+20], w, h))
        rows = entry['fen'].split(' ')[0].split('/')
        for r, row in enumerate(rows):
            c = 0
            for char in row:
                if char.isdigit(): c += int(char)
                else:
                    center_sq = np.array([[[c*100+50, r*100+50]]], dtype=np.float32)
                    px = cv2.perspectiveTransform(center_sq, M_inv)[0][0]
                    box = [px[0]-40, px[1]-80, px[0]+40, px[1]+20]
                    labels.append([piece_map[char]] + normalize(box, w, h))
                    c += 1
        
        split = 'train' if hash(img_id) % 10 != 0 else 'valid'
        shutil.copy(path, OUT/split/'images'/f"cr_{img_id}")
        with open(OUT/split/'labels'/f"cr_{img_id}.txt", 'w') as f:
            for l in labels: f.write(f"{l[0]} {' '.join(map(str, l[1:]))}\n")

def convert_xml_to_yolo(xml_file, width, height):
    labels = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            class_id = intelligent_map(name)
            if class_id is None: continue
            bndbox = obj.find('bndbox')
            xmin, ymin = float(bndbox.find('xmin').text), float(bndbox.find('ymin').text)
            xmax, ymax = float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
            x_c, y_c = ((xmin + xmax) / 2) / width, ((ymin + ymax) / 2) / height
            bw, bh = (xmax - xmin) / width, (ymax - ymin) / height
            labels.append(f"{class_id} {x_c} {y_c} {bw} {bh}\n")
    except: return []
    return labels

def process_kaggle_tanner(folder_path):
    print(f"🔨 Processing TannerGi: {folder_path.name}")
    base_dir = folder_path / "Chess Detection"
    if not base_dir.exists(): base_dir = folder_path if (folder_path / "images").exists() else None
    if not base_dir: return

    img_dir, ann_dir = base_dir / "images", base_dir / "annotations"
    files = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    
    for i, img_file in enumerate(tqdm(files, desc="Tanner")):
        xml_file = ann_dir / img_file.with_suffix(".xml").name
        if not xml_file.exists(): continue
        try:
             tree = ET.parse(xml_file)
             w = int(tree.getroot().find('size').find('width').text)
             h = int(tree.getroot().find('size').find('height').text)
        except:
             img = cv2.imread(str(img_file))
             if img is None: continue
             h, w = img.shape[:2]

        new_labels = convert_xml_to_yolo(xml_file, w, h)
        if new_labels:
             # 80/20 Split for verification
             split = 'train' if i % 5 != 0 else 'valid'
             unique_name = f"tanner_{img_file.name}"
             shutil.copy(img_file, OUT / split / 'images' / unique_name)
             with open(OUT / split / 'labels' / f"tanner_{img_file.stem}.txt", 'w') as f:
                 f.writelines(new_labels)

def process_roboflow_yolo(folder_path, prefix):
    yaml_path = folder_path / "data.yaml"
    src_names = None
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            src_names = yaml.safe_load(f).get('names', [])

    for split in ['train', 'valid', 'test']:
        # LOGIC CHANGE: Merge 'test' into 'valid' so you can verify during training
        target_split = 'train' if split == 'train' else 'valid'
        
        img_dir = folder_path / split / "images"
        lbl_dir = folder_path / split / "labels"
        
        if not img_dir.exists(): continue
        files = [p for p in img_dir.glob("*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']]
        
        for img_file in tqdm(files, desc=f"{prefix} {split}", leave=False):
            lbl_file = lbl_dir / img_file.with_suffix(".txt").name
            if not lbl_file.exists(): continue

            new_labels = []
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    try:
                        cls_idx = int(parts[0])
                        raw_name = None
                        if src_names:
                             if isinstance(src_names, list) and cls_idx < len(src_names):
                                 raw_name = src_names[cls_idx]
                             elif isinstance(src_names, dict) and cls_idx in src_names:
                                 raw_name = src_names[cls_idx]
                        std_id = None
                        if raw_name: std_id = intelligent_map(raw_name)
                        elif cls_idx < 13: std_id = cls_idx 

                        if std_id is not None:
                            new_labels.append(f"{std_id} {' '.join(parts[1:])}\n")
                    except: continue

            if new_labels:
                unique_name = f"{prefix}_{img_file.name}"
                shutil.copy(img_file, OUT / target_split / 'images' / unique_name)
                with open(OUT / target_split / 'labels' / f"{prefix}_{img_file.stem}.txt", 'w') as f:
                    f.writelines(new_labels)

def main():
    if OUT.exists(): shutil.rmtree(OUT)
    for s in ['train', 'valid']:
        (OUT/s/'images').mkdir(parents=True, exist_ok=True)
        (OUT/s/'labels').mkdir(parents=True, exist_ok=True)
        
    print("🚀 STARTING TITAN PROCESSOR (Merged Test->Valid)...")

    if (RAW / "chessred").exists(): process_chessred(RAW / "chessred")
    
    for rf_folder in RAW.glob("rf_*"):
        if rf_folder.is_dir(): process_roboflow_yolo(rf_folder, rf_folder.name)

    target = None
    if (RAW / "kg_tanner").exists(): target = RAW / "kg_tanner"
    elif (RAW / "kaggle_tanner").exists(): target = RAW / "kaggle_tanner"
    if target: process_kaggle_tanner(target)

    yaml_data = {'path': str(OUT.absolute()), 'train': 'train/images', 'val': 'valid/images', 
                 'nc': len(CLASSES), 'names': {i:n for i,n in enumerate(CLASSES)}}
    with open(OUT/'data.yaml', 'w') as f: yaml.dump(yaml_data, f)

    t_count = len(list((OUT/'train'/'images').glob('*')))
    v_count = len(list((OUT/'valid'/'images').glob('*')))
    
    print(f"\n✅ PROCESS COMPLETE.")
    print(f"   Train Images: {t_count} (For learning)")
    print(f"   Valid Images: {v_count} (For verification while training)")
    print(f"   Total: {t_count + v_count}")

if __name__ == "__main__":
    main()