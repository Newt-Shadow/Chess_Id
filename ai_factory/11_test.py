import cv2
import numpy as np
import threading
import webbrowser
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier
import chess
from collections import deque, Counter
from PIL import Image, ImageDraw

# ================= CONFIG =================
ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "input_video.mp4"
OUT_DIR = ROOT / "data-set"
ICON_DIR = ROOT / "assets" / "chess_pieces"
MODEL_PATH = ROOT / "chess_grandmaster/round2_finetune/weights/best.pt"

BOARD_PIXELS = 640
SQUARE_SIZE = BOARD_PIXELS // 8

# TUNING
CONF_THRESHOLD = 0.35     
CHANGE_THRESHOLD = 1500   
MOTION_THRESHOLD = 15000  
HISTORY_LEN = 10          
STABILITY_THRESHOLD = 7   

# >>> HAND LOGIC <<<
PIXEL_DIFF_THRESH = 30    
CHANGED_PIXELS_MIN = 50   
MAX_HAND_SQUARES = 6      # If >6 squares change, it's a hand. Pause.

OUT_DIR.mkdir(exist_ok=True)

# ================= WEB CORNER PICKER =================
def get_corners_via_web(frame):
    app = Flask(__name__)
    pts = []
    done = threading.Event()
    HTML = """
    <html><body style="text-align:center; background:#222; color:#fff;">
    <h2>Click: TL(A1) -> TR(A8) -> BR(H8) -> BL(H1)</h2>
    <canvas id="c"></canvas>
    <script>
    let img=new Image(); img.src="/img";
    let c=document.getElementById("c"),ctx=c.getContext("2d"),p=[];
    img.onload=()=>{c.width=img.width;c.height=img.height;ctx.drawImage(img,0,0);}
    c.onclick=e=>{
        if(p.length>=4) return;
        let r=c.getBoundingClientRect();
        let x=Math.round(e.clientX-r.left), y=Math.round(e.clientY-r.top);
        p.push({x,y});
        ctx.fillStyle="#0f0";ctx.beginPath();ctx.arc(x,y,8,0,6.28);ctx.fill();
        if(p.length===4)
            fetch("/submit",{method:"POST",headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    }
    </script></body></html>
    """
    @app.route("/")
    def idx(): return render_template_string(HTML)
    @app.route("/img")
    def img():
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes(), 200, {"Content-Type":"image/jpeg"}
    @app.route("/submit", methods=["POST"])
    def submit():
        nonlocal pts
        pts = request.json
        done.set()
        return jsonify(ok=True)
    
    server = threading.Thread(target=lambda: app.run(port=5000, debug=False, use_reloader=False), daemon=True)
    server.start()
    webbrowser.open("http://127.0.0.1:5000")
    print("🌐 Select board corners in browser...")
    done.wait()
    return np.array([(p["x"], p["y"]) for p in pts], dtype=np.float32)

# ================= GEOMETRY =================

def warp_board(frame, corners):
    dst = np.array([
        [0,0], [BOARD_PIXELS-1,0],
        [BOARD_PIXELS-1,BOARD_PIXELS-1], [0,BOARD_PIXELS-1]
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame, H, (BOARD_PIXELS, BOARD_PIXELS))

def map_pixel_to_square(raw_x, raw_y):
    rank_idx = int(raw_x // SQUARE_SIZE)
    file_idx = int(raw_y // SQUARE_SIZE)
    return chess.square(max(0,min(7,file_idx)), max(0,min(7,rank_idx)))

def get_square_roi(image, sq):
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    x = rank * SQUARE_SIZE
    y = file * SQUARE_SIZE
    gap = int(SQUARE_SIZE * 0.25)
    return image[y+gap:y+SQUARE_SIZE-gap, x+gap:x+SQUARE_SIZE-gap]

# ================= LEARNING =================

class BoardClassifier:
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.trained = False
    
    def extract(self, roi):
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        mean, std = cv2.meanStdDev(lab)
        return np.array([mean[0][0], mean[1][0], mean[2][0], std[0][0]]).flatten()

    def train(self, frame, board):
        print("🎓 Learning Colors...")
        X, y = [], []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            roi = get_square_roi(frame, sq)
            features = self.extract(roi)
            label = 1 if piece else 0
            X.append(features)
            y.append(label)
        self.knn.fit(X, y)
        self.trained = True

    def predict(self, roi):
        if not self.trained: return False
        return self.knn.predict([self.extract(roi)])[0] == 1

# ================= VISION LOGIC =================

def get_background_vision(model, frame, last_valid, board, classifier):
    # 1. YOLO
    yolo_occ = set()
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, _, _ = box
        sq = map_pixel_to_square((x1+x2)/2, y2 - (y2-y1)*0.15)
        yolo_occ.add(sq)
        
    # 2. Classifier
    class_occ = set()
    for sq in chess.SQUARES:
        if classifier.predict(get_square_roi(frame, sq)):
            class_occ.add(sq)
            
    # 3. Change Detection (PIXEL COUNT)
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_last = cv2.cvtColor(last_valid, cv2.COLOR_BGR2GRAY)
    
    final_occ = set()
    changed_squares = set()
    
    for sq in chess.SQUARES:
        r1 = get_square_roi(gray_curr, sq)
        r2 = get_square_roi(gray_last, sq)
        diff = cv2.absdiff(r1, r2)
        _, thresh = cv2.threshold(diff, PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        
        if cv2.countNonZero(thresh) > CHANGED_PIXELS_MIN:
            changed_squares.add(sq)
            # Visual Change -> Trust Vision
            if (sq in yolo_occ) or (sq in class_occ):
                final_occ.add(sq)
        else:
            # No Change -> Trust Logic
            if board.piece_at(sq):
                final_occ.add(sq)
                
    return final_occ, changed_squares

# ================= BEST FIT MOVE SOLVER =================

def find_best_matching_move(board, visual_occupancy):
    best_move = None
    min_error = float('inf')
    
    current_logic_occ = {sq for sq in chess.SQUARES if board.piece_at(sq)}
    current_error = len(current_logic_occ.symmetric_difference(visual_occupancy))
    diff_squares = current_logic_occ.symmetric_difference(visual_occupancy)
    
    for move in board.legal_moves:
        if len(diff_squares) > 0:
            if move.from_square not in diff_squares and move.to_square not in diff_squares:
                continue

        board.push(move)
        pred_occ = {sq for sq in chess.SQUARES if board.piece_at(sq)}
        error = len(pred_occ.symmetric_difference(visual_occupancy))
        
        if error < min_error:
            min_error = error
            best_move = move
        board.pop()
        
    if best_move:
        if min_error < current_error: return best_move
        elif min_error == current_error and min_error <= 1: return best_move
            
    return None

def check_hand_motion(curr, prev):
    if prev is None: return False
    g1 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh) > MOTION_THRESHOLD

# ================= VISUALS =================

def draw_hud(image, board, changed_squares):
    overlay = image.copy()
    for r in range(8):
        for c in range(8):
            sq = chess.square(r, c)
            x, y = c * SQUARE_SIZE, r * SQUARE_SIZE
            cx, cy = int(x + SQUARE_SIZE/2), int(y + SQUARE_SIZE/2)
            
            # Highlight Blue Changes
            color = (0,255,255)
            if sq in changed_squares:
                cv2.rectangle(overlay, (x, y), (x+SQUARE_SIZE, y+SQUARE_SIZE), (255,0,0), 2) # Blue
                color = (255,0,0) 
            else:
                cv2.rectangle(overlay, (x, y), (x+SQUARE_SIZE, y+SQUARE_SIZE), (0,255,255), 1) # Yellow
                
            cv2.putText(overlay, chess.square_name(sq).upper(), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Logic State
            if board.piece_at(sq):
                cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), -1)
            else:
                cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)

    return cv2.addWeighted(overlay, 0.8, image, 0.2, 0)

def draw_virtual_board(board):
    sq = BOARD_PIXELS // 8
    img = Image.new("RGB",(BOARD_PIXELS,BOARD_PIXELS),"white")
    draw = ImageDraw.Draw(img)
    for r in range(8):
        for c in range(8):
            col = "#DDB88C" if (r+c)%2==0 else "#A66D4F"
            draw.rectangle([c*sq,(7-r)*sq,(c+1)*sq,(8-r)*sq], fill=col)
    for s, p in board.piece_map().items():
        f = chess.square_file(s)
        r = chess.square_rank(s)
        name = ("w" if p.color else "b") + p.symbol().upper()
        try:
            icon = Image.open(ICON_DIR / f"{name}.png").resize((sq, sq))
            img.paste(icon, (f*sq, (7-r)*sq), icon)
        except:
            draw.text((f*sq+10, (7-r)*sq+10), p.symbol(), fill="black")
    return img

# ================= MAIN =================
def process_video():
    print("🧠 Loading AI...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, first = cap.read()
    if not ok: return

    corners = get_corners_via_web(first)
    board = chess.Board()
    move_id = 0
    
    occupancy_buffer = deque(maxlen=HISTORY_LEN)
    last_valid = warp_board(first, corners)
    prev_frame = last_valid.copy()
    
    classifier = BoardClassifier()
    classifier.train(last_valid, board)
    
    out0 = OUT_DIR / "move_000_init"
    out0.mkdir(parents=True, exist_ok=True)
    draw_virtual_board(board).save(out0/"virtual.png")
    
    hand_timer = 0
    print("🎬 Processing...")
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        warped = warp_board(frame, corners)
        
        # 1. Global Motion Freeze
        if check_hand_motion(warped, prev_frame):
            hand_timer += 1
            if hand_timer < 30: 
                cv2.putText(warped, "MOTION...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                cv2.imshow("Chess AI", draw_hud(warped, board, set()))
                prev_frame = warped.copy()
                if cv2.waitKey(1) == 27: break
                continue
            else:
                hand_timer = 0
        else:
            hand_timer = 0
        prev_frame = warped.copy()
        
        # 2. Get Vision & Changes
        visual_occ, changes = get_background_vision(model, warped, last_valid, board, classifier)
        
        # >>> HAND GATE <<<
        # If too many squares changed (>6), assume Hand is present. Skip Logic.
        if len(changes) > MAX_HAND_SQUARES:
            cv2.putText(warped, "HAND ON BOARD", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
            # Show the chaos so user knows why it's paused
            cv2.imshow("Chess AI", draw_hud(warped, board, changes))
            if cv2.waitKey(1) == 27: break
            continue # SKIP EVERYTHING ELSE
            
        # 3. Buffer (Only if Hand is gone)
        occupancy_buffer.append(frozenset(visual_occ))
        
        # 4. Vote
        most_common, agreement = Counter(occupancy_buffer).most_common(1)[0]
        stable_visual_occ = set(most_common)
        
        # 5. Move Logic
        if agreement >= STABILITY_THRESHOLD:
            move = find_best_matching_move(board, stable_visual_occ)
            
            if move:
                print(f"✅ MOVE: {move.uci()}")
                board.push(move)
                move_id += 1
                
                last_valid = warped.copy()
                occupancy_buffer.clear()
                
                out = OUT_DIR / f"move_{move_id:03d}"
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out/"board.jpg"), warped)
                draw_virtual_board(board).save(out/"virtual.png")
                (out/"fen.txt").write_text(board.fen())
        
        # 6. Draw
        debug = draw_hud(warped, board, changes)
        cv2.imshow("Chess AI", debug)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()