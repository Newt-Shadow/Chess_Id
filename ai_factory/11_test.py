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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ================= CONFIG =================
ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "input_video.mp4"
OUT_DIR = ROOT / "data-set"
ICON_DIR = ROOT / "assets" / "chess_pieces"
MODEL_PATH = ROOT / "chess_grandmaster/round2_finetune/weights/best.pt"

BOARD_PIXELS = 640
SQUARE_SIZE = BOARD_PIXELS // 8

# ADAPTIVE TUNING
CONF_THRESHOLD = 0.35     
CHANGE_THRESHOLD = 1500   
MOTION_THRESHOLD = 15000  
HISTORY_LEN = 8           
STABILITY_THRESHOLD = 6   

# >>> PER-SQUARE ADAPTIVE LOGIC <<<
EDGE_CHANGE_THRESHOLD = 100 # Lowered to catch faint black pieces
PIXEL_DIFF_THRESH = 25      # More sensitive to dark-on-dark changes
CHANGED_PIXELS_MIN = 40     
MAX_HAND_SQUARES = 10     
STUCK_TIMEOUT = 40        

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

# ================= GEOMETRY & MAPPING =================

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

# ================= ADAPTIVE VISION LOGIC =================

def get_adaptive_edges(roi):
    """
    Computes Canny edges dynamically for a SINGLE SQUARE.
    This fixes the issue where dark squares were ignored.
    """
    v = np.median(roi)
    # Tighter thresholds for dark squares to catch faint lines
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(roi, lower, upper)

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
            
    # 3. Change Detection (INDEPENDENT SQUARE TRACKING)
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_last = cv2.cvtColor(last_valid, cv2.COLOR_BGR2GRAY)
    
    final_occ = set()
    changed_squares = set()
    
    # Visualization canvas for edges
    full_edge_debug = np.zeros_like(gray_curr)
    
    for sq in chess.SQUARES:
        # Get ROIs
        r_curr = get_square_roi(gray_curr, sq)
        r_last = get_square_roi(gray_last, sq)
        
        # A. Pixel Change (Adaptive Thresholding per square)
        diff = cv2.absdiff(r_curr, r_last)
        _, thresh = cv2.threshold(diff, PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        has_pixel_change = cv2.countNonZero(thresh) > CHANGED_PIXELS_MIN
        
        # B. Edge Change (Adaptive per square)
        e_curr = get_adaptive_edges(r_curr)
        e_last = get_adaptive_edges(r_last)
        e_diff = cv2.absdiff(e_curr, e_last)
        has_edge_change = cv2.countNonZero(e_diff) > EDGE_CHANGE_THRESHOLD
        
        # Paste edge diff into debug view
        y, x = (chess.square_rank(sq)*SQUARE_SIZE, chess.square_file(sq)*SQUARE_SIZE)
        gap = int(SQUARE_SIZE * 0.25)
        # Handle slice boundaries safely
        try:
            full_edge_debug[y+gap:y+SQUARE_SIZE-gap, x+gap:x+SQUARE_SIZE-gap] = e_diff
        except: pass

        # COMBINED LOGIC
        if has_pixel_change or has_edge_change:
            changed_squares.add(sq)
            if (sq in yolo_occ) or (sq in class_occ):
                final_occ.add(sq)
        else:
            if board.piece_at(sq):
                final_occ.add(sq)
                
    return final_occ, changed_squares, full_edge_debug

def heal_background(last_valid, current_frame, sq):
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    x = rank * SQUARE_SIZE
    y = file * SQUARE_SIZE
    last_valid[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE] = current_frame[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE]
    return last_valid

# ================= HISTOGRAM VISUALIZER (FIXED) =================

def draw_histogram_window(frame, changed_squares):
    """
    Creates a visual histogram for up to 4 'Changed' squares.
    """
    if not changed_squares:
        blank = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(blank, "No Active Changes", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        return blank

    fig, axes = plt.subplots(1, min(4, len(changed_squares)), figsize=(5, 3), dpi=80)
    if len(changed_squares) == 1: axes = [axes]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    active_list = list(changed_squares)[:4]
    
    for i, sq in enumerate(active_list):
        roi = get_square_roi(gray, sq)
        hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
        
        ax = axes[i]
        ax.plot(hist, color='white')
        ax.set_title(chess.square_name(sq), fontsize=10, color='blue')
        ax.axis('off')
        # Set dark background for plot
        ax.set_facecolor('black')
        
    fig.patch.set_facecolor('black')
    plt.tight_layout()
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # FIX: Use buffer_rgba() instead of tostring_rgb()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    
    # Convert RGBA -> BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    return img

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
            
            color = (255,0,0) if sq in changed_squares else (0,255,255)
            cv2.rectangle(overlay, (x, y), (x+SQUARE_SIZE, y+SQUARE_SIZE), color, 1)
            
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
    square_stuck_counters = np.zeros(64, dtype=int)
    
    # AUTO CALIBRATION
    print("⚖️ Calibrating Vision...")
    for _ in range(30):
        ret, frame = cap.read()
        warped = warp_board(frame, corners)
        get_adaptive_edges(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    print("✅ Calibration Complete.")
    
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
        
        # 2. Vision & Changes
        visual_occ, changes, edge_debug = get_background_vision(model, warped, last_valid, board, classifier)
        
        # 3. Hand Gate
        if len(changes) > MAX_HAND_SQUARES:
            cv2.putText(warped, "HAND ON BOARD", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
            cv2.imshow("Chess AI", draw_hud(warped, board, changes))
            if cv2.waitKey(1) == 27: break
            continue 
            
        # 4. Buffer
        occupancy_buffer.append(frozenset(visual_occ))
        
        # 5. Vote
        most_common, agreement = Counter(occupancy_buffer).most_common(1)[0]
        stable_visual_occ = set(most_common)
        
        # 6. Move Logic
        move_found = False
        if agreement >= STABILITY_THRESHOLD:
            move = find_best_matching_move(board, stable_visual_occ)
            
            if move:
                print(f"✅ MOVE: {move.uci()}")
                board.push(move)
                move_id += 1
                
                last_valid = warped.copy()
                occupancy_buffer.clear()
                square_stuck_counters.fill(0)
                move_found = True
                
                out = OUT_DIR / f"move_{move_id:03d}"
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out/"board.jpg"), warped)
                draw_virtual_board(board).save(out/"virtual.png")
                (out/"fen.txt").write_text(board.fen())
        
        # 7. Healing
        if not move_found:
            for sq in range(64):
                if sq in changes:
                    square_stuck_counters[sq] += 1
                    if square_stuck_counters[sq] > STUCK_TIMEOUT:
                        last_valid = heal_background(last_valid, warped, sq)
                        square_stuck_counters[sq] = 0
                else:
                    square_stuck_counters[sq] = 0
                    
        # 8. Draw
        debug = draw_hud(warped, board, changes)
        cv2.imshow("Chess AI", debug)
        cv2.imshow("Mask View", edge_debug)
        
        if len(changes) > 0:
            hist_img = draw_histogram_window(warped, changes)
            cv2.imshow("Pixel Histograms", hist_img)
        
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()