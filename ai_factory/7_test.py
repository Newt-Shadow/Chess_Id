import os
import cv2
import gc
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw
from itertools import combinations
from flask import Flask, request, jsonify, render_template_string
import threading
import webbrowser


# ======================= CONFIG =======================
ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "simple_output" / "images"
OUT_DIR = ROOT / "data-set"
ICON_DIR = ROOT / "assets" / "chess_pieces"
MODEL_PATH = ROOT / "chess_grandmaster/round2_finetune/weights/best.pt"

CONF = 0.35
BOARD_PIXELS = 400
BATCH_SIZE = 8
DEVICE = 0   # CUDA:0
MANUAL_DONE = False


torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    'bB','bK','bN','bP','bQ','bR',
    'wB','wK','wN','wP','wQ','wR'
]
ICON_MAP = {c: f"{c}.png" for c in CLASSES}

# ================= BOARD TRACKING STATE =================
INITIAL_CORNERS = None
PREV_GRAY = None
TRACKED_CORNERS = None

# ======================================================
# ---------------- MANUAL CORNER PICK ------------------
# ======================================================
def select_board_corners_once(img, save_path):
    cv2.imwrite(str(save_path), img)

    print("\n🖼️ First image saved for manual corner selection:")
    print(save_path)
    print("Open this image and ENTER 4 corners as: x y")
    print("Order: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT\n")

    points = []
    for name in ["TL", "TR", "BR", "BL"]:
        while True:
            try:
                x, y = map(int, input(f"{name} corner (x y): ").split())
                points.append((x, y))
                break
            except:
                print("Invalid input. Enter as: x y")

    return np.array(points, dtype=np.float32)

# ======================================================
# ---------------- CORNER TRACKING ---------------------
# ======================================================
def track_corners(prev_gray, curr_gray, prev_pts):
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    if next_pts is None or status.sum() < 4:
        return None
    return next_pts

# ======================================================
# ---------------- GEOMETRY LOGIC ----------------------
# ======================================================
def estimate_square_size(centers):
    dists = []
    for (x1,y1),(x2,y2) in combinations(centers,2):
        d = np.hypot(x1-x2, y1-y2)
        if d > 10:
            dists.append(d)
    return np.percentile(dists, 20)

def infer_board_from_pieces(detections):
    if len(detections) < 4:
        return {}

    centers = np.array([pt for _, pt in detections])
    mean = centers.mean(axis=0)
    centered = centers - mean
    _, _, vt = np.linalg.svd(centered)

    axis_x, axis_y = vt[0], vt[1]
    square = estimate_square_size(centers)
    if square <= 0 or np.isnan(square):
        return {}

    board = {}
    for piece, (x, y) in detections:
        v = np.array([x, y]) - mean
        col = int(np.round((v @ axis_x) / square)) + 4
        row = int(np.round((v @ axis_y) / square)) + 4
        if 0 <= col < 8 and 0 <= row < 8:
            board[f"{chr(97+col)}{row+1}"] = piece

    return board

def draw_virtual_board(fen):
    sq = BOARD_PIXELS // 8
    img = Image.new("RGB", (BOARD_PIXELS, BOARD_PIXELS), "white")
    draw = ImageDraw.Draw(img)

    for r in range(8):
        for c in range(8):
            color = "lightgray" if (r+c)%2 == 0 else "darkgray"
            draw.rectangle(
                [c*sq, (7-r)*sq, (c+1)*sq, (8-r)*sq],
                fill=color
            )

    for pos, piece in fen.items():
        icon = Image.open(ICON_DIR / ICON_MAP[piece]).resize((sq, sq))
        c = ord(pos[0]) - 97
        r = int(pos[1]) - 1
        img.paste(icon, (c*sq, (7-r)*sq), icon)

    return img

# ======================================================
# ---------------- DATA HANDLING -----------------------
# ======================================================
def collect_images():
    imgs = []
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                imgs.append(Path(root) / f)
    return sorted(imgs)


def get_corners_via_web(image_path):
    app = Flask(__name__)
    corners = []

    HTML = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Select Board Corners</title>
        <style>
            body { text-align: center; font-family: sans-serif; }
            canvas { border: 2px solid black; cursor: crosshair; }
        </style>
    </head>
    <body>
        <h2>Click 4 corners: TL → TR → BR → BL</h2>
        <canvas id="canvas"></canvas>
        <p id="info"></p>

        <script>
            const img = new Image();
            img.src = "/image";
            const canvas = document.getElementById("canvas");
            const ctx = canvas.getContext("2d");
            const points = [];

            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
            };

            canvas.addEventListener("click", (e) => {
                if (points.length >= 4) return;
                const rect = canvas.getBoundingClientRect();
                const x = Math.round(e.clientX - rect.left);
                const y = Math.round(e.clientY - rect.top);
                points.push({x, y});

                ctx.fillStyle = "red";
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fill();

                document.getElementById("info").innerText =
                    `Selected ${points.length}/4`;

                if (points.length === 4) {
                    fetch("/submit", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify(points)
                    });
                }
            });
        </script>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/image")
    def image():
        with open(image_path, "rb") as f:
            return f.read(), 200, {"Content-Type": "image/jpeg"}

    @app.route("/submit", methods=["POST"])
    def submit():
        nonlocal corners
        corners = request.json
        shutdown()
        return jsonify({"status": "ok"})

    def shutdown():
        func = request.environ.get("werkzeug.server.shutdown")
        if func:
            func()

    def run():
        app.run(port=5000, debug=False)

    threading.Thread(target=run).start()
    webbrowser.open("http://127.0.0.1:5000")

    print("🌐 Web app opened. Click 4 corners in browser...")
    while len(corners) < 4:
        pass

    return np.array([(p["x"], p["y"]) for p in corners], dtype=np.float32)

# ======================================================
# ---------------- GPU PIPELINE ------------------------
# ======================================================
def process_batches():
    global INITIAL_CORNERS, PREV_GRAY, TRACKED_CORNERS, MANUAL_DONE

    print("\n🔥 GPU PIPELINE (MANUAL FIRST FRAME + AUTO TRACKING)\n")

    model = YOLO(MODEL_PATH)
    model.to("cuda")
    model.eval()

    images = collect_images()
    if not images:
        raise RuntimeError(f"No images found in {RAW_DIR}")

    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        print(f"🚀 Batch {i//BATCH_SIZE + 1}")

        for idx, img_path in enumerate(batch):
            try:
                out = OUT_DIR / f"img_{i+idx:05d}"
                out.mkdir(parents=True, exist_ok=True)

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # ---------- FIRST IMAGE: MANUAL ----------
                if not MANUAL_DONE:
                    ref_img_path = out / "manual_corner_reference.jpg"
                    cv2.imwrite(str(ref_img_path), img)
                    INITIAL_CORNERS = get_corners_via_web(str(ref_img_path))


                    if INITIAL_CORNERS is None or len(INITIAL_CORNERS) != 4:
                        raise RuntimeError("Corner selection failed on first image")

                    TRACKED_CORNERS = INITIAL_CORNERS.reshape(-1, 1, 2)
                    PREV_GRAY = gray.copy()
                    MANUAL_DONE = True


                # ---------- NEXT IMAGES: TRACK ----------
                else:
                    TRACKED_CORNERS = track_corners(
                        PREV_GRAY, gray, TRACKED_CORNERS
                    )
                    if TRACKED_CORNERS is None:
                        print("⚠️ Tracking lost — skipping image (no re-popup)")
                        PREV_GRAY = gray.copy()
                        continue

                    PREV_GRAY = gray.copy()

                # visualize tracked corners
                vis = img.copy()
                for (x, y) in TRACKED_CORNERS.reshape(-1, 2):
                    cv2.circle(vis, (int(x), int(y)), 6, (0,255,0), -1)
                cv2.imwrite(str(out / "tracked_corners.jpg"), vis)

                # ---------- YOLO PIECE DETECTION ----------
                res = model.predict(
                    img, conf=CONF, device=DEVICE, verbose=False
                )[0]
                cv2.imwrite(str(out / "detected_pieces.jpg"), res.plot())

                detections = []
                for x1,y1,x2,y2,conf,cls in res.boxes.data.tolist():
                    piece = CLASSES[int(cls)]
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    detections.append((piece, (cx, cy)))

                if detections:
                    fen = infer_board_from_pieces(detections)
                    if fen:
                        draw_virtual_board(fen).save(out / "virtual_board.png")

            except Exception as e:
                print(f"⚠️ Failed on {img_path.name}: {e}")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

# ======================================================
if __name__ == "__main__":
    process_batches()
    print("\n✅ DONE — MANUAL FIRST FRAME + AUTO BOARD TRACKING COMPLETE")
