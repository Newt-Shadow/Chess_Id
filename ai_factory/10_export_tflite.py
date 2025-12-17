import os
import sys
import gc
import time
import torch
import psutil
from ultralytics import YOLO

# ---------------- CONFIG ----------------
PROJECT_NAME = "chess_grandmaster"
ROUND_NAME = "round2_finetune"
DATA_YAML = "datasets/processed/data.yaml"
WEIGHTS_PATH = f"{PROJECT_NAME}/{ROUND_NAME}/weights/best.pt"

RAM_LIMIT_GB = 15.5
CUDA_DEVICE = 0
# --------------------------------------


# ---------- HARD SAFETY GUARDS ----------
def abort(msg):
    print(f"\n❌ ABORT: {msg}")
    sys.exit(1)


def check_gpu_or_abort():
    print("🔍 Checking GPU...")
    if not torch.cuda.is_available():
        abort("CUDA GPU NOT AVAILABLE. Export must run on GPU.")

    gpu = torch.cuda.get_device_name(CUDA_DEVICE)
    vram = torch.cuda.get_device_properties(CUDA_DEVICE).total_memory / 1e9
    print(f"✅ GPU OK: {gpu} ({vram:.1f} GB VRAM)")

    # Hard bind CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)


def limit_tensorflow_cpu():
    """
    Prevent TensorFlow from eating all RAM via CPU parallelism
    """
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def check_tensorflow_or_abort():
    print("🔍 Checking TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow OK: {tf.__version__}")
    except ImportError:
        abort("TensorFlow not installed. Install it first.")


def check_weights_or_abort():
    if not os.path.exists(WEIGHTS_PATH):
        abort(f"best.pt not found:\n{WEIGHTS_PATH}")

    size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
    print(f"✅ Found weights ({size_mb:.1f} MB)")


def get_ram_gb():
    return psutil.virtual_memory().used / 1e9


def ram_guard():
    ram = get_ram_gb()
    if ram > RAM_LIMIT_GB:
        abort(f"RAM exceeded {RAM_LIMIT_GB} GB (current: {ram:.1f} GB)")


# ---------- EXPORT ----------
def export_tflite_safe():
    print("\n📦 Starting SAFE TFLite INT8 export (GPU enforced)...")

    gc.collect()
    torch.cuda.empty_cache()
    ram_guard()

    model = YOLO(WEIGHTS_PATH)

    start = time.time()
    model.export(
        format="tflite",
        half=True,
        device=CUDA_DEVICE
    )

    ram_guard()
    print(f"\n✅ Export completed in {(time.time() - start)/60:.1f} min")


# ---------- MAIN ----------
def main():
    print("\n🚀 GPU-LOCKED POST-TRAINING EXPORT\n")

    limit_tensorflow_cpu()
    check_gpu_or_abort()
    check_tensorflow_or_abort()
    check_weights_or_abort()
    export_tflite_safe()

    print("\n🏁 DONE — GPU USED, RAM PROTECTED, NO TRAINING TOUCHED")


if __name__ == "__main__":
    main()
