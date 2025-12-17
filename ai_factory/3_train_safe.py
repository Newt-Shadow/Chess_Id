import sys
import os
import torch
import gc
import psutil  # Needs: pip install psutil
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURATION ---
DATA_YAML = 'datasets/processed/data.yaml'
MODEL_SIZE = 'yolov8s.pt' 
PROJECT_NAME = 'chess_grandmaster'

# RAM THRESHOLD (in GB) - Script triggers emergency cleanup if usage exceeds this
RAM_LIMIT_GB = 13.5 

# --- COLOR CODES ---
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# --- RAM GUARDIAN ---
def get_ram_usage():
    """Returns used RAM in GB"""
    return psutil.virtual_memory().used / 1e9

def ram_guardian_callback(trainer):
    """
    This runs inside YOLO every epoch to protect your system.
    """
    current_ram = get_ram_usage()
    if current_ram > RAM_LIMIT_GB:
        print(f"\n{RED}⚠️  RAM ALERT: Usage at {current_ram:.1f}GB (Threshold: {RAM_LIMIT_GB}GB){RESET}")
        print(f"{YELLOW}🧹 Triggering Emergency Garbage Collection...{RESET}")
        
        # 1. Python GC
        gc.collect()
        # 2. PyTorch CUDA Cache
        torch.cuda.empty_cache()
        # 3. System Sync (Linux specific, helps sometimes)
        if os.name == 'posix':
            os.sync()
            
        new_ram = get_ram_usage()
        print(f"{GREEN}✅ RAM Stabilized to {new_ram:.1f}GB{RESET}")

def check_system():
    print(f"\n{CYAN}🔍 SYSTEM DIAGNOSTIC...{RESET}")
    if not torch.cuda.is_available():
        print(f"\n{RED}❌ ABORTING: NO GPU FOUND.{RESET}")
        sys.exit(1)
    
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{GREEN}✅ GPU DETECTED: {torch.cuda.get_device_name(0)} ({vram:.1f} GB){RESET}")
    
    # Conservative Batch size to balance the RAM load
    batch = 12 if vram > 5.5 else 8
    print(f"{CYAN}⚙️  Safe Batch Size: {batch}{RESET}")
    return batch

def get_resume_status(round_name):
    """
    Checks if a training round was interrupted and can be resumed.
    Returns: (should_resume, weight_path)
    """
    base_dir = Path(PROJECT_NAME) / round_name / 'weights'
    last_pt = base_dir / 'last.pt'
    best_pt = base_dir / 'best.pt'

    # Case 1: Interrupted run exists (last.pt)
    if last_pt.exists():
        print(f"{YELLOW}🔄 DETECTED INTERRUPTED RUN: {round_name}{RESET}")
        print(f"   Resuming from: {last_pt}")
        return True, str(last_pt)
    
    # Case 2: Round already finished (best.pt exists but training script called again)
    if best_pt.exists():
        print(f"{GREEN}✅ {round_name} ALREADY COMPLETED.{RESET}")
        return False, str(best_pt)
    
    # Case 3: Fresh start
    return False, None

def train_round_1(batch_size):
    ROUND_NAME = 'round1_base'
    print(f"\n{YELLOW}🥊 ROUND 1: STANDARD TRAINING{RESET}")
    
    resume, weight_path = get_resume_status(ROUND_NAME)
    
    # If already done, return the best weight
    if not resume and weight_path: 
        return weight_path

    # If resuming, load 'last.pt'. If fresh, load 'yolov8s.pt'
    model_to_load = weight_path if resume else MODEL_SIZE
    model = YOLO(model_to_load)

    # Attach RAM Guardian
    model.add_callback("on_train_epoch_end", ram_guardian_callback)

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=batch_size,
        device=0,
        
        # --- CRITICAL RAM SETTINGS ---
        workers=4,                # Low workers prevents RAM explosion
        cache=False,              # Disk-based loading
        # persistent_workers=False, # Shut down workers between epochs (Saves RAM)
        
        project=PROJECT_NAME,
        name=ROUND_NAME,
        resume=resume,            # <--- MAGIC KEYWORD
        exist_ok=True if resume else False,
        
        # Augmentation
        degrees=10.0,
        hsv_h=0.015,
        mosaic=1.0,
        verbose=True
    )
    
    return f"{PROJECT_NAME}/{ROUND_NAME}/weights/best.pt"

def train_round_2(round1_weights, batch_size):
    ROUND_NAME = 'round2_finetune'
    print(f"\n{YELLOW}🥊 ROUND 2: VALIDATION FINE-TUNING{RESET}")
    
    resume, weight_path = get_resume_status(ROUND_NAME)
    
    # If already done
    if not resume and weight_path:
        return weight_path
    
    # If resuming, load 'last.pt'. If fresh, load Round 1 Best
    model_to_load = weight_path if resume else round1_weights
    
    # Check if Round 1 actually existed before trying to load it
    if not os.path.exists(model_to_load):
        print(f"{RED}❌ ERROR: Logic Flaw. Round 1 weights missing: {model_to_load}{RESET}")
        sys.exit(1)

    model = YOLO(model_to_load)
    model.add_callback("on_train_epoch_end", ram_guardian_callback)

    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=batch_size,
        device=0,
        
        workers=4,
        cache=False,
        # persistent_workers=False,
        
        project=PROJECT_NAME,
        name=ROUND_NAME,
        resume=resume,
        exist_ok=True if resume else False,
        
        lr0=0.001,
        lrf=0.01,
        verbose=True
    )
    
    return f"{PROJECT_NAME}/{ROUND_NAME}/weights/best.pt"

def final_verification(final_weights):
    print(f"\n{YELLOW}🏆 FINAL VERIFICATION{RESET}")
    if not os.path.exists(final_weights):
        print(f"{RED}❌ Cannot verify. File missing: {final_weights}{RESET}")
        return

    model = YOLO(final_weights)
    # Force garbage collect before validation to free space
    gc.collect()
    torch.cuda.empty_cache()
    
    metrics = model.val(data=DATA_YAML, split='val')
    print(f"\n{GREEN}   FINAL mAP@50: {metrics.box.map50:.2%}{RESET}")

def main():
    try:
        # Install psutil if missing
        try:
            import psutil
        except ImportError:
            print(f"{RED}⚠️  Missing 'psutil'. Installing...{RESET}")
            os.system(f"{sys.executable} -m pip install psutil")
            import psutil

        gc.collect()
        batch = check_system()
        
        # --- EXECUTION FLOW ---
        # The script intelligently decides whether to start, resume, or skip each round.
        
        best_w1 = train_round_1(batch)
        print(f"{CYAN}✅ Round 1 Milestone Reached.{RESET}")
        
        best_w2 = train_round_2(best_w1, batch)
        print(f"{CYAN}✅ Round 2 Milestone Reached.{RESET}")
        
        final_verification(best_w2)
        
        print(f"\n{CYAN}📦 Exporting TFLite...{RESET}")
        model = YOLO(best_w2)
        model.export(format='tflite', int8=True, data=DATA_YAML)
        print(f"{GREEN}✅ DONE. Safe and Sound.{RESET}")

    except KeyboardInterrupt:
        print(f"\n{RED}🛑 Interrupted by User. Progress saved in 'last.pt'.{RESET}")
        print(f"{CYAN}   Run this script again to RESUME exactly here.{RESET}")
    except Exception as e:
        print(f"\n{RED}❌ CRASH: {e}{RESET}")
        print(f"{CYAN}   Don't worry. Fix the error and re-run. It will RESUME.{RESET}")

if __name__ == "__main__":
    main()