import sys
import torch

def check_system():
    print("🔍 DIAGNOSTICS...")
    if not torch.cuda.is_available():
        print("❌ CRITICAL: NO GPU FOUND. STOPPING.")
        sys.exit(1)
    
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {name} | VRAM: {vram:.1f}GB")
    
    if vram < 5.0:
        print("⚠️ WARNING: VRAM < 5GB. Script 3 will force Batch=8.")
    
    print("🚀 SYSTEM READY.")

if __name__ == "__main__":
    check_system()