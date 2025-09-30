#!/usr/bin/env python3
"""
Google Colab Setup Script for hcho-200M
Automatically sets up the environment and starts training
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and print the result"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("⚠️  No GPU detected. Training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def main():
    print("🚀 Setting up hcho-200M on Google Colab")
    print("=" * 50)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("⚠️  Not running in Google Colab - some features may not work")
    
    # Install PyTorch with CUDA
    if not run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "Installing PyTorch with CUDA support"
    ):
        return False
    
    # Install other dependencies
    if not run_command(
        "pip install transformers datasets accelerate tokenizers tqdm pyyaml",
        "Installing other dependencies"
    ):
        return False
    
    # Check GPU availability
    check_gpu()
    
    # Clone repository if not already present
    if not os.path.exists("hcho-200M"):
        if not run_command(
            "git clone https://github.com/kl-charizard/hcho-200M.git",
            "Cloning repository"
        ):
            return False
        os.chdir("hcho-200M")
    else:
        print("✅ Repository already exists")
        os.chdir("hcho-200M")
    
    # Test data loader
    print("🔄 Testing data loader...")
    if run_command("python data_loader.py", "Testing data loader"):
        print("✅ Data loader test passed")
    else:
        print("⚠️  Data loader test failed, but continuing...")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python train_llm.py --config config.yaml")
    print("2. Monitor: python monitor.py (in another cell)")
    print("3. Quantize: python quantize_gguf.py --model-path models/final_model.pt --all-formats")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
