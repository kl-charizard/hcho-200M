#!/usr/bin/env python3
"""
Setup script for LLM training project
Automatically installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("❌ CUDA is not available. Please install CUDA toolkit.")
            return False
    except ImportError:
        print("❌ PyTorch not installed yet. Will check after installation.")
        return False

def main():
    print("🚀 Setting up LLM Training Environment")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        sys.exit(1)
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install PyTorch with CUDA support
    print("\n📦 Installing PyTorch with CUDA support...")
    if platform.system() == "Windows":
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    if not run_command(torch_command, "Installing PyTorch"):
        print("❌ Failed to install PyTorch. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Install other requirements
    if not run_command("pip install -r requirements.txt", "Installing other dependencies"):
        print("❌ Failed to install requirements. Please check requirements.txt")
        sys.exit(1)
    
    # Check CUDA availability
    if not check_cuda():
        print("\n⚠️  CUDA not available. Training will be very slow on CPU.")
        print("   Please install CUDA toolkit for GPU acceleration.")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python train_llm.py --help")
    print("2. Start training: python train_llm.py --config config.yaml")
    print("3. Monitor training: python monitor.py")

if __name__ == "__main__":
    main()
