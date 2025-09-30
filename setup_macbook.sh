#!/bin/bash
echo "🚀 Setting up hcho-100M on MacBook"
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
echo ""
echo "🔄 Creating virtual environment..."
python3 -m venv hcho_env
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source hcho_env/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support (Apple Silicon)
echo ""
echo "🔄 Installing PyTorch with Apple Silicon support..."
if [[ $(uname -m) == "arm64" ]]; then
    echo "   Detected Apple Silicon (M1/M2/M3)"
    pip install torch torchvision torchaudio
else
    echo "   Detected Intel Mac"
    pip install torch torchvision torchaudio
fi

# Install other requirements
echo ""
echo "🔄 Installing other requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

# Check PyTorch installation
echo ""
echo "🔄 Checking PyTorch installation..."
python3 -c "
import torch
print('✅ PyTorch version:', torch.__version__)
print('✅ MPS available:', torch.backends.mps.is_available())
print('✅ MPS built:', torch.backends.mps.is_built())
if torch.backends.mps.is_available():
    print('✅ Apple Silicon GPU acceleration enabled')
else:
    print('⚠️  Using CPU (still fast for 200M model)')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source hcho_env/bin/activate"
echo "2. Start training: python train_llm.py --config config.yaml"
echo "3. Monitor training: python monitor.py"
echo ""
echo "Training time: 2-6 hours on MacBook"
echo "Model size: ~200M parameters"
