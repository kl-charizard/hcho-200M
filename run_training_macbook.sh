#!/bin/bash
echo "🚀 Starting hcho-200M Training on MacBook"
echo "========================================"

# Activate virtual environment
source hcho_env/bin/activate

# Check if environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Virtual environment not activated"
    echo "Please run setup_macbook.sh first"
    exit 1
fi

echo "✅ Virtual environment activated"

# Check PyTorch and MPS
echo "🔄 Checking PyTorch and Apple Silicon support..."
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
if torch.backends.mps.is_available():
    print('✅ Apple Silicon GPU acceleration enabled')
else:
    print('⚠️  Using CPU (still fast for 200M model)')
"

# Start training
echo ""
echo "🚀 Starting training..."
echo "Expected training time: 2-6 hours"
echo "Model size: ~200M parameters"
echo "Press Ctrl+C to stop training"
echo ""

python3 train_llm.py --config config.yaml

echo ""
echo "Training completed!"
echo "Check the 'models' folder for your trained model"
