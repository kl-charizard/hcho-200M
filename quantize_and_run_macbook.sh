#!/bin/bash
echo "🚀 hcho-200M Quantization and Inference Pipeline"
echo "================================================"

# Activate virtual environment
source hcho_env/bin/activate

# Check if environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Virtual environment not activated"
    echo "Please run setup_macbook.sh first"
    exit 1
fi

echo "✅ Virtual environment activated"

# Check if model exists
if [ ! -f "models/final_model.pt" ]; then
    echo "❌ Trained model not found at models/final_model.pt"
    echo "Please train the model first using run_training_macbook.sh"
    exit 1
fi

echo "✅ Found trained model"

# Create quantized models directory
mkdir -p quantized_models

echo ""
echo "🔄 Quantizing model to GGUF format..."
echo "This will create multiple quantization levels for different use cases"

# Quantize to multiple formats
python3 quantize_gguf.py --model-path models/final_model.pt --all-formats --output-dir quantized_models

if [ $? -ne 0 ]; then
    echo "❌ Quantization failed"
    exit 1
fi

echo ""
echo "✅ Quantization complete!"
echo ""
echo "📊 Available quantized models:"
ls -la quantized_models/*.gguf

echo ""
echo "🚀 Starting interactive chat with hcho-200M Q4_K_M model (best balance of size/quality)..."

# Start interactive chat with Q4_K_M model
python3 inference_gguf.py --model quantized_models/hcho-200m-q4_k_m.gguf --interactive

echo ""
echo "🎉 GGUF pipeline complete!"
