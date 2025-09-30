# hcho-100M - Lightweight LLM Training

A complete implementation for training hcho-100M, a 100-million parameter Large Language Model optimized for MacBook, Google Colab, and consumer hardware with massive dataset support.

## ✨ Features

- **🚀 Multiple Platforms**: MacBook, Google Colab, Windows, Linux
- **📊 Massive Dataset**: 179 million tokens from 6+ diverse datasets with smart standardization
- **⚡ GPU Acceleration**: Apple Silicon MPS, CUDA, CPU fallback
- **📈 Progress Monitoring**: Real-time progress bars with speed and loss
- **🔧 Easy Setup**: One-click installation and training
- **💾 Model Quantization**: GGUF support for efficient deployment
- **🎯 Optimized**: 2.0 tokens per parameter for effective training
- **📱 Mobile Ready**: Quantized models for mobile deployment

## 🚀 Quick Start

### MacBook Setup (Recommended)

1. **Run the setup script:**
   ```bash
   ./setup_macbook.sh
   ```

2. **Start training:**
   ```bash
   ./run_training_macbook.sh
   ```

3. **Quantize and run inference:**
   ```bash
   ./quantize_and_run_macbook.sh
   ```

4. **Monitor training:**
   ```bash
   python monitor.py
   ```

### Windows Setup

1. **Run the setup script:**
   ```bash
   setup_windows.bat
   ```

2. **Start training:**
   ```bash
   run_training.bat
   ```

3. **Quantize and run inference:**
   ```bash
   quantize_and_run.bat
   ```

4. **Monitor training:**
   ```bash
   python monitor.py
   ```

### Google Colab Setup (Recommended for GPU Training)

1. **Open the Colab notebook:**
   - Click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kl-charizard/hcho-200M/blob/main/hcho_200m_colab.ipynb)
   - Or create a new notebook and copy the code below

2. **Enable GPU:**
   - Runtime → Change runtime type → GPU → T4 (free) or V100/A100 (Pro)

3. **Run the setup cell:**
   ```python
   # Install dependencies
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   !pip install transformers datasets accelerate tokenizers tqdm pyyaml
   
   # Clone repository
   !git clone https://github.com/kl-charizard/hcho-200M.git
   %cd hcho-200M
   ```

4. **Start training:**
   ```python
   # Run training with GPU acceleration
   !python train_llm.py --config config.yaml
   ```

5. **Monitor training:**
   ```python
   # Monitor in real-time
   !python monitor.py
   ```

### Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv llm_env
   llm_env\Scripts\activate  # Windows
   source llm_env/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   python setup.py
   ```

3. **Start training:**
   ```bash
   python train_llm.py --config config.yaml
   ```

## 📁 Project Structure

```
hcho-200M/
├── config.yaml              # Training configuration (100M model)
├── ds_config.json           # DeepSpeed configuration
├── requirements.txt         # Python dependencies (optimized)
├── setup.py                 # Automatic setup script
├── setup_macbook.sh         # MacBook setup script
├── setup_colab.py           # Google Colab setup script
├── run_training_macbook.sh  # MacBook training script
├── quantize_and_run_macbook.sh # MacBook quantization and inference
├── train_llm.py            # Main training script
├── model.py                 # 100M parameter model implementation
├── data_loader.py           # Smart dataset loading with standardization
├── quantize_gguf.py         # GGUF quantization script
├── inference_gguf.py        # GGUF model inference
├── monitor.py               # Training monitoring
├── hcho_200m_colab.ipynb   # Google Colab notebook
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## 🏗️ Model Architecture

- **Parameters**: ~100 million (optimized for MacBook & Colab)
- **Architecture**: GPT-style transformer
- **Hidden Size**: 576
- **Layers**: 10
- **Attention Heads**: 9
- **Context Length**: 512 tokens (memory optimized)
- **Vocabulary Size**: 50,257

## 💾 Memory Optimizations

The implementation includes several memory optimization techniques for MacBook and consumer hardware:

- **CPU/GPU Training**: Optimized for Apple Silicon and Intel Macs
- **Efficient Batching**: Larger batch sizes for faster training
- **Memory Management**: Automatic memory optimization
- **Accelerate**: Simplified training without heavy dependencies
- **Lightweight Dependencies**: Removed heavy GPU-specific libraries

## 📊 Datasets

The training script automatically loads multiple large datasets with smart feature standardization:

- **WikiText-103**: 100,000 samples (main dataset)
- **SQuAD**: 50,000 question-answer pairs (combined as text)
- **AG News**: 50,000 news articles
- **Yelp Reviews**: 50,000 customer reviews
- **Amazon Polarity**: 50,000 product reviews
- **GLUE SST-2**: 50,000 sentiment analysis texts
- **Fallback**: 50,000 sample texts if online datasets fail

**Total Training Data**: ~350,000 samples × 512 tokens = **~179 million tokens** (2.0 tokens per parameter)

**Note**: All datasets are automatically standardized to have consistent text format, handling different field names and combining question-answer pairs intelligently.

## ⚙️ Configuration

Edit `config.yaml` to customize:

- Model architecture parameters
- Training hyperparameters
- Memory optimization settings
- Dataset configuration
- Logging options

## 📈 Monitoring

The `monitor.py` script provides real-time monitoring of:

- GPU memory usage
- GPU utilization
- CPU and RAM usage
- Training loss
- Learning rate

## 🔧 GGUF Quantization

The system includes comprehensive GGUF quantization support:

### **Quantization Formats:**
- **Q4_K_M**: 4-bit medium quality (recommended)
- **Q4_K_S**: 4-bit small size
- **Q5_K_M**: 5-bit medium quality
- **Q5_K_S**: 5-bit small size
- **Q6_K**: 6-bit quality
- **Q8_0**: 8-bit quality
- **F16**: 16-bit float
- **F32**: 32-bit float

### **Usage:**

**Quantize trained model:**
```bash
python quantize_gguf.py --model-path models/final_model.pt --all-formats
```

**Run inference:**
```bash
python inference_gguf.py --model quantized_models/hcho-q4_k_m.gguf --interactive
```

**Single generation:**
```bash
python inference_gguf.py --model quantized_models/hcho-q4_k_m.gguf --prompt "The future of AI is"
```

### **Benefits:**
- **Smaller file sizes**: Up to 10x compression
- **Faster inference**: Optimized for CPU/GPU
- **Lower memory usage**: Perfect for deployment
- **Multiple formats**: Choose quality vs size trade-off

## 🔧 Requirements

### MacBook Training
- **Hardware**: MacBook Air/Pro (M1/M2/M3) or Intel Mac
- **RAM**: 8GB+ recommended (16GB+ for faster training)
- **Storage**: 15GB+ free space (for large datasets)
- **Python**: 3.8+
- **macOS**: 10.15+ (Catalina or later)

### Google Colab Training
- **Hardware**: Free GPU (T4) or Pro GPU (V100/A100)
- **RAM**: 12GB+ (free) or 25GB+ (Pro)
- **Storage**: 15GB+ free space
- **Runtime**: GPU enabled

## 🚨 Important Notes

1. **Training Time**: 
   - MacBook: 2-6 hours
   - Colab T4 (free): 3-4 hours
   - Colab V100 (Pro): 1-2 hours
   - Colab A100 (Pro): 45-60 minutes
2. **Dataset Size**: 179 million tokens (2.0 tokens per parameter)
3. **Memory Management**: Monitor RAM usage during training
4. **Checkpointing**: Regular checkpoints prevent data loss

## 🛠️ Troubleshooting

### Out of Memory Errors
- Reduce batch size in `config.yaml`
- Enable CPU offloading
- Use gradient checkpointing
- Reduce sequence length

### Slow Training
- Ensure CUDA is properly installed
- Check GPU utilization
- Use mixed precision training
- Optimize data loading

### Dataset Issues
- Check internet connection
- Verify dataset availability
- Use sample datasets for testing
- **Feature alignment errors**: Fixed automatically with smart dataset standardization
- **SQuAD format**: Question-answer pairs are automatically combined into text format

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📞 Support

For questions and support, please open an issue on GitHub.
