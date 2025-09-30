#!/usr/bin/env python3
"""
Debug script for hcho-200M training issues
Helps diagnose CUDA errors and data problems
"""

import torch
import yaml
from data_loader import LLMDataLoader
from model import create_model

def debug_data_loader():
    """Debug the data loader for issues"""
    print("🔍 Debugging Data Loader...")
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create data loader
    data_loader = LLMDataLoader(config)
    data_loader.setup_tokenizer()
    
    print(f"✅ Tokenizer vocab size: {data_loader.tokenizer.vocab_size}")
    print(f"✅ Pad token ID: {data_loader.tokenizer.pad_token_id}")
    print(f"✅ EOS token ID: {data_loader.tokenizer.eos_token_id}")
    
    # Test data loading
    try:
        train_dataset, eval_dataset = data_loader.load_datasets()
        print(f"✅ Datasets loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Test a few samples
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            input_ids = sample['input_ids']
            print(f"Sample {i}: input_ids shape={len(input_ids)}, min={min(input_ids)}, max={max(input_ids)}")
            
            # Check for invalid token IDs
            invalid_ids = [tid for tid in input_ids if tid < 0 or tid >= data_loader.tokenizer.vocab_size]
            if invalid_ids:
                print(f"❌ Invalid token IDs found: {invalid_ids[:10]}...")
            else:
                print(f"✅ All token IDs valid")
        
        # Test dataloader
        train_loader, eval_loader = data_loader.create_dataloaders()
        print(f"✅ DataLoaders created: {len(train_loader)} train batches, {len(eval_loader)} eval batches")
        
        # Test a batch
        batch = next(iter(train_loader))
        print(f"✅ Batch test: input_ids shape={batch['input_ids'].shape}, dtype={batch['input_ids'].dtype}")
        print(f"   Min token ID: {batch['input_ids'].min()}, Max token ID: {batch['input_ids'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_model():
    """Debug the model for issues"""
    print("\n🔍 Debugging Model...")
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    try:
        # Create model
        model = create_model(config)
        print(f"✅ Model created: {model.count_parameters():,} parameters")
        
        # Test with dummy data
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config['model']['vocab_size'], (batch_size, seq_len))
        
        print(f"✅ Test input: shape={input_ids.shape}, min={input_ids.min()}, max={input_ids.max()}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"✅ Forward pass successful: logits shape={outputs['logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_cuda():
    """Debug CUDA setup"""
    print("\n🔍 Debugging CUDA...")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test basic CUDA operations
        try:
            x = torch.randn(2, 3).cuda()
            y = x + 1
            print("✅ Basic CUDA operations work")
        except Exception as e:
            print(f"❌ CUDA operations failed: {e}")
            return False
    
    return True

def main():
    print("🚀 hcho-200M Debug Script")
    print("=" * 50)
    
    # Run all debug checks
    cuda_ok = debug_cuda()
    data_ok = debug_data_loader()
    model_ok = debug_model()
    
    print("\n📊 Debug Summary:")
    print(f"CUDA: {'✅' if cuda_ok else '❌'}")
    print(f"Data Loader: {'✅' if data_ok else '❌'}")
    print(f"Model: {'✅' if model_ok else '❌'}")
    
    if all([cuda_ok, data_ok, model_ok]):
        print("\n🎉 All checks passed! Training should work.")
    else:
        print("\n⚠️  Some issues found. Check the errors above.")
        print("\n💡 Suggestions:")
        print("- Set CUDA_LAUNCH_BLOCKING=1 for detailed CUDA errors")
        print("- Check if token IDs are within vocabulary range")
        print("- Verify dataset preprocessing is working correctly")

if __name__ == "__main__":
    main()
