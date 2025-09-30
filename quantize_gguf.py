"""
GGUF Quantization Script
Converts trained LLM model to GGUF format with various quantization levels
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import gguf

logger = logging.getLogger(__name__)

class GGUFQuantizer:
    """Quantize LLM model to GGUF format"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.tokenizer = None
        self.config = None
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_model(self):
        """Load the trained model"""
        logger.info(f"🔄 Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if os.path.isdir(self.model_path):
                # HuggingFace format
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="cpu"
                )
            else:
                # PyTorch checkpoint
                from model import create_model
                self.model = create_model(self.config)
                checkpoint = torch.load(self.model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint)
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def quantize_to_gguf(self, output_path: str, quantization_type: str = "Q4_K_M"):
        """Convert model to GGUF format with quantization"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"🔄 Converting model to GGUF format with {quantization_type} quantization")
        
        # Create GGUF writer
        writer = gguf.GGUFWriter(output_path, "hcho-200m")
        
        # Add model metadata
        writer.add_name("hcho-200m")
        writer.add_description("hcho-200M - 200M Parameter Large Language Model")
        writer.add_file_type(gguf.GGMLQuantizationType[quantization_type])
        
        # Add model configuration
        model_config = self.config['model']
        writer.add_context_length(model_config['max_position_embeddings'])
        writer.add_embedding_length(model_config['hidden_size'])
        writer.add_block_count(model_config['num_layers'])
        writer.add_feed_forward_length(model_config['intermediate_size'])
        writer.add_head_count(model_config['num_heads'])
        writer.add_head_count_kv(model_config['num_heads'])
        writer.add_layer_norm_rms_eps(1e-6)
        writer.add_rope_freq_base(10000.0)
        writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        
        # Add vocabulary
        vocab_size = len(self.tokenizer)
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(self.tokenizer.get_vocab().keys())
        writer.add_token_types([gguf.TokenType.NORMAL] * vocab_size)
        
        # Add special tokens
        writer.add_bos_token_id(self.tokenizer.bos_token_id or 0)
        writer.add_eos_token_id(self.tokenizer.eos_token_id or 0)
        writer.add_pad_token_id(self.tokenizer.pad_token_id or 0)
        
        # Convert model weights
        self._convert_weights(writer, quantization_type)
        
        # Write the file
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        
        logger.info(f"✅ GGUF model saved: {output_path}")
        
        # Print file size
        file_size = os.path.getsize(output_path) / (1024**3)
        logger.info(f"   File size: {file_size:.2f} GB")
    
    def _convert_weights(self, writer: gguf.GGUFWriter, quantization_type: str):
        """Convert PyTorch weights to GGUF format"""
        
        model_config = self.config['model']
        hidden_size = model_config['hidden_size']
        num_layers = model_config['num_layers']
        num_heads = model_config['num_heads']
        intermediate_size = model_config['intermediate_size']
        vocab_size = model_config['vocab_size']
        
        # Token embeddings
        token_embeddings = self.model.token_embedding.weight.data
        writer.add_tensor("token_embd.weight", token_embeddings)
        
        # Position embeddings
        position_embeddings = self.model.position_embedding.weight.data
        writer.add_tensor("position_embd.weight", position_embeddings)
        
        # Transformer layers
        for layer_idx in range(num_layers):
            layer = self.model.layers[layer_idx]
            
            # Attention weights
            prefix = f"blk.{layer_idx}"
            
            # Q, K, V projections
            q_proj = layer.attention.q_proj.weight.data
            k_proj = layer.attention.k_proj.weight.data
            v_proj = layer.attention.v_proj.weight.data
            
            writer.add_tensor(f"{prefix}.attn_q.weight", q_proj)
            writer.add_tensor(f"{prefix}.attn_k.weight", k_proj)
            writer.add_tensor(f"{prefix}.attn_v.weight", v_proj)
            
            # Output projection
            out_proj = layer.attention.out_proj.weight.data
            writer.add_tensor(f"{prefix}.attn_output.weight", out_proj)
            
            # Feed-forward weights
            ff1 = layer.feed_forward.linear1.weight.data
            ff2 = layer.feed_forward.linear2.weight.data
            
            writer.add_tensor(f"{prefix}.ffn_up.weight", ff1)
            writer.add_tensor(f"{prefix}.ffn_down.weight", ff2)
            
            # Layer norms
            ln1 = layer.ln1.weight.data
            ln2 = layer.ln2.weight.data
            
            writer.add_tensor(f"{prefix}.attn_norm.weight", ln1)
            writer.add_tensor(f"{prefix}.ffn_norm.weight", ln2)
        
        # Final layer norm
        ln_f = self.model.ln_f.weight.data
        writer.add_tensor("output_norm.weight", ln_f)
        
        # Language modeling head
        lm_head = self.model.lm_head.weight.data
        writer.add_tensor("output.weight", lm_head)
    
    def quantize_multiple_formats(self, output_dir: str):
        """Create multiple quantization formats"""
        
        quantization_types = [
            ("Q4_K_M", "4-bit medium quality"),
            ("Q4_K_S", "4-bit small size"),
            ("Q5_K_M", "5-bit medium quality"),
            ("Q5_K_S", "5-bit small size"),
            ("Q6_K", "6-bit quality"),
            ("Q8_0", "8-bit quality"),
            ("F16", "16-bit float"),
            ("F32", "32-bit float")
        ]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔄 Creating {len(quantization_types)} quantization formats")
        
        for quant_type, description in quantization_types:
            logger.info(f"🔄 Creating {quant_type} ({description})")
            
            output_file = output_path / f"hcho-200m-{quant_type.lower()}.gguf"
            
            try:
                self.quantize_to_gguf(str(output_file), quant_type)
                
                # Print compression ratio
                original_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
                compressed_size = os.path.getsize(output_file) / (1024**3)
                compression_ratio = original_size / compressed_size
                
                logger.info(f"✅ {quant_type}: {compressed_size:.2f} GB (compression: {compression_ratio:.1f}x)")
                
            except Exception as e:
                logger.error(f"❌ Failed to create {quant_type}: {e}")
                continue
        
        logger.info("🎉 Quantization complete!")

def main():
    parser = argparse.ArgumentParser(description="Quantize LLM to GGUF format")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output-path", type=str, help="Output GGUF file path")
    parser.add_argument("--output-dir", type=str, help="Output directory for multiple formats")
    parser.add_argument("--quantization", type=str, default="Q4_K_M", 
                       choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16", "F32"],
                       help="Quantization type")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--all-formats", action="store_true", help="Create all quantization formats")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create quantizer
    quantizer = GGUFQuantizer(args.model_path, args.config)
    
    # Load model
    quantizer.load_model()
    
    if args.all_formats:
        # Create all formats
        output_dir = args.output_dir or "quantized_models"
        quantizer.quantize_multiple_formats(output_dir)
    else:
        # Create single format
        if not args.output_path:
            args.output_path = f"hcho-200m-{args.quantization.lower()}.gguf"
        
        quantizer.quantize_to_gguf(args.output_path, args.quantization)

if __name__ == "__main__":
    main()
