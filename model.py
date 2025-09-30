"""
100M Parameter LLM Model Implementation
Based on GPT architecture with optimizations for MacBook and Colab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            # Reshape attention mask to match scores shape (batch_size, num_heads, seq_len, seq_len)
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            # Use a smaller value for FP16 compatibility
            mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(attn_mask == 0, mask_value)
        
        # Causal mask (prevent looking at future tokens)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        # Use a smaller value for FP16 compatibility
        mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
        scores = scores.masked_fill(causal_mask == 0, mask_value)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(self.ln1(x), attention_mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x

class LLMModel(nn.Module):
    """hcho - 100M Parameter LLM Model"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        
        # Model dimensions
        self.vocab_size = config['model']['vocab_size']
        self.hidden_size = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        self.num_heads = config['model']['num_heads']
        self.max_position_embeddings = config['model']['max_position_embeddings']
        self.intermediate_size = config['model']['intermediate_size']
        self.dropout = config['model']['dropout']
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                self.hidden_size,
                self.num_heads,
                self.intermediate_size,
                self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"✅ Initialized hcho model with {self.count_parameters():,} parameters")
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> dict:
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Debug: Check for invalid token IDs
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            logger.error(f"Invalid token IDs detected: min={input_ids.min()}, max={input_ids.max()}, vocab_size={self.vocab_size}")
            # Clamp invalid token IDs to valid range
            input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Ensure position_ids are within bounds
        position_ids = torch.clamp(position_ids, 0, self.max_position_embeddings - 1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Apply dropout
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # Pass through transformer layers with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            for layer in self.layers:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, use_reentrant=False
                )
        else:
            # Normal forward pass
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        """Generate text using the model"""
        self.eval()
        device = input_ids.device
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, -float('inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

def create_model(config: dict) -> LLMModel:
    """Create and return a new LLM model"""
    return LLMModel(config)

if __name__ == "__main__":
    # Test model creation
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model = create_model(config)
    print(f"✅ Model created with {model.count_parameters():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config['model']['vocab_size'], (batch_size, seq_len))
    
    outputs = model(input_ids)
    print(f"✅ Forward pass successful")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Loss: {outputs['loss']}")
