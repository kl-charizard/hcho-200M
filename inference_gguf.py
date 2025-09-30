"""
GGUF Model Inference Script
Run inference using quantized GGUF models with llama-cpp-python
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python not installed. Please install it:")
    print("pip install llama-cpp-python")
    exit(1)

logger = logging.getLogger(__name__)

class GGUFInference:
    """Inference engine for GGUF quantized models"""
    
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.llm = None
        
        # Default parameters optimized for 24GB VRAM
        default_params = {
            'n_ctx': 2048,           # Context length
            'n_gpu_layers': -1,      # Use all GPU layers
            'n_threads': 8,          # CPU threads
            'n_batch': 512,          # Batch size
            'verbose': False,        # Verbose output
            'use_mmap': True,        # Memory mapping
            'use_mlock': True,       # Lock memory
            'low_vram': False,       # Low VRAM mode
            'f16_kv': True,         # Use FP16 for key-value cache
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        self.params = default_params
    
    def load_model(self):
        """Load the GGUF model"""
        logger.info(f"🔄 Loading GGUF model: {self.model_path}")
        
        try:
            self.llm = Llama(
                model_path=self.model_path,
                **self.params
            )
            
            logger.info("✅ Model loaded successfully")
            
            # Print model info
            if hasattr(self.llm, 'ctx'):
                ctx_size = self.llm.ctx.params.n_ctx_train
                logger.info(f"   Context size: {ctx_size}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 40,
                 repeat_penalty: float = 1.1,
                 stop: Optional[List[str]] = None) -> str:
        """Generate text from prompt"""
        
        if self.llm is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"🔄 Generating text (max_tokens={max_tokens}, temp={temperature})")
        
        start_time = time.time()
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                echo=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract generated text
            generated_text = response['choices'][0]['text']
            
            # Calculate tokens per second
            tokens_generated = len(generated_text.split())
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            logger.info(f"✅ Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
    
    def chat(self, 
              messages: List[dict], 
              max_tokens: int = 100,
              temperature: float = 0.7,
              **kwargs) -> str:
        """Chat with the model using conversation format"""
        
        if self.llm is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def _messages_to_prompt(self, messages: List[dict]) -> str:
        """Convert chat messages to prompt format"""
        prompt = ""
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"Human: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant:"
        return prompt
    
    def benchmark(self, prompts: List[str], **kwargs):
        """Benchmark the model performance"""
        logger.info(f"🔄 Running benchmark with {len(prompts)} prompts")
        
        total_time = 0
        total_tokens = 0
        
        for i, prompt in enumerate(prompts):
            logger.info(f"🔄 Benchmark prompt {i+1}/{len(prompts)}")
            
            start_time = time.time()
            response = self.generate(prompt, **kwargs)
            generation_time = time.time() - start_time
            
            tokens = len(response.split())
            total_time += generation_time
            total_tokens += tokens
            
            logger.info(f"   Generated {tokens} tokens in {generation_time:.2f}s")
        
        # Calculate averages
        avg_time = total_time / len(prompts)
        avg_tokens = total_tokens / len(prompts)
        avg_tokens_per_second = total_tokens / total_time
        
        logger.info("📊 Benchmark Results:")
        logger.info(f"   Average time per prompt: {avg_time:.2f}s")
        logger.info(f"   Average tokens per prompt: {avg_tokens:.1f}")
        logger.info(f"   Average tokens per second: {avg_tokens_per_second:.1f}")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Total tokens: {total_tokens}")

def interactive_chat(model_path: str, **model_params):
    """Interactive chat interface"""
    inference = GGUFInference(model_path, **model_params)
    inference.load_model()
    
    print("🤖 Interactive Chat with hcho-200M")
    print("Type 'quit' to exit, 'clear' to clear history")
    print("=" * 50)
    
    messages = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'clear':
                messages = []
                print("🧹 Chat history cleared")
                continue
            elif not user_input:
                continue
            
            # Add user message
            messages.append({'role': 'user', 'content': user_input})
            
            # Generate response
            print("🤖 Assistant: ", end="", flush=True)
            response = inference.chat(messages, max_tokens=150, temperature=0.7)
            print(response)
            
            # Add assistant response
            messages.append({'role': 'assistant', 'content': response})
            
            # Keep only last 10 messages to prevent context overflow
            if len(messages) > 20:
                messages = messages[-20:]
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with GGUF quantized model")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model file")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of GPU layers (-1 for all)")
    parser.add_argument("--n-threads", type=int, default=8, help="Number of CPU threads")
    parser.add_argument("--low-vram", action="store_true", help="Enable low VRAM mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Model parameters
    model_params = {
        'n_gpu_layers': args.n_gpu_layers,
        'n_threads': args.n_threads,
        'low_vram': args.low_vram,
    }
    
    if args.interactive:
        # Interactive chat
        interactive_chat(args.model, **model_params)
    elif args.benchmark:
        # Benchmark mode
        inference = GGUFInference(args.model, **model_params)
        inference.load_model()
        
        benchmark_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The key to solving climate change lies in",
            "Education in the 21st century should focus on",
            "The most important skill for the future is"
        ]
        
        inference.benchmark(benchmark_prompts, max_tokens=args.max_tokens, temperature=args.temperature)
    else:
        # Single generation
        inference = GGUFInference(args.model, **model_params)
        inference.load_model()
        
        if args.prompt:
            response = inference.generate(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
            print(f"\n🤖 Response:\n{response}")
        else:
            print("❌ Please provide a prompt with --prompt or use --interactive for chat mode")

if __name__ == "__main__":
    main()
