"""
Main training script for 7B Parameter LLM
Optimized for 24GB VRAM with memory-efficient techniques
"""

import os
import sys
import yaml
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from model import create_model
from data_loader import LLMDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LLMTrainer:
    """Main trainer class for hcho LLM"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.accelerator = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.global_step = 0
        self.epoch = 0
        
        # Create output directories
        self.setup_directories()
        
        # Setup accelerator
        self.setup_accelerator()
        
        # Setup model and data
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        # Setup logging
        self.setup_logging()
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['output']['model_dir'],
            self.config['output']['checkpoint_dir'],
            self.config['output']['log_dir'],
            self.config['output']['cache_dir']
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {dir_path}")
    
    def setup_accelerator(self):
        """Setup Accelerate for distributed training and memory optimization"""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            mixed_precision='fp16' if self.config['training']['fp16'] else 'no',
            log_with='wandb' if self.config['logging']['use_wandb'] else None,
            project_dir=self.config['output']['log_dir'],
            gradient_clip_norm=self.config['training']['max_grad_norm']
        )
        
        logger.info(f"✅ Accelerator setup complete")
        logger.info(f"   Device: {self.accelerator.device}")
        logger.info(f"   Mixed precision: {self.accelerator.mixed_precision}")
        logger.info(f"   Gradient accumulation steps: {self.accelerator.gradient_accumulation_steps}")
    
    def setup_model(self):
        """Setup the model"""
        logger.info("🔄 Setting up model...")
        
        # Create model
        self.model = create_model(self.config)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        
        # Move model to device
        self.model = self.accelerator.prepare(self.model)
        
        logger.info(f"✅ Model setup complete")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def setup_data(self):
        """Setup data loaders"""
        logger.info("🔄 Setting up data...")
        
        # Create data loader
        data_loader = LLMDataLoader(self.config)
        data_loader.setup_tokenizer()
        
        # Create dataloaders
        self.train_dataloader, self.eval_dataloader = data_loader.create_dataloaders()
        
        # Prepare dataloaders with accelerator
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        
        logger.info("✅ Data setup complete")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        logger.info("🔄 Setting up optimizer...")
        
        # Debug config values
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        logger.info(f"   Learning rate: {lr} (type: {type(lr)})")
        logger.info(f"   Weight decay: {weight_decay} (type: {type(weight_decay)})")
        
        # Ensure numeric types
        if isinstance(lr, str):
            lr = float(lr)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        # Scheduler
        total_steps = len(self.train_dataloader) * self.config['training']['num_epochs']
        warmup_steps = self.config['training']['warmup_steps']
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr * 0.1
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        # Prepare with accelerator
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)
        
        logger.info("✅ Optimizer setup complete")
    
    def setup_logging(self):
        """Setup logging and monitoring"""
        if self.config['logging']['use_wandb']:
            self.accelerator.init_trackers(
                project_name=self.config['logging']['project_name'],
                config=self.config
            )
            logger.info("✅ Weights & Biases logging enabled")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Create progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.config['training']['num_epochs']}",
            unit="batch",
            leave=True
        )
        
        # Memory management
        empty_cache_steps = self.config['optimization'].get('empty_cache_steps', 100)
        
        for batch_idx, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Optimizer step (Accelerate handles gradient clipping automatically)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Memory management - clear cache periodically
                if batch_idx % empty_cache_steps == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'LR': f'{lr:.2e}',
                    'Step': self.global_step
                })
                
                # Logging
                if self.global_step % self.config['training']['logging_steps'] == 0:
                    logger.info(f"Step {self.global_step}: Loss={avg_loss:.4f}, LR={lr:.2e}")
                    
                    if self.config['logging']['use_wandb']:
                        self.accelerator.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/epoch': self.epoch
                        })
                
                # Evaluation
                if self.global_step % self.config['training']['eval_steps'] == 0:
                    self.evaluate()
                
                # Checkpointing
                if self.global_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint()
    
    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels']
                )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        if self.config['logging']['use_wandb']:
            self.accelerator.log({
                'eval/loss': avg_loss,
                'eval/perplexity': perplexity,
                'eval/epoch': self.epoch
            })
        
        self.model.train()
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config['output']['checkpoint_dir']) / f"checkpoint-{self.global_step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), checkpoint_path / "model.pt")
        
        # Save optimizer and scheduler
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), checkpoint_path / "scheduler.pt")
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config
        }
        torch.save(training_state, checkpoint_path / "training_state.pt")
        
        logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            logger.info(f"📚 Starting epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            self.train_epoch()
            
            # Final evaluation and checkpoint
            self.evaluate()
            self.save_checkpoint()
        
        # Save final model
        final_model_path = Path(self.config['output']['model_dir']) / "final_model.pt"
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), final_model_path)
        
        training_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {training_time/3600:.2f} hours")
        
        if self.config['logging']['use_wandb']:
            self.accelerator.end_training()

def main():
    parser = argparse.ArgumentParser(description="Train 7B Parameter LLM")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("🚀 Starting hcho Training")
    logger.info(f"   Config: {args.config}")
    logger.info(f"   Seed: {args.seed}")
    logger.info(f"   Resume: {args.resume}")
    
    # Create trainer
    trainer = LLMTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"🔄 Resuming from checkpoint: {args.resume}")
        # TODO: Implement checkpoint loading
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
