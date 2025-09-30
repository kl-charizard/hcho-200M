"""
Data loading utilities for LLM training
Handles multiple dataset sources and preprocessing
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Union
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class LLMDataLoader:
    """Data loader for LLM training with multiple dataset support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_tokenizer(self, model_name: str = "gpt2"):
        """Setup tokenizer for the model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"✅ Loaded tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            raise
    
    def load_datasets(self) -> tuple[Dataset, Dataset]:
        """Load and preprocess datasets"""
        logger.info("🔄 Loading datasets...")
        
        datasets = []
        
        # Try to load multiple datasets with correct syntax
        dataset_configs = [
            {"path": "wikitext", "name": "wikitext-2-raw-v1", "split": "train"},
            {"path": "wikitext", "name": "wikitext-103-raw-v1", "split": "train"},
            {"path": "bookcorpus", "split": "train"},
            {"path": "stories", "split": "train"},
            {"path": "squad", "split": "train"},
            {"path": "glue", "name": "sst2", "split": "train"},
            {"path": "imdb", "split": "train"},
            {"path": "ag_news", "split": "train"},
            {"path": "yelp_review_full", "split": "train"},
            {"path": "amazon_polarity", "split": "train"},
        ]
        
        for dataset_config in dataset_configs:
            try:
                logger.info(f"🔄 Loading {dataset_config['path']}...")
                dataset = load_dataset(**dataset_config)
                
                # Convert to non-streaming if needed and limit size
                if hasattr(dataset, 'take'):
                    # Increase dataset size significantly
                    if dataset_config["path"] == "wikitext" and dataset_config.get("name") == "wikitext-103-raw-v1":
                        dataset = dataset.take(100000)  # Much larger for main dataset
                    elif dataset_config["path"] == "wikitext" and dataset_config.get("name") == "wikitext-2-raw-v1":
                        dataset = dataset.take(50000)  # Larger for secondary dataset
                    else:
                        dataset = dataset.take(50000)  # Increased from 25000
                
                datasets.append(dataset)
                logger.info(f"✅ Loaded {dataset_config['path']}: {len(dataset)} samples")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {dataset_config['path']}: {e}")
                continue
        
        if not datasets:
            logger.warning("⚠️  No online datasets could be loaded!")
            logger.info("🔄 Creating sample dataset for training...")
            sample_dataset = create_sample_dataset()
            datasets.append(sample_dataset)
            logger.info(f"✅ Created sample dataset with {len(sample_dataset)} samples")
        
        # Standardize dataset features before combining
        logger.info("🔄 Standardizing dataset features...")
        standardized_datasets = []
        for i, dataset in enumerate(datasets):
            try:
                # Remove all columns except text and create standardized format
                if 'text' in dataset.column_names:
                    # Already has text column
                    standardized_dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
                else:
                    # Need to preprocess to create text column
                    standardized_dataset = dataset.map(
                        self.preprocess_text,
                        batched=True,
                        remove_columns=dataset.column_names
                    )
                standardized_datasets.append(standardized_dataset)
                logger.info(f"✅ Standardized dataset {i+1}: {len(standardized_dataset)} samples")
            except Exception as e:
                logger.warning(f"⚠️  Failed to standardize dataset {i+1}: {e}")
                continue
        
        if not standardized_datasets:
            logger.error("❌ No datasets could be standardized!")
            raise RuntimeError("No datasets available after standardization")
        
        # Combine standardized datasets
        logger.info("🔄 Combining datasets...")
        combined_dataset = concatenate_datasets(standardized_datasets)
        
        # Split into train/eval
        train_size = int(len(combined_dataset) * self.config['data']['train_split'])
        eval_size = len(combined_dataset) - train_size
        
        train_dataset = combined_dataset.select(range(train_size))
        eval_dataset = combined_dataset.select(range(train_size, train_size + eval_size))
        
        logger.info(f"✅ Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        return train_dataset, eval_dataset
    
    def preprocess_text(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Preprocess text data and standardize features"""
        # Handle different text field names
        if 'text' in examples:
            texts = examples['text']
        elif 'content' in examples:
            texts = examples['content']
        elif 'question' in examples and 'context' in examples:
            # For SQuAD, combine question and context
            texts = [f"Question: {q} Context: {c}" for q, c in zip(examples['question'], examples['context'])]
        else:
            # Fallback: try to find any text field
            text_fields = [k for k in examples.keys() if 'text' in k.lower() or 'content' in k.lower()]
            if text_fields:
                texts = examples[text_fields[0]]
            else:
                texts = [str(v) for v in examples.values() if isinstance(v, list) and len(v) > 0][0]
        
        processed_texts = []
        for text in texts:
            if isinstance(text, str):
                # Basic cleaning
                text = text.strip()
                if len(text) < self.config['data']['preprocessing']['min_length']:
                    continue
                if len(text) > self.config['data']['preprocessing']['max_length']:
                    text = text[:self.config['data']['preprocessing']['max_length']]
                processed_texts.append(text)
        
        return {'text': processed_texts}
    
    def tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List]:
        """Tokenize text data"""
        texts = examples['text']
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config['data']['max_length'],
            return_tensors=None  # Return lists instead of tensors
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def collate_fn(self, batch):
        """Custom collate function to convert lists to tensors"""
        # Convert lists to tensors
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set up. Call setup_tokenizer() first.")
        
        # Load datasets (already preprocessed and standardized)
        train_dataset, eval_dataset = self.load_datasets()
        
        # Tokenize
        logger.info("🔄 Tokenizing datasets...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        eval_dataset = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['dataloader_num_workers'],
            pin_memory=False,  # Disable for MPS compatibility
            collate_fn=self.collate_fn
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['dataloader_num_workers'],
            pin_memory=False,  # Disable for MPS compatibility
            collate_fn=self.collate_fn
        )
        
        logger.info("✅ DataLoaders created successfully")
        return train_dataloader, eval_dataloader

def create_sample_dataset():
    """Create a large sample dataset for training"""
    base_texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample text for training.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers to learn patterns.",
        "Transformers have revolutionized the field of natural language processing.",
        "Large language models can generate human-like text and answer questions.",
        "Training neural networks requires large amounts of data and computational power.",
        "The attention mechanism allows models to focus on relevant parts of input.",
        "Pre-training and fine-tuning are common approaches in modern NLP.",
        "Language models can be used for text generation, translation, and summarization.",
        "Artificial intelligence is transforming various industries and applications.",
        "Neural networks are inspired by the structure and function of biological neurons.",
        "Backpropagation is a key algorithm for training neural networks.",
        "Convolutional neural networks excel at image recognition tasks.",
        "Recurrent neural networks are designed to handle sequential data.",
        "The transformer architecture uses self-attention mechanisms effectively.",
        "Transfer learning allows models to leverage pre-trained knowledge.",
        "Data augmentation techniques help improve model generalization.",
        "Regularization methods prevent overfitting in machine learning models.",
        "Hyperparameter tuning is crucial for optimizing model performance."
    ]
    
    # Create a much larger dataset by repeating and varying the texts
    sample_texts = []
    for i in range(50000):  # 50,000 samples for much more data
        base_text = base_texts[i % len(base_texts)]
        # Add much more variation and context to make longer sequences
        varied_text = f"{base_text} This is variation {i} with extensive additional context about machine learning, artificial intelligence, deep learning, neural networks, transformers, natural language processing, computer vision, reinforcement learning, and various applications in technology. The field of AI continues to evolve rapidly with new breakthroughs in large language models, multimodal systems, and advanced reasoning capabilities. These developments are transforming industries and creating new opportunities for innovation and research."
        sample_texts.append(varied_text)
    
    dataset = Dataset.from_dict({"text": sample_texts})
    return dataset

if __name__ == "__main__":
    # Test the data loader
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    data_loader = LLMDataLoader(config)
    data_loader.setup_tokenizer()
    
    try:
        train_loader, eval_loader = data_loader.create_dataloaders()
        print(f"✅ Successfully created dataloaders")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Eval batches: {len(eval_loader)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔄 Creating sample dataset for testing...")
        
        # Create sample dataset
        sample_dataset = create_sample_dataset()
        print(f"✅ Created sample dataset with {len(sample_dataset)} samples")
