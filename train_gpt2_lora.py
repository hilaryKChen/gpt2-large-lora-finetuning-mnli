#!/usr/bin/env python3
"""
LoRA fine-tuning script for GPT2-large on MultiNLI dataset.
Optimized for 36G vGPU with memory-efficient training strategies.
"""

import os
import json
import torch
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
from datasets import load_from_disk, Dataset
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2LoRATrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize GPT2-large model and tokenizer with LoRA configuration."""
        logger.info("Loading GPT2-large tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            padding_side="right"
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Loading GPT2-large model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float16 if self.config['use_fp16'] else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            target_modules=self.config['target_modules'],
            bias="none"
        )
        
        logger.info(f"Applying LoRA with config: {lora_config}")
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
    def load_datasets(self) -> Dict[str, Dataset]:
        """Load preprocessed datasets."""
        datasets = {}
        data_dir = self.config['data_dir']
        
        for split in ['train', 'validation', 'test_matched', 'test_mismatched']:
            dataset_path = os.path.join(data_dir, f"{split}_dataset")
            if os.path.exists(dataset_path):
                datasets[split] = load_from_disk(dataset_path)
                logger.info(f"Loaded {split} dataset: {len(datasets[split])} samples")
            else:
                logger.warning(f"Dataset not found: {dataset_path}")
                
        return datasets
    
    def tokenize_function(self, examples):
        """Tokenize the input text for training."""
        # Tokenize the text
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=self.config['max_length'],
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def prepare_datasets(self, datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """Tokenize and prepare datasets for training."""
        processed_datasets = {}
        
        for split_name, dataset in datasets.items():
            logger.info(f"Tokenizing {split_name} dataset...")
            tokenized_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Tokenizing {split_name}"
            )
            processed_datasets[split_name] = tokenized_dataset
            
        return processed_datasets
    
    def train(self):
        """Main training loop."""
        # Disable wandb if not requested
        if not self.config.get('use_wandb', False):
            import os
            os.environ['WANDB_DISABLED'] = 'true'
            os.environ['WANDB_MODE'] = 'disabled'
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Load and prepare datasets
        datasets = self.load_datasets()
        tokenized_datasets = self.prepare_datasets(datasets)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Prepare training arguments with compatibility handling
        training_args_dict = {
            "output_dir": self.config['output_dir'],
            "overwrite_output_dir": True,
            
            # Training parameters
            "num_train_epochs": self.config['num_epochs'],
            "per_device_train_batch_size": self.config['train_batch_size'],
            "per_device_eval_batch_size": self.config['eval_batch_size'],
            "gradient_accumulation_steps": self.config['gradient_accumulation_steps'],
            
            # Optimization
            "learning_rate": self.config['learning_rate'],
            "weight_decay": self.config['weight_decay'],
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": self.config['max_grad_norm'],
            
            # Memory optimization
            "fp16": self.config['use_fp16'],
            "dataloader_pin_memory": False,
            "gradient_checkpointing": self.config['gradient_checkpointing'],
            
            # Logging and evaluation
            "logging_steps": self.config['logging_steps'],
            "eval_steps": self.config['eval_steps'],
            "save_steps": self.config['save_steps'],
            "save_strategy": "steps",
            "save_total_limit": self.config['save_total_limit'],
            
            # Early stopping
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            
            # Reporting
            "report_to": "wandb" if self.config['use_wandb'] else [],
            "run_name": f"gpt2-large-lora-multinli-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            
            # Misc
            "seed": self.config['seed'],
            "remove_unused_columns": False,
        }
        
        # Handle evaluation_strategy vs eval_strategy compatibility
        import transformers
        from packaging import version
        
        # Check transformers version and use appropriate parameter name
        if version.parse(transformers.__version__) >= version.parse("4.34.0"):
            training_args_dict["eval_strategy"] = "steps"
        else:
            training_args_dict["evaluation_strategy"] = "steps"
        
        # Create TrainingArguments
        training_args = TrainingArguments(**training_args_dict)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets.get('validation'),
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config['early_stopping_patience'])]
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        final_model_path = os.path.join(self.config['output_dir'], "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"Training completed. Model saved to {final_model_path}")
        
        return trainer

def get_default_config() -> Dict[str, Any]:
    """Get default configuration for training."""
    return {
        # Model configuration
        'model_name': 'gpt2-large',
        'max_length': 512,
        
        # LoRA configuration
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'target_modules': ['c_attn', 'c_proj', 'c_fc'],  # GPT2 attention and MLP layers
        
        # Training configuration
        'num_epochs': 3,
        'train_batch_size': 4,  # Optimized for 36G vGPU
        'eval_batch_size': 8,
        'gradient_accumulation_steps': 8,  # Effective batch size = 4 * 8 = 32
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        
        # Memory optimization
        'use_fp16': True,
        'gradient_checkpointing': True,
        
        # Logging and evaluation
        'logging_steps': 50,
        'eval_steps': 200,
        'save_steps': 200,
        'save_total_limit': 3,
        'early_stopping_patience': 3,
        
        # Paths
        'data_dir': './processed_data',
        'output_dir': './gpt2_lora_multinli',
        
        # Misc
        'seed': 42,
        'use_wandb': False,
    }

def main():
    # Disable wandb by default to prevent auto-initialization
    import os
    if 'WANDB_DISABLED' not in os.environ:
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_MODE'] = 'disabled'
    
    parser = argparse.ArgumentParser(description="Fine-tune GPT2-large with LoRA on MultiNLI")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--data_dir", type=str, default="./processed_data", 
                       help="Directory containing processed datasets")
    parser.add_argument("--output_dir", type=str, default="./gpt2_lora_multinli",
                       help="Output directory for model and logs")
    parser.add_argument("--model_name", type=str, default="gpt2-large",
                       help="Pre-trained model name")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'model_name': args.model_name,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'train_batch_size': args.train_batch_size,
    })
    
    # Only override wandb setting if explicitly requested
    if args.no_wandb:
        config['use_wandb'] = False
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        # Re-enable wandb for this run
        os.environ['WANDB_DISABLED'] = 'false'
        os.environ['WANDB_MODE'] = 'online'
        
        wandb.init(
            project="gpt2-lora-multinli",
            config=config,
            name=f"gpt2-large-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    else:
        # Ensure wandb stays disabled
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_MODE'] = 'disabled'
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config['output_dir'], 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
    logger.info(f"Training configuration: {json.dumps(config, indent=2)}")
    
    # Initialize trainer and start training
    trainer = GPT2LoRATrainer(config)
    trainer.train()
    
    if config.get('use_wandb', False):
        wandb.finish()

if __name__ == "__main__":
    main() 