#!/usr/bin/env python3
"""
Data preprocessing script for MultiNLI dataset to prepare for GPT2-large LoRA fine-tuning.
Loads the actual MultiNLI training data and uses dev sets for evaluation only.
Converts JSONL format to text-to-text format suitable for causal language modeling.
"""

import json
import pandas as pd
from datasets import Dataset, load_dataset
from typing import List, Dict, Any, Optional
import argparse
import os

class MultiNLIPreprocessor:
    def __init__(self):
        self.label_map = {
            'entailment': 'entailment',
            'contradiction': 'contradiction', 
            'neutral': 'neutral'
        }
        
    def load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def format_nli_prompt(self, premise: str, hypothesis: str, label: str = None) -> str:
        """
        Format NLI data into a prompt suitable for GPT2 training.
        Format: "Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: {label}"
        """
        if label:
            return f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: {label}<|endoftext|>"
        else:
            return f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship:"
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert raw MultiNLI data to training format."""
        processed_data = []
        
        for item in data:
            # Extract relevant fields
            premise = item['sentence1'].strip()
            hypothesis = item['sentence2'].strip()
            label = item['gold_label']
            
            # Skip items with invalid labels
            if label not in self.label_map:
                continue
                
            # Create training example
            text = self.format_nli_prompt(premise, hypothesis, self.label_map[label])
            
            processed_data.append({
                'text': text,
                'premise': premise,
                'hypothesis': hypothesis,
                'label': label,
                'genre': item.get('genre', ''),
                'pairID': item.get('pairID', '')
            })
            
        return processed_data
    
    def load_multinli_from_hf(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load the MultiNLI dataset from HuggingFace with proper structure."""
        print("Loading MultiNLI dataset from HuggingFace (nyu-mll/multi_nli)...")
        try:
            # Load the full MultiNLI dataset
            dataset = load_dataset("nyu-mll/multi_nli")
            
            # Label mapping: 0=entailment, 1=neutral, 2=contradiction
            label_names = ['entailment', 'neutral', 'contradiction']
            
            processed_data = {}
            
            # Process each split
            for split_name in ['train', 'validation_matched', 'validation_mismatched']:
                if split_name in dataset:
                    split_data = dataset[split_name]
                    print(f"Processing {split_name}: {len(split_data)} samples")
                    
                    # Filter out samples with invalid labels (-1)
                    valid_samples = split_data.filter(lambda x: x['label'] != -1)
                    print(f"After filtering invalid labels: {len(valid_samples)} samples")
                    
                    # Convert to our format
                    samples = []
                    for sample in valid_samples:
                        samples.append({
                            'sentence1': sample['premise'],
                            'sentence2': sample['hypothesis'],
                            'gold_label': label_names[sample['label']],
                            'genre': sample.get('genre', ''),
                            'pairID': sample.get('pairID', ''),
                            'promptID': sample.get('promptID', '')
                        })
                    
                    processed_data[split_name] = samples
                    print(f"Processed {split_name}: {len(samples)} valid samples")
                    
            return processed_data
            
        except Exception as e:
            print(f"Error loading MultiNLI from HuggingFace: {e}")
            return {}

    def save_validation_files(self, validation_data: Dict[str, List[Dict[str, Any]]], 
                            output_dir: str = "."):
        """Save validation data as JSONL files for future use."""
        import json
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, data in validation_data.items():
            if 'validation' in split_name:
                # Convert back to original format for saving
                jsonl_data = []
                for item in data:
                    jsonl_item = {
                        'sentence1': item['sentence1'],
                        'sentence2': item['sentence2'],
                        'gold_label': item['gold_label'],
                        'genre': item['genre'],
                        'pairID': item['pairID'],
                        'promptID': item['promptID']
                    }
                    jsonl_data.append(jsonl_item)
                
                # Determine filename
                if 'matched' in split_name:
                    filename = 'dev_matched_sampled_hf.jsonl'
                else:
                    filename = 'dev_mismatched_sampled_hf.jsonl'
                
                filepath = os.path.join(output_dir, filename)
                
                # Save as JSONL
                with open(filepath, 'w', encoding='utf-8') as f:
                    for item in jsonl_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                print(f"Saved {len(jsonl_data)} samples to {filepath}")

    def create_dataset(self, output_dir: str = "./processed_data", 
                      max_train_samples: Optional[int] = None,
                      save_validation_files: bool = True) -> Dict[str, Dataset]:
        """Create datasets using HuggingFace MultiNLI data and save validation files."""
        
        # Load all data from HuggingFace
        hf_data = self.load_multinli_from_hf()
        
        if not hf_data:
            raise ValueError("Failed to load MultiNLI dataset from HuggingFace")
        
        # Process training data
        if 'train' in hf_data:
            train_data_raw = hf_data['train']
            print(f"Original training data: {len(train_data_raw)} samples")
            
            # Limit training samples if specified
            if max_train_samples and len(train_data_raw) > max_train_samples:
                import random
                random.seed(42)
                train_data_raw = random.sample(train_data_raw, max_train_samples)
                print(f"Limited training data to {max_train_samples} samples")
            
            # Preprocess training data
            print("Preprocessing training data...")
            processed_train_data = self.preprocess_data(train_data_raw)
            
            # Split into train/validation (90/10)
            split_idx = int(len(processed_train_data) * 0.9)
            train_data = processed_train_data[:split_idx]
            val_data = processed_train_data[split_idx:]
            
            print(f"Training samples: {len(train_data)}")
            print(f"Internal validation samples: {len(val_data)}")
        else:
            raise ValueError("No training data found in HuggingFace dataset")
        
        # Process validation data for testing
        test_datasets = {}
        
        if 'validation_matched' in hf_data:
            matched_data = hf_data['validation_matched']
            print(f"Matched validation data: {len(matched_data)} samples")
            processed_matched = self.preprocess_data(matched_data)
            test_datasets['test_matched'] = Dataset.from_list(processed_matched)
            
        if 'validation_mismatched' in hf_data:
            mismatched_data = hf_data['validation_mismatched']
            print(f"Mismatched validation data: {len(mismatched_data)} samples")
            processed_mismatched = self.preprocess_data(mismatched_data)
            test_datasets['test_mismatched'] = Dataset.from_list(processed_mismatched)
        
        # Save validation files if requested
        if save_validation_files:
            print("Saving validation files...")
            validation_data = {k: v for k, v in hf_data.items() if 'validation' in k}
            self.save_validation_files(validation_data, output_dir)
        
        # Create training datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        return {
            'train': train_dataset,
            'validation': val_dataset,
            **test_datasets
        }
    
    def save_datasets(self, datasets: Dict[str, Dataset], output_dir: str):
        """Save processed datasets to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, dataset in datasets.items():
            output_path = os.path.join(output_dir, f"{split_name}_dataset")
            dataset.save_to_disk(output_path)
            print(f"Saved {split_name} dataset to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess MultiNLI data for GPT2 training using HuggingFace dataset")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                       help="Output directory for processed datasets")
    parser.add_argument("--max_train_samples", type=int, default=50000,
                       help="Maximum number of training samples to use")
    parser.add_argument("--no_save_validation", action="store_true",
                       help="Don't save validation files as JSONL")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = MultiNLIPreprocessor()
    
    # Create datasets
    datasets = preprocessor.create_dataset(
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        save_validation_files=not args.no_save_validation
    )
    
    # Save datasets
    preprocessor.save_datasets(datasets, args.output_dir)
    
    # Print sample
    print("\nSample training example:")
    print(datasets['train'][0]['text'])
    
    # Print label distribution
    train_labels = [item['label'] for item in datasets['train']]
    label_counts = pd.Series(train_labels).value_counts()
    print(f"\nLabel distribution in training set:")
    print(label_counts)

if __name__ == "__main__":
    main() 