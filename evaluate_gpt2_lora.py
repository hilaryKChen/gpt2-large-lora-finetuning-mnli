#!/usr/bin/env python3
"""
Evaluation script for GPT2-large LoRA model fine-tuned on MultiNLI dataset.
Evaluates on both matched and mismatched test sets with detailed metrics.
"""

import os
import json
import torch
import argparse
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2LoRAEvaluator:
    def __init__(self, model_path: str, base_model_name: str = "gpt2-large"):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the fine-tuned LoRA model and tokenizer."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
        
    def format_nli_prompt(self, premise: str, hypothesis: str) -> str:
        """Format NLI data into the same prompt format used during training."""
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship:"
        
    def predict_single(self, premise: str, hypothesis: str, max_new_tokens: int = 10) -> str:
        """Predict the relationship for a single premise-hypothesis pair."""
        prompt = self.format_nli_prompt(premise, hypothesis)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the prediction (everything after "Relationship:")
        if "Relationship:" in generated_text:
            prediction = generated_text.split("Relationship:")[-1].strip()
            # Extract just the first word (the label)
            prediction = prediction.split()[0].lower() if prediction.split() else ""
            
            # Map to valid labels
            if prediction in self.label_map:
                return prediction
            elif "entail" in prediction:
                return "entailment"
            elif "contradict" in prediction:
                return "contradiction"
            elif "neutral" in prediction:
                return "neutral"
            else:
                return "neutral"  # Default fallback
        else:
            return "neutral"  # Default fallback
            
    def evaluate_dataset(self, dataset: Dataset, dataset_name: str) -> Dict[str, Any]:
        """Evaluate the model on a dataset and return metrics."""
        logger.info(f"Evaluating on {dataset_name} dataset ({len(dataset)} samples)...")
        
        predictions = []
        true_labels = []
        
        # Evaluate each sample
        for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
            premise = sample['premise']
            hypothesis = sample['hypothesis']
            true_label = sample['label']
            
            # Get prediction
            pred_label = self.predict_single(premise, hypothesis)
            
            predictions.append(pred_label)
            true_labels.append(true_label)
            
            # Log progress every 100 samples
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} samples")
                
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0, 
            labels=['entailment', 'contradiction', 'neutral']
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, 
                            labels=['entailment', 'contradiction', 'neutral'])
        
        # Compile results
        results = {
            'dataset_name': dataset_name,
            'num_samples': len(dataset),
            'accuracy': accuracy,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1,
            'per_class_metrics': {
                'entailment': {
                    'precision': per_class_precision[0],
                    'recall': per_class_recall[0],
                    'f1': per_class_f1[0],
                    'support': support[0]
                },
                'contradiction': {
                    'precision': per_class_precision[1],
                    'recall': per_class_recall[1],
                    'f1': per_class_f1[1],
                    'support': support[1]
                },
                'neutral': {
                    'precision': per_class_precision[2],
                    'recall': per_class_recall[2],
                    'f1': per_class_f1[2],
                    'support': support[2]
                }
            },
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        return results
        
    def plot_confusion_matrix(self, results: Dict[str, Any], output_dir: str):
        """Plot and save confusion matrix."""
        cm = np.array(results['confusion_matrix'])
        labels = ['Entailment', 'Contradiction', 'Neutral']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {results["dataset_name"]}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        output_path = os.path.join(output_dir, f'confusion_matrix_{results["dataset_name"]}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
        
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to JSON file."""
        # Remove predictions and true_labels for cleaner JSON (they're large)
        results_clean = {k: v for k, v in results.items() 
                        if k not in ['predictions', 'true_labels']}
        
        output_path = os.path.join(output_dir, f'results_{results["dataset_name"]}.json')
        with open(output_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
        
        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'true_label': results['true_labels'],
            'predicted_label': results['predictions'],
            'correct': [t == p for t, p in zip(results['true_labels'], results['predictions'])]
        })
        
        predictions_path = os.path.join(output_dir, f'predictions_{results["dataset_name"]}.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Detailed predictions saved to {predictions_path}")
        
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {results['dataset_name'].upper()}")
        print(f"{'='*60}")
        print(f"Number of samples: {results['num_samples']}")
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Weighted Precision: {results['weighted_precision']:.4f}")
        print(f"Weighted Recall: {results['weighted_recall']:.4f}")
        print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"{class_name:<12} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} {metrics['support']:<10}")
                  
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT2-large LoRA model on MultiNLI")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned LoRA model")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                       help="Directory containing processed test datasets")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--base_model", type=str, default="gpt2-large",
                       help="Base model name")
    parser.add_argument("--batch_eval", action="store_true",
                       help="Enable batch evaluation (faster but more memory)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = GPT2LoRAEvaluator(args.model_path, args.base_model)
    evaluator.load_model()
    
    # Load test datasets
    test_datasets = {}
    for dataset_name in ['test_matched', 'test_mismatched']:
        dataset_path = os.path.join(args.data_dir, f"{dataset_name}_dataset")
        if os.path.exists(dataset_path):
            test_datasets[dataset_name] = load_from_disk(dataset_path)
            logger.info(f"Loaded {dataset_name}: {len(test_datasets[dataset_name])} samples")
        else:
            logger.warning(f"Dataset not found: {dataset_path}")
    
    # Evaluate on each test dataset
    all_results = {}
    for dataset_name, dataset in test_datasets.items():
        logger.info(f"Starting evaluation on {dataset_name}...")
        
        results = evaluator.evaluate_dataset(dataset, dataset_name)
        all_results[dataset_name] = results
        
        # Print results summary
        evaluator.print_results_summary(results)
        
        # Save results
        evaluator.save_results(results, args.output_dir)
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(results, args.output_dir)
    
    # Save combined results
    combined_results_path = os.path.join(args.output_dir, 'combined_results.json')
    combined_results = {k: {kk: vv for kk, vv in v.items() 
                           if kk not in ['predictions', 'true_labels']} 
                       for k, v in all_results.items()}
    
    with open(combined_results_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"Combined results saved to {combined_results_path}")
    
    # Print comparison summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Dataset':<20} {'Accuracy':<12} {'Weighted F1':<12} {'Samples':<10}")
        print("-" * 80)
        
        for dataset_name, results in all_results.items():
            print(f"{dataset_name:<20} {results['accuracy']:<12.4f} "
                  f"{results['weighted_f1']:<12.4f} {results['num_samples']:<10}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main() 