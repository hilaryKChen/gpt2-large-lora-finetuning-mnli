#!/usr/bin/env python3
"""
Comparison script to analyze baseline vs fine-tuned model performance.
Loads results from both evaluations and creates detailed comparisons.
"""

import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results(baseline_dir: str, finetuned_dir: str):
    """Load baseline and fine-tuned results."""
    results = {}
    
    # Load baseline results
    baseline_path = os.path.join(baseline_dir, 'combined_baseline_results.json')
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            results['baseline'] = json.load(f)
        print(f"✅ Loaded baseline results from {baseline_path}")
    else:
        print(f"❌ Baseline results not found at {baseline_path}")
        return None
    
    # Load fine-tuned results
    finetuned_path = os.path.join(finetuned_dir, 'combined_results.json')
    if os.path.exists(finetuned_path):
        with open(finetuned_path, 'r') as f:
            results['finetuned'] = json.load(f)
        print(f"✅ Loaded fine-tuned results from {finetuned_path}")
    else:
        print(f"❌ Fine-tuned results not found at {finetuned_path}")
        return None
    
    return results

def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create a comparison table of key metrics."""
    comparison_data = []
    
    for dataset_name in ['test_matched', 'test_mismatched']:
        if dataset_name in results['baseline'] and dataset_name in results['finetuned']:
            baseline_data = results['baseline'][dataset_name]
            finetuned_data = results['finetuned'][dataset_name]
            
            # Overall metrics
            comparison_data.append({
                'Dataset': dataset_name.replace('test_', '').title(),
                'Model': 'Baseline (Pretrained)',
                'Accuracy': baseline_data['accuracy'],
                'Precision': baseline_data['weighted_precision'],
                'Recall': baseline_data['weighted_recall'],
                'F1-Score': baseline_data['weighted_f1']
            })
            
            comparison_data.append({
                'Dataset': dataset_name.replace('test_', '').title(),
                'Model': 'Fine-tuned (LoRA)',
                'Accuracy': finetuned_data['accuracy'],
                'Precision': finetuned_data['weighted_precision'],
                'Recall': finetuned_data['weighted_recall'],
                'F1-Score': finetuned_data['weighted_f1']
            })
    
    return pd.DataFrame(comparison_data)

def calculate_improvements(results: dict) -> dict:
    """Calculate improvement percentages."""
    improvements = {}
    
    for dataset_name in ['test_matched', 'test_mismatched']:
        if dataset_name in results['baseline'] and dataset_name in results['finetuned']:
            baseline = results['baseline'][dataset_name]
            finetuned = results['finetuned'][dataset_name]
            
            improvements[dataset_name] = {
                'accuracy_improvement': ((finetuned['accuracy'] - baseline['accuracy']) / baseline['accuracy']) * 100,
                'f1_improvement': ((finetuned['weighted_f1'] - baseline['weighted_f1']) / baseline['weighted_f1']) * 100,
                'precision_improvement': ((finetuned['weighted_precision'] - baseline['weighted_precision']) / baseline['weighted_precision']) * 100,
                'recall_improvement': ((finetuned['weighted_recall'] - baseline['weighted_recall']) / baseline['weighted_recall']) * 100,
                'baseline_accuracy': baseline['accuracy'],
                'finetuned_accuracy': finetuned['accuracy'],
                'baseline_f1': baseline['weighted_f1'],
                'finetuned_f1': finetuned['weighted_f1']
            }
    
    return improvements

def plot_comparison_charts(results: dict, output_dir: str):
    """Create comparison visualization charts."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comparison data
    datasets = []
    baseline_acc = []
    finetuned_acc = []
    baseline_f1 = []
    finetuned_f1 = []
    
    for dataset_name in ['test_matched', 'test_mismatched']:
        if dataset_name in results['baseline'] and dataset_name in results['finetuned']:
            datasets.append(dataset_name.replace('test_', '').title())
            baseline_acc.append(results['baseline'][dataset_name]['accuracy'])
            finetuned_acc.append(results['finetuned'][dataset_name]['accuracy'])
            baseline_f1.append(results['baseline'][dataset_name]['weighted_f1'])
            finetuned_f1.append(results['finetuned'][dataset_name]['weighted_f1'])
    
    # Create side-by-side bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Accuracy comparison
    ax1.bar(x - width/2, baseline_acc, width, label='Baseline (Pretrained)', alpha=0.8)
    ax1.bar(x + width/2, finetuned_acc, width, label='Fine-tuned (LoRA)', alpha=0.8)
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (baseline, finetuned) in enumerate(zip(baseline_acc, finetuned_acc)):
        ax1.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, finetuned + 0.01, f'{finetuned:.3f}', ha='center', va='bottom')
    
    # F1-Score comparison
    ax2.bar(x - width/2, baseline_f1, width, label='Baseline (Pretrained)', alpha=0.8)
    ax2.bar(x + width/2, finetuned_f1, width, label='Fine-tuned (LoRA)', alpha=0.8)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (baseline, finetuned) in enumerate(zip(baseline_f1, finetuned_f1)):
        ax2.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom')
        ax2.text(i + width/2, finetuned + 0.01, f'{finetuned:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'baseline_vs_finetuned_comparison.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison chart saved to {chart_path}")

def print_detailed_comparison(results: dict, improvements: dict):
    """Print detailed comparison results."""
    print("\n" + "="*80)
    print("📊 BASELINE vs FINE-TUNED MODEL COMPARISON")
    print("="*80)
    
    for dataset_name in ['test_matched', 'test_mismatched']:
        if dataset_name in improvements:
            imp = improvements[dataset_name]
            dataset_display = dataset_name.replace('test_', '').title()
            
            print(f"\n🎯 {dataset_display} Dataset Results:")
            print("-" * 50)
            print(f"Baseline Accuracy:    {imp['baseline_accuracy']:.4f}")
            print(f"Fine-tuned Accuracy:  {imp['finetuned_accuracy']:.4f}")
            print(f"Accuracy Improvement: {imp['accuracy_improvement']:+.2f}%")
            print()
            print(f"Baseline F1-Score:    {imp['baseline_f1']:.4f}")
            print(f"Fine-tuned F1-Score:  {imp['finetuned_f1']:.4f}")
            print(f"F1-Score Improvement: {imp['f1_improvement']:+.2f}%")
    
    # Overall summary
    avg_acc_imp = np.mean([imp['accuracy_improvement'] for imp in improvements.values()])
    avg_f1_imp = np.mean([imp['f1_improvement'] for imp in improvements.values()])
    
    print(f"\n🏆 OVERALL IMPROVEMENT SUMMARY:")
    print("-" * 50)
    print(f"Average Accuracy Improvement: {avg_acc_imp:+.2f}%")
    print(f"Average F1-Score Improvement: {avg_f1_imp:+.2f}%")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned model results")
    parser.add_argument("--baseline_dir", type=str, default="./baseline_results",
                       help="Directory containing baseline results")
    parser.add_argument("--finetuned_dir", type=str, default="./evaluation_results",
                       help="Directory containing fine-tuned model results")
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                       help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.baseline_dir, args.finetuned_dir)
    if results is None:
        print("❌ Could not load both baseline and fine-tuned results. Exiting.")
        return
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    
    # Save comparison table
    table_path = os.path.join(args.output_dir, 'comparison_table.csv')
    comparison_df.to_csv(table_path, index=False)
    print(f"📋 Comparison table saved to {table_path}")
    
    # Calculate improvements
    improvements = calculate_improvements(results)
    
    # Save improvements
    improvements_path = os.path.join(args.output_dir, 'improvements.json')
    with open(improvements_path, 'w') as f:
        json.dump(improvements, f, indent=2)
    print(f"📈 Improvements data saved to {improvements_path}")
    
    # Create visualization
    plot_comparison_charts(results, args.output_dir)
    
    # Print detailed comparison
    print_detailed_comparison(results, improvements)
    
    # Print table
    print(f"\n📋 DETAILED COMPARISON TABLE:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    main() 