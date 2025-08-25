#!/bin/bash

# GPT2-Large LoRA Fine-tuning Pipeline for MultiNLI
# Optimized for 36G vGPU

set -e  # Exit on any error

echo "=========================================="
echo "GPT2-Large LoRA Fine-tuning on MultiNLI"
echo "=========================================="

# Configuration
DATA_DIR="./processed_data"
OUTPUT_DIR="./gpt2_lora_multinli"
EVAL_OUTPUT_DIR="./evaluation_results"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure CUDA is properly installed."
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

# Step 2: Preprocess data
echo "Step 2: Preprocessing MultiNLI data from HuggingFace..."
if [ ! -d "$DATA_DIR" ]; then
    echo "Loading MultiNLI dataset (nyu-mll/multi_nli) from HuggingFace..."
    echo "This will create new validation files and process training data..."
    python data_preprocessing.py \
        --output_dir "$DATA_DIR" \
        --max_train_samples 50000
    echo "Data preprocessing completed!"
    echo "New validation files created: dev_matched_sampled_hf.jsonl and dev_mismatched_sampled_hf.jsonl"
else
    echo "Processed data already exists. Skipping preprocessing..."
fi

# Step 3: Fine-tune GPT2-large with LoRA
echo "Step 3: Starting LoRA fine-tuning..."

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Training with optimized settings for 36G vGPU using config file
python train_gpt2_lora.py \
    --config "config_36g_vgpu.json" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Fine-tuning completed!"

# Step 4: Evaluate the model
echo "Step 4: Evaluating fine-tuned model..."

# Find the final model path
FINAL_MODEL_PATH="$OUTPUT_DIR/final_model"

if [ -d "$FINAL_MODEL_PATH" ]; then
    python evaluate_gpt2_lora.py \
        --model_path "$FINAL_MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$EVAL_OUTPUT_DIR" \
        --base_model "gpt2-large"
    
    echo "Evaluation completed!"
    echo "Results saved to: $EVAL_OUTPUT_DIR"
else
    echo "Error: Final model not found at $FINAL_MODEL_PATH"
    exit 1
fi

# Step 5: Display results summary
echo "=========================================="
echo "TRAINING AND EVALUATION COMPLETED"
echo "=========================================="
echo "Model saved to: $OUTPUT_DIR"
echo "Evaluation results: $EVAL_OUTPUT_DIR"
echo ""
echo "Key files:"
echo "- Training config: $OUTPUT_DIR/training_config.json"
echo "- Model weights: $FINAL_MODEL_PATH"
echo "- Evaluation results: $EVAL_OUTPUT_DIR/combined_results.json"
echo "- Confusion matrices: $EVAL_OUTPUT_DIR/confusion_matrix_*.png"
echo ""

# Display quick results if available
if [ -f "$EVAL_OUTPUT_DIR/combined_results.json" ]; then
    echo "Quick Results Summary:"
    python -c "
import json
with open('$EVAL_OUTPUT_DIR/combined_results.json', 'r') as f:
    results = json.load(f)
for dataset_name, metrics in results.items():
    print(f'{dataset_name}: Accuracy = {metrics[\"accuracy\"]:.4f}, F1 = {metrics[\"weighted_f1\"]:.4f}')
"
fi

echo "==========================================" 