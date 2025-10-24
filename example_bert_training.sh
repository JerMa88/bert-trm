#!/bin/bash
# Example BERT Training Script
# This script demonstrates the complete pipeline for training BERT on sample text data

set -e  # Exit on error

echo "==================================================================="
echo "BERT Training Example - Complete Pipeline"
echo "==================================================================="

# Step 1: Create sample text data
echo ""
echo "Step 1: Creating sample text data..."
mkdir -p data/sample_text

cat > data/sample_text/sample.txt << 'EOF'
The Transformer architecture is based on self-attention mechanisms.
BERT uses bidirectional training of Transformer encoders.
Masked language modeling helps BERT learn contextual representations.
Natural language processing has been revolutionized by pre-training.
Deep learning models require large amounts of training data.
Transfer learning allows models to adapt to new tasks quickly.
The attention mechanism computes weighted combinations of input representations.
Language models predict the probability of word sequences.
Pre-training on large corpora improves downstream task performance.
Bidirectional context is essential for understanding language semantics.
The vocabulary size determines the model's token representation capacity.
Transformers have become the dominant architecture in NLP.
Self-supervised learning eliminates the need for labeled data.
Fine-tuning adapts pre-trained models to specific applications.
Neural networks learn hierarchical representations of data.
EOF

echo "Created sample text file with 15 sentences"

# Step 2: Build MLM dataset
echo ""
echo "Step 2: Building MLM dataset from text..."
python dataset/build_text_mlm_dataset.py \
  --input-files data/sample_text/sample.txt \
  --output-dir data/sample-mlm \
  --vocab-size 1000 \
  --max-seq-length 128 \
  --mask-prob 0.15 \
  --seed 42

# Step 3: Train BERT model
echo ""
echo "Step 3: Training BERT model (small version for demo)..."
echo "This will train for 1000 steps. For production, use many more steps."

python pretrain.py \
  arch=bert \
  data_paths="[data/sample-mlm]" \
  global_batch_size=4 \
  epochs=1000 \
  eval_interval=100 \
  lr=1e-4 \
  arch.num_layers=2 \
  arch.hidden_size=128 \
  arch.num_heads=4 \
  +run_name=bert_demo \
  project_name=bert-trm-demo

echo ""
echo "==================================================================="
echo "Training complete!"
echo "==================================================================="
echo ""
echo "To view training logs:"
echo "  - Check wandb dashboard (if configured)"
echo "  - Look for checkpoint files in outputs/"
echo ""
echo "To train on larger datasets:"
echo "  1. Prepare larger text corpus (Wikipedia, BookCorpus, etc.)"
echo "  2. Rebuild dataset with larger vocab_size (e.g., 30000)"
echo "  3. Increase model size (num_layers=12, hidden_size=768 for BERT-Base)"
echo "  4. Train for 1M+ steps with multi-GPU setup"
echo ""
echo "See BERT_TRAINING.md for detailed instructions."
