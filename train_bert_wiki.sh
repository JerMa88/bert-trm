#!/bin/bash
# Complete pipeline: Download Wikipedia data, build dataset, train BERT, and test

set -e  # Exit on error

echo "======================================================================"
echo "BERT Training Pipeline on Wikipedia Data"
echo "======================================================================"
echo

# Step 1: Download Wikipedia data (if not already downloaded)
if [ ! -f "data/wikipedia/wiki_text.txt" ]; then
    echo "Step 1: Downloading Wikipedia articles..."
    python download_wiki_text.py
else
    echo "Step 1: Wikipedia data already exists, skipping download"
fi

echo
echo "Step 2: Building MLM dataset from Wikipedia..."
python -m dataset.build_text_mlm_dataset \
    --input-files data/wikipedia/wiki_text.txt \
    --output-dir data/wiki-mlm \
    --vocab-size 10000 \
    --max-seq-length 256 \
    --mask-prob 0.15 \
    --seed 42

echo
echo "Step 3: Training BERT model..."
export PYTHONPATH=/Users/zma/Documents/GitHub/bert-trm
export WANDB_MODE=disabled

python pretrain.py \
    arch=bert \
    data_paths="[data/wiki-mlm]" \
    global_batch_size=8 \
    epochs=100 \
    eval_interval=20 \
    lr=1e-4 \
    arch.num_layers=4 \
    arch.hidden_size=256 \
    arch.num_heads=8 \
    arch.max_position_embeddings=256 \
    +run_name=bert_wiki \
    +project_name=bert-wiki-training

echo
echo "Step 4: Finding latest checkpoint..."
CHECKPOINT=$(ls -t outputs/bert/*/checkpoint_*.pt | head -1)
echo "Latest checkpoint: $CHECKPOINT"

echo
echo "Step 5: Testing trained model..."
python test_bert_model.py \
    --checkpoint "$CHECKPOINT" \
    --vocab data/wiki-mlm/vocab.json \
    --config data/wiki-mlm/train/dataset.json

echo
echo "======================================================================"
echo "Training and testing complete!"
echo "======================================================================"
echo
echo "To test interactively, run:"
echo "  python test_bert_model.py --checkpoint \"$CHECKPOINT\" --vocab data/wiki-mlm/vocab.json --config data/wiki-mlm/train/dataset.json --interactive"
