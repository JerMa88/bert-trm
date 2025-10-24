# BERT Training Guide

This repository now supports training BERT (Bidirectional Encoder Representations from Transformers) for Masked Language Modeling, in addition to the original TRM (Tiny Recursion Model) for puzzle-solving tasks.

## Overview

The BERT implementation follows the original BERT paper ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)) and includes:

- **Bidirectional Transformer Encoder** with learned position embeddings
- **Masked Language Modeling (MLM)** pre-training objective
- **Optional Next Sentence Prediction (NSP)** task
- Compatible with the existing TRM training infrastructure

## Key Differences: BERT vs TRM

| Feature | TRM (Original) | BERT (New) |
|---------|----------------|------------|
| **Task** | Puzzle solving (supervised) | Language modeling (self-supervised) |
| **Architecture** | Recursive reasoning with deep supervision | Standard bidirectional Transformer |
| **Forward Pass** | Up to 16 recursive improvement steps | Single forward pass |
| **Training Data** | Question-answer pairs (puzzles) | Raw text with automatic masking |
| **Parameters** | 7M (tiny network) | 110M (BERT-Base) or 340M (BERT-Large) |

## Quick Start

### 1. Prepare Your Text Data

First, prepare your text corpus. You can use Wikipedia, BookCorpus, or any text files:

```bash
# Example: Create a simple text file
echo "This is a sample sentence for BERT training." > data/sample.txt
echo "BERT learns bidirectional representations from unlabeled text." >> data/sample.txt
echo "Masked language modeling is the key pre-training task." >> data/sample.txt
```

### 2. Build MLM Dataset

Convert your text data into the MLM dataset format:

```bash
python dataset/build_text_mlm_dataset.py \
  --input-files data/sample.txt \
  --output-dir data/sample-mlm \
  --vocab-size 30000 \
  --max-seq-length 512 \
  --mask-prob 0.15
```

**Parameters:**
- `--input-files`: One or more text files to process
- `--output-dir`: Output directory for the processed dataset
- `--vocab-size`: Maximum vocabulary size (default: 30000)
- `--max-seq-length`: Maximum sequence length (default: 512)
- `--mask-prob`: Probability of masking tokens (default: 0.15, as in BERT paper)
- `--test-split`: Fraction of data for testing (default: 0.05)

**For larger datasets (recommended for BERT):**

```bash
# Wikipedia + BookCorpus example
python dataset/build_text_mlm_dataset.py \
  --input-files data/wiki/*.txt data/books/*.txt \
  --output-dir data/wiki-books-mlm \
  --vocab-size 30000 \
  --max-seq-length 512 \
  --mask-prob 0.15
```

### 3. Train BERT Model

Train BERT using the processed dataset:

```bash
# Single GPU training
python pretrain.py \
  arch=bert \
  data_paths="[data/sample-mlm]" \
  global_batch_size=32 \
  lr=1e-4 \
  epochs=100000 \
  +run_name=my_bert_model

# Multi-GPU training (recommended for large-scale BERT)
torchrun --nproc-per-node 4 pretrain.py \
  arch=bert \
  data_paths="[data/wiki-books-mlm]" \
  global_batch_size=256 \
  lr=1e-4 \
  epochs=1000000 \
  +run_name=bert_base
```

**Training Parameters:**
- `arch=bert`: Use BERT architecture (defined in `config/arch/bert.yaml`)
- `data_paths`: List of dataset paths
- `global_batch_size`: Total batch size across all GPUs (BERT paper uses 256)
- `lr`: Learning rate (BERT paper uses 1e-4)
- `epochs`: Number of training epochs

### 4. Monitor Training

Training metrics are logged to Weights & Biases (wandb):

```bash
# Login to wandb (first time only)
wandb login YOUR_API_KEY

# Metrics tracked:
# - mlm_loss: Masked language modeling loss
# - accuracy: Token-level accuracy on masked tokens
# - exact_accuracy: Sequence-level accuracy
```

## Configuration

### BERT-Base (110M parameters)

The default configuration in `config/arch/bert.yaml`:

```yaml
num_layers: 12       # Transformer layers
hidden_size: 768     # Hidden dimension
num_heads: 12        # Attention heads
expansion: 4         # FFN expansion (intermediate_size = 3072)
max_position_embeddings: 512
vocab_size: 30000    # Set during dataset creation
```

### BERT-Large (340M parameters)

To train BERT-Large, modify `config/arch/bert.yaml`:

```yaml
num_layers: 24
hidden_size: 1024
num_heads: 16
expansion: 4  # intermediate_size = 4096
```

### Custom Tiny-BERT

For faster experimentation with smaller models:

```yaml
num_layers: 4
hidden_size: 256
num_heads: 4
expansion: 4
```

## Masking Strategy

The MLM masking follows BERT's original strategy:

1. **Select 15% of tokens** randomly for prediction
2. Of the selected tokens:
   - **80%**: Replace with `[MASK]` token
   - **10%**: Replace with a random token
   - **10%**: Keep unchanged

This strategy prevents the model from simply memorizing that `[MASK]` means "predict this token" during fine-tuning (when `[MASK]` doesn't appear).

## Advanced Features

### Next Sentence Prediction (NSP)

To enable NSP task (as in original BERT):

1. Modify `config/arch/bert.yaml`:
   ```yaml
   loss:
     use_nsp: true
   ```

2. Update your dataset builder to include sentence pairs and NSP labels

### Custom Tokenization

The default implementation uses simple whitespace tokenization. For production use, consider:

- **WordPiece tokenization** (as in original BERT)
- **BPE (Byte-Pair Encoding)**
- **SentencePiece**

Replace the `tokenize_text()` function in `dataset/build_text_mlm_dataset.py` with your preferred tokenizer.

### Fine-tuning on Downstream Tasks

After pre-training, you can fine-tune BERT on specific tasks:

1. **Classification**: Add a classification head on top of `[CLS]` token
2. **Question Answering**: Add span prediction heads
3. **Named Entity Recognition**: Add token-level classification

## Example: Full Training Pipeline

```bash
# 1. Download text data (example: Wikipedia dump)
# wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# python wikiextractor/WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 -o data/wiki/

# 2. Build MLM dataset
python dataset/build_text_mlm_dataset.py \
  --input-files data/wiki/**/*.txt \
  --output-dir data/wiki-mlm \
  --vocab-size 30000 \
  --max-seq-length 512 \
  --mask-prob 0.15

# 3. Train BERT-Base on 4 GPUs
torchrun --nproc-per-node 4 pretrain.py \
  arch=bert \
  data_paths="[data/wiki-mlm]" \
  global_batch_size=256 \
  lr=1e-4 \
  weight_decay=0.01 \
  lr_warmup_steps=10000 \
  epochs=1000000 \
  eval_interval=10000 \
  +run_name=bert_base_wiki

# 4. Monitor training (in another terminal)
wandb login
# View results at https://wandb.ai/your-username/bert-trm
```

## Training Time Estimates

| Model | GPUs | Batch Size | Steps | Estimated Time |
|-------|------|------------|-------|----------------|
| BERT-Base | 4x H100 | 256 | 1M | ~3-5 days |
| BERT-Large | 8x H100 | 256 | 1M | ~7-10 days |
| Tiny-BERT | 1x L40S | 32 | 100k | ~1-2 days |

Original BERT paper training details:
- **Data**: 3.3B words (BooksCorpus + Wikipedia)
- **Steps**: 1M steps
- **Batch size**: 256 sequences × 512 tokens = 131,072 tokens/batch
- **Hardware**: 4-16 Cloud TPUs

## Comparison with TRM

You can still use the original TRM for puzzle-solving tasks:

```bash
# TRM for ARC-AGI
python pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=trm_arc

# BERT for language modeling
python pretrain.py \
  arch=bert \
  data_paths="[data/wiki-mlm]" \
  global_batch_size=256 \
  +run_name=bert_wiki
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `global_batch_size`
- Reduce `max_seq_length`
- Use gradient checkpointing (to be implemented)
- Use smaller model (reduce `num_layers` or `hidden_size`)

### Poor Accuracy
- Check vocabulary coverage (increase `vocab_size`)
- Increase training steps
- Verify masking probability (should be ~15%)
- Check data quality (remove duplicates, corrupted text)

### Slow Training
- Increase `global_batch_size` (with more GPUs)
- Use `bfloat16` or `float16` (already enabled by default)
- Optimize data loading (more workers)

## References

- **BERT Paper**: [Devlin et al., 2018 - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- **TRM Paper**: [Jolicoeur-Martineau, 2025 - Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)

## Citation

If you use this BERT implementation, please cite both:

```bibtex
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

@article{jolicoeurmartineau2025trm,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}
```
