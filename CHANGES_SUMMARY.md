# Summary of Changes: BERT Implementation

## Overview

This codebase has been extended to support BERT (Bidirectional Encoder Representations from Transformers) for Masked Language Modeling, while maintaining full compatibility with the original TRM (Tiny Recursion Model) for puzzle-solving tasks.

## New Files Created

### 1. Model Architecture
- **`models/recursive_reasoning/bert_mlm.py`**
  - Complete BERT model implementation
  - Bidirectional Transformer encoder with learned position embeddings
  - Token, position, and segment embeddings
  - MLM prediction head
  - Optional Next Sentence Prediction (NSP) head
  - Compatible with existing TRM training infrastructure

### 2. Loss Functions
- **`models/losses.py`** (modified)
  - Added `BERTMLMLoss` class
  - Computes cross-entropy loss only on masked tokens
  - Supports optional NSP loss
  - Compatible with TRM training loop interface

### 3. Dataset Builder
- **`dataset/build_text_mlm_dataset.py`**
  - Processes raw text files into MLM dataset format
  - Builds vocabulary from corpus
  - Implements BERT's masking strategy:
    - 15% of tokens selected for prediction
    - 80% → [MASK], 10% → random, 10% → unchanged
  - Creates binary dataset compatible with `PuzzleDataset`

### 4. Configuration Files
- **`config/arch/bert.yaml`**
  - BERT-Base architecture (110M parameters)
  - 12 layers, 768 hidden size, 12 attention heads
  - Configurable for BERT-Large or custom sizes
  - Compatible with TRM config system

- **`config/cfg_bert_pretrain.yaml`**
  - BERT pre-training hyperparameters
  - Batch size: 256
  - Learning rate: 1e-4 with warmup
  - Optimizer: AdamW (β1=0.9, β2=0.999)
  - Weight decay: 0.01

### 5. Documentation
- **`BERT_TRAINING.md`**
  - Comprehensive guide for BERT training
  - Dataset preparation instructions
  - Training configuration examples
  - Comparison with TRM
  - Troubleshooting tips

- **`example_bert_training.sh`**
  - End-to-end example script
  - Demonstrates complete BERT training pipeline
  - Creates sample data, builds dataset, trains model

- **`CHANGES_SUMMARY.md`** (this file)
  - Summary of all modifications

## Key Design Decisions

### 1. **Compatibility with TRM Infrastructure**
All BERT components implement the same interfaces as TRM:
- `initial_carry()` - Returns dummy carry for BERT
- `forward()` - Compatible signature with TRM training loop
- `puzzle_emb` property - Returns None for BERT
- Loss functions follow same structure

This means the existing `pretrain.py` works for both TRM and BERT with minimal modifications.

### 2. **No Recursion for Standard BERT**
Unlike TRM's recursive reasoning approach:
- BERT uses a single forward pass
- No deep supervision (no iterative refinement)
- No ACT (Adaptive Computation Time) halting
- Simpler training loop

However, the architecture could be extended to support recursive refinement if desired.

### 3. **Modular Loss Function**
`BERTMLMLoss` is separate from the model:
- Supports different loss types (softmax, stablemax)
- Easy to add NSP task
- Can compute various metrics (accuracy, perplexity)

### 4. **Simple Tokenization (Extensible)**
Current implementation uses whitespace tokenization for simplicity:
- Easy to understand and debug
- Fast processing
- Can be replaced with WordPiece, BPE, or SentencePiece

### 5. **Flexible Dataset Format**
MLM dataset format matches `PuzzleDataset`:
- Binary numpy arrays for efficient loading
- Metadata JSON for configuration
- Support for train/test splits
- Easy to extend with more features

## How the Systems Differ

| Aspect | TRM | BERT |
|--------|-----|------|
| **Purpose** | Solve puzzles (ARC-AGI, Sudoku, Maze) | Language understanding |
| **Data** | Question-answer pairs | Raw text (self-supervised) |
| **Training** | Deep supervision + recursion | Standard supervised learning |
| **Architecture** | Tiny network (2 layers, 5-7M params) | Standard Transformer (12-24 layers, 110-340M params) |
| **Forward Pass** | Up to 16 recursive improvement steps | Single forward pass |
| **Loss** | ACTLossHead with Q-learning halting | BERTMLMLoss on masked tokens |
| **Position Encoding** | RoPE (Rotary Position Embeddings) | Learned absolute positions |
| **Use Case** | Hard reasoning tasks with small data | Language pre-training on large corpora |

## Usage Examples

### Train TRM (Original)
```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4
```

### Train BERT (New)
```bash
python pretrain.py \
  arch=bert \
  data_paths="[data/wiki-mlm]" \
  global_batch_size=256 \
  lr=1e-4
```

## Testing

To verify the BERT implementation works:

```bash
# Run the example script
./example_bert_training.sh

# Or manually:
# 1. Create sample data
echo "Sample text for testing." > data/test.txt

# 2. Build dataset
python dataset/build_text_mlm_dataset.py \
  --input-files data/test.txt \
  --output-dir data/test-mlm \
  --vocab-size 1000 \
  --max-seq-length 128

# 3. Train small model
python pretrain.py \
  arch=bert \
  data_paths="[data/test-mlm]" \
  global_batch_size=4 \
  epochs=100 \
  arch.num_layers=2 \
  arch.hidden_size=128
```

## Future Enhancements

Potential improvements that could be added:

1. **Advanced Tokenization**
   - WordPiece tokenizer (as in original BERT)
   - Byte-Pair Encoding (BPE)
   - SentencePiece integration

2. **Next Sentence Prediction**
   - Proper sentence pair sampling
   - NSP data preprocessing
   - Multi-task training

3. **Hybrid TRM-BERT**
   - BERT pre-training + TRM recursive fine-tuning
   - Combine language understanding with reasoning

4. **Evaluation Tasks**
   - Perplexity computation
   - Downstream task fine-tuning (classification, QA)
   - GLUE benchmark evaluation

5. **Optimization**
   - Gradient checkpointing for memory efficiency
   - Mixed precision training (already supported via bfloat16)
   - FlashAttention integration

6. **Data Pipeline**
   - Streaming dataset support for large corpora
   - On-the-fly masking during training
   - Better text preprocessing

## Backward Compatibility

All original TRM functionality is preserved:
- TRM models still work unchanged
- Original configs and datasets compatible
- No breaking changes to existing code
- BERT is purely additive

## References

1. **BERT Paper**: Devlin et al., 2018 - [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
2. **TRM Paper**: Jolicoeur-Martineau, 2025 - [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)

## Questions?

See `BERT_TRAINING.md` for detailed documentation or check the original TRM README for puzzle-solving tasks.
