"""
Build BERT Masked Language Modeling Dataset from Text Corpus

This script processes text data and creates a binary dataset with MLM masking
compatible with the PuzzleDataset loader.

Usage:
    python dataset/build_text_mlm_dataset.py \
        --input-files data/wiki.txt data/books.txt \
        --output-dir data/wiki-mlm \
        --vocab-size 30000 \
        --max-seq-length 512 \
        --mask-prob 0.15
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import random

from dataset.common import PuzzleDatasetMetadata


# Special tokens (matching BERT)
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3
MASK_ID = 4

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]
IGNORE_LABEL_ID = -100


def build_vocab(text_files: List[str], vocab_size: int = 30000) -> Dict[str, int]:
    """
    Build vocabulary from text files

    Args:
        text_files: List of paths to text files
        vocab_size: Maximum vocabulary size

    Returns:
        vocab: Dictionary mapping tokens to IDs
    """
    print(f"Building vocabulary from {len(text_files)} files...")

    # Count word frequencies
    word_counts = {}
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Counting words in {Path(text_file).name}"):
                # Simple whitespace tokenization (can replace with WordPiece)
                words = line.strip().split()
                for word in words:
                    word = word.lower()  # Lowercase
                    word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Build vocab: special tokens + most frequent words
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    # Add most frequent words (leaving room for special tokens)
    for word, _ in sorted_words[:vocab_size - len(SPECIAL_TOKENS)]:
        vocab[word] = len(vocab)

    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def tokenize_text(text: str, vocab: Dict[str, int]) -> List[int]:
    """
    Tokenize text using vocabulary

    Args:
        text: Input text
        vocab: Vocabulary dictionary

    Returns:
        token_ids: List of token IDs
    """
    words = text.strip().lower().split()
    token_ids = [vocab.get(word, UNK_ID) for word in words]
    return token_ids


def create_mlm_example(
    tokens: List[int],
    max_seq_length: int,
    vocab_size: int,
    mask_prob: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create MLM training example from token sequence

    Implements BERT's masking strategy:
    - Select 15% of tokens for prediction
    - Of selected: 80% -> [MASK], 10% -> random, 10% -> unchanged

    Args:
        tokens: List of token IDs
        max_seq_length: Maximum sequence length
        vocab_size: Vocabulary size
        mask_prob: Probability of masking a token

    Returns:
        inputs: Input sequence with masked tokens [seq_len]
        targets: Target sequence (IGNORE_LABEL_ID for non-masked) [seq_len]
    """
    # Add [CLS] at start and [SEP] at end
    tokens = [CLS_ID] + tokens + [SEP_ID]

    # Truncate if too long
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]

    seq_len = len(tokens)

    # Initialize inputs and targets
    inputs = np.array(tokens, dtype=np.int32)
    targets = np.full(seq_len, IGNORE_LABEL_ID, dtype=np.int32)

    # Select positions to mask (excluding [CLS], [SEP], [PAD])
    maskable_positions = [
        i for i in range(seq_len)
        if tokens[i] not in [CLS_ID, SEP_ID, PAD_ID]
    ]

    num_to_mask = max(1, int(len(maskable_positions) * mask_prob))
    masked_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))

    for pos in masked_positions:
        # Save original token as target
        targets[pos] = tokens[pos]

        # Apply masking strategy
        rand = random.random()
        if rand < 0.8:
            # 80%: Replace with [MASK]
            inputs[pos] = MASK_ID
        elif rand < 0.9:
            # 10%: Replace with random token
            inputs[pos] = random.randint(len(SPECIAL_TOKENS), vocab_size - 1)
        # else: 10%: Keep original token

    # Pad to max_seq_length
    if seq_len < max_seq_length:
        pad_len = max_seq_length - seq_len
        inputs = np.pad(inputs, (0, pad_len), constant_values=PAD_ID)
        targets = np.pad(targets, (0, pad_len), constant_values=IGNORE_LABEL_ID)

    return inputs, targets


def process_text_files(
    text_files: List[str],
    vocab: Dict[str, int],
    output_dir: str,
    max_seq_length: int = 512,
    mask_prob: float = 0.15,
    split: str = "train"
) -> Tuple[int, int]:
    """
    Process text files and create MLM dataset

    Args:
        text_files: List of text file paths
        vocab: Vocabulary dictionary
        output_dir: Output directory
        max_seq_length: Maximum sequence length
        mask_prob: Masking probability
        split: Dataset split name

    Returns:
        num_examples: Number of examples created
        num_groups: Number of groups (1 for text data)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_file = output_dir / f"{split}_inputs.npy"
    targets_file = output_dir / f"{split}_targets.npy"

    all_inputs = []
    all_targets = []

    print(f"Processing {len(text_files)} text files for {split} split...")

    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {Path(text_file).name}"):
                line = line.strip()
                if not line:
                    continue

                # Tokenize
                tokens = tokenize_text(line, vocab)

                if len(tokens) < 5:  # Skip very short sequences
                    continue

                # Create MLM example
                inputs, targets = create_mlm_example(
                    tokens, max_seq_length, len(vocab), mask_prob
                )

                all_inputs.append(inputs)
                all_targets.append(targets)

    # Save as numpy arrays
    inputs_array = np.stack(all_inputs, axis=0)
    targets_array = np.stack(all_targets, axis=0)

    np.save(inputs_file, inputs_array)
    np.save(targets_file, targets_array)

    print(f"Saved {len(all_inputs)} examples to {output_dir}")
    print(f"  Inputs shape: {inputs_array.shape}")
    print(f"  Targets shape: {targets_array.shape}")

    return len(all_inputs), 1  # Single group for all text data


def main():
    parser = argparse.ArgumentParser(description="Build BERT MLM dataset from text")
    parser.add_argument("--input-files", nargs="+", required=True, help="Input text files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=30000, help="Vocabulary size")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Masking probability")
    parser.add_argument("--test-split", type=float, default=0.05, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build vocabulary
    vocab = build_vocab(args.input_files, args.vocab_size)

    # Save vocabulary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = output_dir / "vocab.json"
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary to {vocab_file}")

    # Split files into train/test
    num_test = max(1, int(len(args.input_files) * args.test_split))
    test_files = args.input_files[:num_test]
    train_files = args.input_files[num_test:]

    # Process train set
    num_train, num_groups = process_text_files(
        train_files, vocab, args.output_dir,
        args.max_seq_length, args.mask_prob, split="train"
    )

    # Process test set
    num_test, _ = process_text_files(
        test_files, vocab, args.output_dir,
        args.max_seq_length, args.mask_prob, split="test"
    )

    # Create metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=args.max_seq_length,
        vocab_size=len(vocab),
        pad_id=PAD_ID,
        ignore_label_id=IGNORE_LABEL_ID,
        blank_identifier_id=0,
        num_puzzle_identifiers=num_groups,
        sets={
            "train": {"num_examples": num_train, "num_groups": num_groups},
            "test": {"num_examples": num_test, "num_groups": 1}
        }
    )

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata.model_dump(), f, indent=2)
    print(f"Saved metadata to {metadata_file}")

    print("\nDataset creation complete!")
    print(f"  Train examples: {num_train}")
    print(f"  Test examples: {num_test}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Max sequence length: {args.max_seq_length}")


if __name__ == "__main__":
    main()
