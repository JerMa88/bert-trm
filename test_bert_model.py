#!/usr/bin/env python3
"""
Test a trained BERT model with masked language modeling predictions.
"""

import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.recursive_reasoning.bert_mlm import BERTModel
from models.losses import BERTMLMLoss


def load_vocab(vocab_path):
    """Load vocabulary from JSON file"""
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    # Create reverse mapping
    id_to_token = {v: k for k, v in vocab.items()}
    return vocab, id_to_token


def load_model(checkpoint_path, config_dict):
    """Load trained BERT model from checkpoint"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    with torch.device(device):
        model = BERTModel(config_dict)
        model = BERTMLMLoss(model, loss_type='softmax_cross_entropy', use_nsp=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, device


def tokenize(text, vocab, max_length=128):
    """Tokenize text using vocabulary"""
    # Simple word-level tokenization
    tokens = text.lower().split()

    # Convert to IDs
    token_ids = []
    for token in tokens:
        if token in vocab:
            token_ids.append(vocab[token])
        else:
            token_ids.append(vocab.get('[UNK]', 3))  # UNK token

    # Truncate or pad
    if len(token_ids) > max_length - 2:
        token_ids = token_ids[:max_length - 2]

    # Add [CLS] and [SEP]
    token_ids = [vocab['[CLS]']] + token_ids + [vocab['[SEP]']]

    # Pad to max_length
    padding_length = max_length - len(token_ids)
    token_ids += [vocab['[PAD]']] * padding_length

    return token_ids


def predict_masked_tokens(model, inputs, vocab, id_to_token, device, top_k=5):
    """
    Predict masked tokens in the input

    Args:
        model: Trained BERT model
        inputs: Token IDs with [MASK] tokens
        vocab: Vocabulary dictionary
        id_to_token: Reverse vocabulary mapping
        device: Device to run on
        top_k: Number of top predictions to return

    Returns:
        predictions: List of (position, [(token, prob), ...])
    """
    mask_id = vocab['[MASK]']

    # Find masked positions
    masked_positions = [i for i, token_id in enumerate(inputs) if token_id == mask_id]

    if not masked_positions:
        return []

    # Prepare batch
    inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(device)
    labels_tensor = torch.zeros_like(inputs_tensor) - 100  # Dummy labels

    batch = {
        'inputs': inputs_tensor,
        'labels': labels_tensor,
        'puzzle_identifiers': torch.zeros((1,), dtype=torch.long).to(device)
    }

    # Get predictions
    with torch.no_grad():
        carry = model.initial_carry(batch)
        carry, loss, metrics, preds, _ = model(carry=carry, batch=batch, return_keys=['mlm_logits'])

        # Get MLM logits
        mlm_logits = preds['mlm_logits']  # Shape: (batch_size, seq_len, vocab_size)

    # Get predictions for masked positions
    predictions = []
    for pos in masked_positions:
        # Get logits for this position
        logits = mlm_logits[0, pos, :]  # Shape: (vocab_size,)

        # Get top-k predictions
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)

        # Convert to tokens
        top_predictions = [
            (id_to_token[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

        predictions.append((pos, top_predictions))

    return predictions


def interactive_test(model, vocab, id_to_token, device):
    """Interactive testing mode"""
    print("\n" + "="*70)
    print("BERT Masked Language Model - Interactive Testing")
    print("="*70)
    print("\nInstructions:")
    print("  - Type a sentence with [MASK] tokens where you want predictions")
    print("  - Example: 'The cat sat on the [MASK]'")
    print("  - Type 'quit' to exit")
    print()

    while True:
        try:
            # Get input
            text = input("\nEnter text (with [MASK]): ").strip()

            if text.lower() == 'quit':
                break

            if '[MASK]' not in text:
                print("⚠ No [MASK] token found in input")
                continue

            # Tokenize
            token_ids = tokenize(text, vocab)

            # Get predictions
            predictions = predict_masked_tokens(
                model, token_ids, vocab, id_to_token, device, top_k=5
            )

            # Display results
            print("\n" + "-"*70)
            print("Top predictions:")
            for pos, top_preds in predictions:
                print(f"\nPosition {pos}:")
                for i, (token, prob) in enumerate(top_preds, 1):
                    print(f"  {i}. {token:20s} ({prob*100:5.2f}%)")
            print("-"*70)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_test(model, test_cases, vocab, id_to_token, device):
    """Test model on predefined test cases"""
    print("\n" + "="*70)
    print("BERT Masked Language Model - Batch Testing")
    print("="*70)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {test_case}")
        print(f"{'='*70}")

        # Tokenize
        token_ids = tokenize(test_case, vocab)

        # Get predictions
        predictions = predict_masked_tokens(
            model, token_ids, vocab, id_to_token, device, top_k=5
        )

        # Display results
        if predictions:
            for pos, top_preds in predictions:
                print(f"\nMasked position {pos}:")
                for j, (token, prob) in enumerate(top_preds, 1):
                    print(f"  {j}. {token:20s} ({prob*100:5.2f}%)")
        else:
            print("  No [MASK] tokens found")


def main():
    parser = argparse.ArgumentParser(description="Test BERT model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", required=True, help="Path to vocabulary file")
    parser.add_argument("--config", required=True, help="Path to dataset metadata (for config)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    print("Loading model and vocabulary...")

    # Load vocabulary
    vocab, id_to_token = load_vocab(args.vocab)
    print(f"✓ Vocabulary loaded: {len(vocab)} tokens")

    # Load metadata to get config
    with open(args.config, 'r') as f:
        metadata = json.load(f)

    # Create model config
    config_dict = {
        'vocab_size': metadata['vocab_size'],
        'seq_len': metadata['seq_len'],
        'hidden_size': 128,
        'num_layers': 2,
        'num_heads': 4,
        'expansion': 4,
        'max_position_embeddings': metadata['seq_len'],
        'type_vocab_size': 2,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'rms_norm_eps': 1e-5,
        'forward_dtype': 'bfloat16',
        'batch_size': 1,
        'num_puzzle_identifiers': 1,
        'causal': False
    }

    # Load model
    model, device = load_model(args.checkpoint, config_dict)
    print(f"✓ Model loaded from: {args.checkpoint}")
    print(f"  Device: {device}")

    if args.interactive:
        # Interactive mode
        interactive_test(model, vocab, id_to_token, device)
    else:
        # Batch test mode
        test_cases = [
            "The cat sat on the [MASK]",
            "I like to eat [MASK] for breakfast",
            "The weather is [MASK] today",
            "She went to the [MASK] to buy groceries",
            "The [MASK] is shining brightly in the sky"
        ]

        batch_test(model, test_cases, vocab, id_to_token, device)

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()
