from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class BERTMLMLoss(nn.Module):
    """
    Loss function for BERT Masked Language Modeling

    Computes cross-entropy loss only on masked tokens (ignoring non-masked positions).
    Optionally includes Next Sentence Prediction (NSP) loss.
    """

    def __init__(self, model: nn.Module, loss_type: str = "softmax_cross_entropy", use_nsp: bool = False):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.use_nsp = use_nsp

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    @property
    def puzzle_emb(self):
        """For compatibility with TRM training loop"""
        return self.model.puzzle_emb

    def forward(
        self,
        return_keys: Sequence[str],
        carry: Any = None,
        batch: Dict[str, torch.Tensor] = None,
        **kwargs
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], bool]:
        """
        Forward pass with MLM loss computation

        Args:
            return_keys: Keys to return in detached_outputs
            carry: Model carry (dummy for BERT)
            batch: Dictionary containing:
                - 'inputs': [batch_size, seq_len] input token ids
                - 'targets' or 'labels': [batch_size, seq_len] target labels
                  (IGNORE_LABEL_ID for non-masked positions)
                - 'nsp_labels': [batch_size] NSP labels (optional)

        Returns:
            carry: Updated carry
            loss: Combined MLM + NSP loss
            metrics: Dictionary of metrics
            detached_outputs: Requested output tensors
            all_halted: Always True for BERT (no iterative refinement)
        """
        # Forward pass through BERT model
        new_carry, outputs = self.model(carry, batch)

        # Get labels (support both 'labels' and 'targets' keys)
        labels = batch.get('labels', batch.get('targets'))
        if labels is None:
            raise ValueError("Batch must contain 'labels' or 'targets' key")

        # Compute MLM loss (only on masked positions)
        mlm_logits = outputs['logits']  # [batch_size, seq_len, vocab_size]

        # Mask for valid positions (not IGNORE_LABEL_ID)
        mask = (labels != IGNORE_LABEL_ID)
        loss_counts = mask.sum(-1)  # Number of masked tokens per sequence
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid division by zero

        # Compute cross-entropy loss on masked tokens
        mlm_loss_per_token = self.loss_fn(mlm_logits, labels, ignore_index=IGNORE_LABEL_ID)
        mlm_loss = (mlm_loss_per_token / loss_divisor).sum()

        total_loss = mlm_loss

        # Optional: NSP loss
        nsp_loss = torch.tensor(0.0, device=mlm_loss.device)
        if self.use_nsp and 'nsp_logits' in outputs:
            nsp_labels = batch.get('nsp_labels')
            if nsp_labels is not None:
                nsp_logits = outputs['nsp_logits']  # [batch_size, 2]
                nsp_loss = F.cross_entropy(
                    nsp_logits.to(torch.float32),
                    nsp_labels.to(torch.long),
                    reduction='mean'
                )
                total_loss = mlm_loss + nsp_loss

        # Compute metrics
        with torch.no_grad():
            # Predictions
            preds = torch.argmax(mlm_logits, dim=-1)
            outputs['preds'] = preds

            # Accuracy (only on masked positions)
            is_correct = mask & (preds == labels)
            token_accuracy = is_correct.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)

            # Sequence-level exact accuracy
            seq_is_correct = is_correct.sum(-1) == loss_counts
            exact_accuracy = seq_is_correct.sum() / labels.shape[0]

            metrics = {
                "count": torch.tensor(labels.shape[0], device=labels.device),
                "accuracy": token_accuracy * labels.shape[0],  # Will be averaged later
                "exact_accuracy": exact_accuracy * labels.shape[0],
                "lm_loss": mlm_loss.detach(),
                "mlm_loss": mlm_loss.detach(),
                "steps": torch.tensor(1.0, device=labels.device),  # Single pass for BERT
            }

            if self.use_nsp:
                metrics['nsp_loss'] = nsp_loss.detach()
                if 'nsp_labels' in batch and 'nsp_logits' in outputs:
                    nsp_preds = torch.argmax(outputs['nsp_logits'], dim=-1)
                    nsp_accuracy = (nsp_preds == batch['nsp_labels']).float().mean()
                    metrics['nsp_accuracy'] = nsp_accuracy * labels.shape[0]

            # Dummy Q-learning metrics for compatibility
            metrics['q_halt_loss'] = torch.tensor(0.0, device=labels.device)
            metrics['q_halt_accuracy'] = metrics['exact_accuracy']

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # Always "halted" for BERT (no iterative refinement)
        all_halted = True

        return new_carry, total_loss, metrics, detached_outputs, all_halted

