"""
BERT: Bidirectional Encoder Representations from Transformers
for Masked Language Modeling (MLM)
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, CastedEmbedding, CastedLinear

IGNORE_LABEL_ID = -100


class BERTConfig(BaseModel):
    batch_size: int
    seq_len: int  # max sequence length (e.g., 512)
    vocab_size: int  # vocabulary size (e.g., 30000)

    # Transformer config
    num_layers: int  # number of transformer layers (e.g., 12 for BERT-Base)
    hidden_size: int  # hidden dimension (e.g., 768 for BERT-Base)
    num_heads: int  # number of attention heads (e.g., 12)
    expansion: float = 4.0  # FFN expansion factor (intermediate_size = hidden_size * expansion)

    # Embeddings
    max_position_embeddings: int = 512
    type_vocab_size: int = 2  # for segment embeddings (sentence A/B)

    # MLM config
    mask_prob: float = 0.15  # percentage of tokens to mask

    # Optional config
    rms_norm_eps: float = 1e-5
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    forward_dtype: str = "bfloat16"
    use_nsp: bool = False  # Next Sentence Prediction task

    # Compatibility with TRM interface (ignored for BERT)
    num_puzzle_identifiers: int = 0
    puzzle_emb_ndim: int = 0


class BERTBlock(nn.Module):
    """Single BERT Transformer block with bidirectional self-attention"""

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.config = config

        # Self-attention (bidirectional)
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False  # Bidirectional attention for BERT
        )

        # Feed-forward network
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] optional mask
        """
        # Self-attention with residual connection and post-norm
        attn_out = self.self_attn(
            hidden_states=hidden_states,
            cos_sin=None  # BERT uses learned position embeddings, not RoPE
        )
        hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)

        # Feed-forward with residual connection and post-norm
        mlp_out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)

        return hidden_states


@dataclass
class BERTCarry:
    """Dummy carry for compatibility with TRM training loop"""
    pass


class BERTModel(nn.Module):
    """
    BERT Model for Masked Language Modeling

    Key differences from TRM:
    - No recursive reasoning (single forward pass)
    - Bidirectional self-attention (can attend to all tokens)
    - Learned position embeddings (not RoPE)
    - Token, segment, and position embeddings
    - MLM head predicts only masked tokens
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = BERTConfig(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Embedding layers
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embeddings
        self.token_embeddings = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )

        # Position embeddings (learned)
        self.position_embeddings = CastedEmbedding(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )

        # Segment embeddings (for sentence A/B distinction)
        self.token_type_embeddings = CastedEmbedding(
            self.config.type_vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            BERTBlock(self.config) for _ in range(self.config.num_layers)
        ])

        # Output heads
        # MLM head: predict masked tokens
        self.mlm_head = CastedLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False
        )

        # Optional: Next Sentence Prediction head
        if self.config.use_nsp:
            self.nsp_head = CastedLinear(self.config.hidden_size, 2, bias=True)

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine token, position, and segment embeddings

        Args:
            input_ids: [batch_size, seq_len] token indices
            token_type_ids: [batch_size, seq_len] segment indices (0 or 1)
            position_ids: [batch_size, seq_len] position indices

        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embeddings(input_ids.to(torch.int32))

        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids.to(torch.int32))

        # Segment embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        segment_embeds = self.token_type_embeddings(token_type_ids.to(torch.int32))

        # Sum all embeddings (as in original BERT)
        embeddings = token_embeds + position_embeds + segment_embeds

        # Scale
        embeddings = self.embed_scale * embeddings

        return embeddings

    def forward(
        self,
        carry: BERTCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[BERTCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass for BERT MLM

        Args:
            carry: Dummy carry for compatibility
            batch: Dictionary containing:
                - 'inputs': [batch_size, seq_len] input token ids
                - 'targets': [batch_size, seq_len] target token ids (with IGNORE_LABEL_ID for non-masked)
                - 'token_type_ids': [batch_size, seq_len] optional segment ids

        Returns:
            carry: Dummy carry
            outputs: Dictionary containing:
                - 'logits': [batch_size, seq_len, vocab_size] MLM predictions
                - 'nsp_logits': [batch_size, 2] NSP predictions (if use_nsp=True)
        """
        input_ids = batch['inputs']
        token_type_ids = batch.get('token_type_ids', None)

        # Get combined embeddings
        hidden_states = self._get_embeddings(input_ids, token_type_ids)

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # MLM predictions for all positions
        mlm_logits = self.mlm_head(hidden_states)

        outputs = {
            'logits': mlm_logits,
            'q_logits': (torch.zeros(input_ids.shape[0], device=input_ids.device),
                        torch.zeros(input_ids.shape[0], device=input_ids.device))  # Dummy for compatibility
        }

        # Optional: NSP predictions using [CLS] token (first token)
        if self.config.use_nsp:
            cls_hidden = hidden_states[:, 0]  # [batch_size, hidden_size]
            nsp_logits = self.nsp_head(cls_hidden)
            outputs['nsp_logits'] = nsp_logits

        return carry, outputs

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> BERTCarry:
        """Return dummy carry for compatibility with TRM training loop"""
        return BERTCarry()

    @property
    def puzzle_emb(self):
        """Dummy property for compatibility with TRM training loop"""
        return None
