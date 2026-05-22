import torch
import torch.nn as nn
import math
import sys
import os

# Allow running directly or as a module
try:
    from shared_modules.rms_norm import RMSNorm
    from shared_modules.gqa import GroupedQueryAttention
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from shared_modules.rms_norm import RMSNorm
    from shared_modules.gqa import GroupedQueryAttention


class TransformerBlock(nn.Module):
    """
    A single Transformer block with pre-norm architecture.
    Shared between BERT and GPT models.
    """
    def __init__(self, hidden_dim, n_heads, n_kv_heads, ffn_dim, dropout=0.1, is_causal=False):
        super().__init__()
        self.is_causal = is_causal

        # Pre-norm for attention
        self.attn_norm = RMSNorm(hidden_dim)

        # Grouped Query Attention
        self.attn = GroupedQueryAttention(hidden_dim, n_heads, n_kv_heads, dropout=dropout)

        # Pre-norm for FFN
        self.ffn_norm = RMSNorm(hidden_dim)

        # FFN: Linear(hidden_dim, ffn_dim) → GELU → Linear(ffn_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, hidden_dim)

        # --- Attention sublayer (pre-norm + residual) ---
        residual = x                               # shape: (batch, seq_len, hidden_dim)
        x = self.attn_norm(x)                      # shape: (batch, seq_len, hidden_dim)
        x = self.attn(x, mask=mask, is_causal=self.is_causal)  # shape: (batch, seq_len, hidden_dim)
        x = self.dropout(x)                        # shape: (batch, seq_len, hidden_dim)
        x = residual + x                           # shape: (batch, seq_len, hidden_dim)

        # --- FFN sublayer (pre-norm + residual) ---
        residual = x                               # shape: (batch, seq_len, hidden_dim)
        x = self.ffn_norm(x)                       # shape: (batch, seq_len, hidden_dim)
        x = self.ffn(x)                            # shape: (batch, seq_len, hidden_dim)
        x = self.dropout(x)                        # shape: (batch, seq_len, hidden_dim)
        x = residual + x                           # shape: (batch, seq_len, hidden_dim)

        return x  # shape: (batch, seq_len, hidden_dim)


class BERTModel(nn.Module):
    """
    BERT-style bidirectional Transformer (~110M params).
    - Pre-norm architecture (RMSNorm before each sublayer)
    - RoPE positional encoding (via GQA, no learned positional embeddings)
    - MLM head with weight tying to token embeddings
    - No NSP objective
    Config: vocab_size=16000, ffn_dim=4096 → ~110M params
    """
    def __init__(
        self,
        vocab_size=16000,   # doubled from 8000 → +6M params in embedding
        hidden_dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        ffn_dim=4096,       # increased from 3072 → +~12M params across 12 layers
        max_seq_len=256,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embedding — NO positional embedding (RoPE handles position inside GQA)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        # shape after embed: (batch, seq_len, hidden_dim)

        # Stack of bidirectional TransformerBlocks (is_causal=False)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, n_kv_heads, ffn_dim, dropout=dropout, is_causal=False)
            for _ in range(n_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(hidden_dim)

        # MLM head: projects hidden states back to vocab
        self.mlm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying: share MLM head weights with token embedding
        self.mlm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        # input_ids shape: (batch, seq_len)

        # Embed tokens and scale by sqrt(hidden_dim)
        x = self.token_embedding(input_ids)           # shape: (batch, seq_len, hidden_dim)
        x = x * math.sqrt(self.hidden_dim)            # shape: (batch, seq_len, hidden_dim)

        # Pass through all TransformerBlocks
        for layer in self.layers:
            x = layer(x, mask=attention_mask)         # shape: (batch, seq_len, hidden_dim)

        # Final norm
        x = self.norm(x)                              # shape: (batch, seq_len, hidden_dim)

        # MLM logits
        logits = self.mlm_head(x)                     # shape: (batch, seq_len, vocab_size)

        return logits  # shape: (batch, seq_len, vocab_size)

    def count_parameters(self):
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BERTModel()
    print(f"BERT parameters: {model.count_parameters()/1e6:.1f}M")

    x = torch.randint(0, 16000, (2, 128))
    out = model(x)
    assert out.shape == (2, 128, 16000), f"Expected (2, 128, 16000), got {out.shape}"
    print("BERT Forward OK")
