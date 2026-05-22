import torch
import torch.nn as nn
import math
import sys
import os

# Allow running directly or as a module — reuse TransformerBlock from bert.py
try:
    from part2_pretraining.bert.bert import TransformerBlock
    from shared_modules.rms_norm import RMSNorm
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from part2_pretraining.bert.bert import TransformerBlock
    from shared_modules.rms_norm import RMSNorm


class GPTModel(nn.Module):
    """
    GPT-style causal/autoregressive Transformer (~124M params).
    - Pre-norm architecture (RMSNorm before each sublayer)
    - RoPE positional encoding (via GQA, no learned positional embeddings)
    - Causal (is_causal=True) TransformerBlocks
    - LM head with weight tying to token embeddings
    Config: vocab_size=16000, ffn_dim=4096 → ~124M params
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

        # Stack of causal TransformerBlocks (is_causal=True)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, n_kv_heads, ffn_dim, dropout=dropout, is_causal=True)
            for _ in range(n_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(hidden_dim)

        # LM head: projects hidden states back to vocab
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying: share LM head weights with token embedding
        self.lm_head.weight = self.token_embedding.weight

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

        # Pass through all causal TransformerBlocks (is_causal=True)
        for layer in self.layers:
            x = layer(x, mask=attention_mask)         # shape: (batch, seq_len, hidden_dim)

        # Final norm
        x = self.norm(x)                              # shape: (batch, seq_len, hidden_dim)

        # LM logits
        logits = self.lm_head(x)                      # shape: (batch, seq_len, vocab_size)

        return logits  # shape: (batch, seq_len, vocab_size)

    def count_parameters(self):
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GPTModel()
    print(f"GPT parameters: {model.count_parameters()/1e6:.1f}M")

    x = torch.randint(0, 16000, (2, 128))
    out = model(x)
    assert out.shape == (2, 128, 16000), f"Expected (2, 128, 16000), got {out.shape}"
    print("GPT Forward OK")
