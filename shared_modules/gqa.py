import torch
import torch.nn as nn
try:
    from .rope import RotaryEmbedding
except ImportError:
    from rope import RotaryEmbedding

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        # n_heads = total query heads (e.g. 8)
        # n_kv_heads = number of key/value heads (e.g. 2)
        # n_heads must be divisible by n_kv_heads
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        # head_dim = hidden_dim // n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout = dropout

        # Projections:
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # RotaryEmbedding(head_dim) instance
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, context=None, mask=None, is_causal=False):
        # x shape: (batch, seq_len, hidden_dim)
        # context shape: (batch, context_len, hidden_dim) or None
        # mask shape: (batch, seq_len) or (batch, 1, 1, context_len) or None
        batch, seq_len, _ = x.shape

        # project q from current input x
        q = self.q_proj(x)  # shape: (batch, seq_len, hidden_dim)

        # project k, v from context (for cross-attention) or x (for self-attention)
        kv_input = context if context is not None else x
        context_len = kv_input.shape[1]

        k = self.k_proj(kv_input)  # shape: (batch, context_len, n_kv_heads * head_dim)
        v = self.v_proj(kv_input)  # shape: (batch, context_len, n_kv_heads * head_dim)

        # reshape q: (batch, seq_len, n_heads, head_dim) → transpose to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # shape: (batch, n_heads, seq_len, head_dim)

        # reshape k, v: (batch, context_len, n_kv_heads, head_dim) → transpose to (batch, n_kv_heads, context_len, head_dim)
        k = k.view(batch, context_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # shape: (batch, n_kv_heads, context_len, head_dim)
        v = v.view(batch, context_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # shape: (batch, n_kv_heads, context_len, head_dim)

        # apply RoPE only for self-attention (when context is None)
        if context is None:
            q, k = self.rope(q, k)  
            # q shape: (batch, n_heads, seq_len, head_dim)
            # k shape: (batch, n_kv_heads, seq_len, head_dim)

        # expand k and v to match n_heads:
        # repeat each kv head (n_heads // n_kv_heads) times
        num_repeats = self.n_heads // self.n_kv_heads
        k = torch.repeat_interleave(k, num_repeats, dim=1)  # shape: (batch, n_heads, context_len, head_dim)
        v = torch.repeat_interleave(v, num_repeats, dim=1)  # shape: (batch, n_heads, context_len, head_dim)

        # convert mask to boolean to ensure correct behavior in PyTorch's scaled_dot_product_attention
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.ndim == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)

        # Handle causal masking combined with padding mask to prevent PyTorch SDPA constraints
        if is_causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device).tril()
            if mask is not None:
                mask = mask & causal_mask
            else:
                mask = causal_mask
            is_causal = False

        # scaled dot product attention:
        # use torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
        dropout_p = self.dropout if self.training else 0.0
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal
        )  # shape: (batch, n_heads, seq_len, head_dim)

        # reshape output → (batch, seq_len, hidden_dim)
        attn_out = attn_out.transpose(1, 2).contiguous()  # shape: (batch, seq_len, n_heads, head_dim)
        attn_out = attn_out.view(batch, seq_len, self.hidden_dim)  # shape: (batch, seq_len, hidden_dim)

        # out_proj
        output = self.out_proj(attn_out)  # shape: (batch, seq_len, hidden_dim)

        # return output
        return output

if __name__ == "__main__":
    # test:
    # x = torch.randn(2, 16, 512)
    # gqa = GroupedQueryAttention(hidden_dim=512, n_heads=8, n_kv_heads=2)
    # out = gqa(x)
    # assert out.shape == (2, 16, 512)
    # print "GQA OK"
    x = torch.randn(2, 16, 512)
    gqa = GroupedQueryAttention(hidden_dim=512, n_heads=8, n_kv_heads=2)
    out = gqa(x)
    assert out.shape == (2, 16, 512), f"Expected (2, 16, 512), got {out.shape}"
    print("GQA OK")
