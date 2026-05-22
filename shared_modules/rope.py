import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        # dim = head_dim (not total hidden dim)
        
        # Precompute cos and sin tables in __init__:
        # theta = 1 / (base ^ (2i/dim)) for i in range(dim//2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        
        # freqs = outer(positions, theta)
        freqs = torch.outer(positions, inv_freq)
        
        # cos_table = cos(freqs), shape (max_seq_len, dim//2)
        cos_table = torch.cos(freqs)  # shape: (max_seq_len, dim//2)
        
        # sin_table = sin(freqs), shape (max_seq_len, dim//2)
        sin_table = torch.sin(freqs)  # shape: (max_seq_len, dim//2)
        
        # register as buffers (not parameters)
        self.register_buffer("cos_table", cos_table)
        self.register_buffer("sin_table", sin_table)

    def apply_rope(self, x, seq_len):
        # x shape: (batch, n_heads, seq_len, head_dim)
        
        d = x.shape[-1] // 2
        # split x into x1 (first half) and x2 (second half) along head_dim
        x1 = x[..., :d]  # shape: (batch, n_heads, seq_len, dim//2)
        x2 = x[..., d:]  # shape: (batch, n_heads, seq_len, dim//2)
        
        # rotated = concat(-x2, x1) along head_dim
        rotated = torch.cat((-x2, x1), dim=-1)  # shape: (batch, n_heads, seq_len, head_dim)
        
        # cos/sin shape: (1, 1, seq_len, dim//2)
        cos = self.cos_table[:seq_len].unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_len, dim//2)
        sin = self.sin_table[:seq_len].unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_len, dim//2)
        
        # Expand cos and sin to match head_dim for element-wise multiplication
        cos = torch.cat([cos, cos], dim=-1)  # shape: (1, 1, seq_len, head_dim)
        sin = torch.cat([sin, sin], dim=-1)  # shape: (1, 1, seq_len, head_dim)
        
        # return x * cos + rotated * sin
        return x * cos + rotated * sin  # shape: (batch, n_heads, seq_len, head_dim)

    def forward(self, q, k):
        # q shape: (batch, n_heads_q, seq_len, head_dim)
        # k shape: (batch, n_heads_k, seq_len, head_dim)
        seq_len = q.shape[2]
        
        # apply_rope to both q and k
        rotated_q = self.apply_rope(q, seq_len)  # shape: (batch, n_heads_q, seq_len, head_dim)
        rotated_k = self.apply_rope(k, seq_len)  # shape: (batch, n_heads_k, seq_len, head_dim)
        
        # return rotated_q, rotated_k
        return rotated_q, rotated_k

if __name__ == "__main__":
    # test: q = k = torch.randn(2, 8, 16, 64)
    q = torch.randn(2, 8, 16, 64)  # batch, heads, seq, head_dim
    k = torch.randn(2, 8, 16, 64)
    
    rope = RotaryEmbedding(64)
    rq, rk = rope(q, k)
    
    assert rq.shape == q.shape, f"Expected {q.shape}, got {rq.shape}"
    print("RoPE OK")
