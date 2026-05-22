import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # weight parameter: nn.Parameter(torch.ones(dim))
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # x shape: (batch, ..., dim)
        
        # compute RMS: sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)  # shape: (batch, ..., 1)
        rms = torch.sqrt(variance + self.eps)       # shape: (batch, ..., 1)
        
        # normalize: x / RMS
        x_norm = x / rms  # shape: (batch, ..., dim)
        
        # scale by weight
        out = x_norm * self.weight  # shape: (batch, ..., dim)
        
        # return normalized tensor, same shape as input
        return out

if __name__ == "__main__":
    # input (2, 10, 512), assert output shape same, print "RMSNorm OK"
    x = torch.randn(2, 10, 512)
    norm = RMSNorm(512)
    out = norm(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("RMSNorm OK")
