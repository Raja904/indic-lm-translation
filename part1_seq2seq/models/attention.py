import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim=512):
        super(BahdanauAttention, self).__init__()
        
        # W1 transforms the encoder outputs (which are bidrectional, hence hidden_dim * 2)
        # (batch, seq_len, hidden_dim*2) -> (batch, seq_len, hidden_dim)
        self.W1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        
        # W2 transforms the decoder hidden state
        # (batch, hidden_dim) -> (batch, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # v generates the scalar score from the combined hidden representations
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, 1)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden shape: (batch, hidden_dim) - typically last layer of decoder
        # encoder_outputs shape: (batch, seq_len, hidden_dim * 2)
        
        # Add sequence dimension to decoder_hidden to broadcast addition
        # (batch, hidden_dim) -> (batch, 1, hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        
        # Compute scores: v(tanh(W1(encoder_outputs) + W2(decoder_hidden)))
        # energy shape: (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))
        
        # scores shape: (batch, seq_len, 1)
        scores = self.v(energy)
        
        # Squeeze out the last dimension
        # scores shape: (batch, seq_len)
        scores = scores.squeeze(2)
        
        if mask is not None:
            # Set scores to very small number where mask is True (pad positions)
            scores = scores.masked_fill(mask == True, -1e4)
            
        # Compute attention weights via softmax
        # attention_weights shape: (batch, seq_len)
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute the context vector
        # attention_weights unsqueeze: (batch, 1, seq_len)
        # encoder_outputs: (batch, seq_len, hidden_dim * 2)
        # bmm gives context shape: (batch, 1, hidden_dim * 2)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        # Squeeze sequence dimension
        # context shape: (batch, hidden_dim * 2)
        context = context.squeeze(1)
        
        return context, attention_weights

if __name__ == "__main__":
    batch = 4
    seq_len = 10
    hidden_dim = 512
    
    attention = BahdanauAttention(hidden_dim=hidden_dim)
    
    # Create dummy encoder_outputs (batch, seq_len, hidden_dim*2)
    encoder_outputs = torch.randn(batch, seq_len, hidden_dim * 2)
    
    # Create dummy decoder_hidden (batch, hidden_dim)
    decoder_hidden = torch.randn(batch, hidden_dim)
    
    # Create dummy mask (batch, seq_len)
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[:, :2] = True  # first 2 positions are True (padding)
    
    # Run forward pass
    context, attention_weights = attention(decoder_hidden, encoder_outputs, mask=mask)
    
    print(f"Context shape: {context.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Asserts
    assert context.shape == (batch, hidden_dim * 2)
    assert attention_weights.shape == (batch, seq_len)
    
    print("Attention OK")
