import torch
import torch.nn as nn
from .attention import BahdanauAttention

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, n_layers=2, dropout=0.3, pad_idx=0):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM input size is embedding_dim + hidden_dim*2 (context from encoder)
        input_size = embedding_dim + hidden_dim * 2
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_dim, 
            n_layers, 
            dropout=dropout if n_layers > 1 else 0, 
            batch_first=True
        )
        
        self.attention = BahdanauAttention(hidden_dim=hidden_dim)
        
        # Linear layer for final prediction
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_token, hidden, cell, encoder_outputs, mask):
        # tgt_token shape: (batch,) - single token at each step
        # hidden shape: (n_layers, batch, hidden_dim)
        # cell shape: (n_layers, batch, hidden_dim)
        # encoder_outputs shape: (batch, seq_len, hidden_dim*2)
        # mask shape: (batch, seq_len)
        
        # 1. Embed the target token
        # Add sequence dimension: (batch,) -> (batch, 1)
        tgt_token = tgt_token.unsqueeze(1)
        
        # embedded shape: (batch, 1, embedding_dim)
        embedded = self.dropout(self.embedding(tgt_token))
        
        # 2. Get context vector using attention
        # We use the top layer's hidden state as query: hidden[-1] -> (batch, hidden_dim)
        context, attention_weights = self.attention(hidden[-1], encoder_outputs, mask=mask)
        
        # context shape: (batch, hidden_dim * 2) -> (batch, 1, hidden_dim * 2)
        context = context.unsqueeze(1)
        
        # 3. Concatenate embedded token and context vector
        # lstm_input shape: (batch, 1, embedding_dim + hidden_dim * 2)
        lstm_input = torch.cat((embedded, context), dim=2)
        
        # 4. Pass through LSTM
        # output shape: (batch, 1, hidden_dim)
        # new_hidden shape: (n_layers, batch, hidden_dim)
        # new_cell shape: (n_layers, batch, hidden_dim)
        output, (new_hidden, new_cell) = self.lstm(lstm_input, (hidden, cell))
        
        # 5. Pass output through Linear layer for final prediction
        # logits shape: (batch, 1, vocab_size)
        logits = self.fc_out(output)
        
        # squeeze to (batch, vocab_size)
        logits = logits.squeeze(1)
        
        return logits, new_hidden, new_cell, attention_weights
