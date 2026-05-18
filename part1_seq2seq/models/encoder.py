import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, n_layers=2, dropout=0.3, pad_idx=0):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional LSTM -> output hidden dim is 2 * hidden_dim
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers, 
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Linear layers to project bidirectional hidden and cell states (hidden_dim*2 -> hidden_dim)
        # This is needed because the decoder expects unidirectional hidden_dim
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, src, src_len):
        # src: (batch, seq_len)
        # src_len: (batch,)
        
        # 1. Embed and dropout
        # embedded: (batch, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(src))
        
        # 2. Pack padded sequence
        # Move src_len to CPU for pack_padded_sequence (PyTorch requirement)
        packed_embedded = pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # 3. Pass through BiLSTM
        # packed_outputs: packed sequence object
        # hidden: (n_layers * 2, batch, hidden_dim)
        # cell: (n_layers * 2, batch, hidden_dim)
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        
        # 4. Unpack padded sequence
        # encoder_outputs: (batch, seq_len, hidden_dim * 2)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=src.shape[1])
        
        # 5. Project hidden and cell states for each layer
        # Reshape to combine forward and backward states
        # hidden: (n_layers, 2, batch, hidden_dim)
        hidden = hidden.view(self.n_layers, 2, src.shape[0], self.hidden_dim)
        
        # Concatenate forward and backward: (n_layers, batch, hidden_dim * 2)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        
        # Project using Linear + tanh: (n_layers, batch, hidden_dim)
        hidden = torch.tanh(self.fc_hidden(hidden))
        
        # Same process for the cell state
        # cell: (n_layers, 2, batch, hidden_dim)
        cell = cell.view(self.n_layers, 2, src.shape[0], self.hidden_dim)
        
        # Concatenate forward and backward: (n_layers, batch, hidden_dim * 2)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        
        # Project using Linear + tanh: (n_layers, batch, hidden_dim)
        cell = torch.tanh(self.fc_cell(cell))
        
        # Returns:
        # encoder_outputs: (batch, seq_len, hidden_dim * 2)
        # hidden: (n_layers, batch, hidden_dim)
        # cell: (n_layers, batch, hidden_dim)
        return encoder_outputs, hidden, cell
