import torch
import torch.nn as nn
import random

from part1_seq2seq.models.encoder import Encoder
from part1_seq2seq.models.decoder import Decoder
from part1_seq2seq.models.attention import BahdanauAttention

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx=0, device='cpu'):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def create_mask(self, src):
        # src shape: (batch, seq_len)
        # mask shape: (batch, seq_len) - True where src == pad_idx
        return src == self.pad_idx

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        # src shape: (batch, src_len)
        # tgt shape: (batch, tgt_len) - this is tgt_in from collate_fn
        # src_len shape: (batch,)
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        
        # Create mask
        # mask shape: (batch, src_len)
        mask = self.create_mask(src)
        
        # Run encoder
        # encoder_outputs: (batch, src_len, hidden_dim * 2)
        # hidden: (n_layers, batch, hidden_dim)
        # cell: (n_layers, batch, hidden_dim)
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        
        # Store predictions
        # outputs shape: (batch, tgt_len - 1, vocab_size)
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)
        
        # First decoder input is tgt[:, 0] (which should be <sos> for all)
        # decoder_input shape: (batch,)
        decoder_input = tgt[:, 0]
        
        # Loop over tgt_len - 1 steps
        for t in range(1, tgt_len):
            # Run decoder for one step
            # logits: (batch, vocab_size)
            # hidden: (n_layers, batch, hidden_dim)
            # cell: (n_layers, batch, hidden_dim)
            logits, hidden, cell, _ = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, mask
            )
            
            # Store logits in outputs tensor
            outputs[:, t - 1, :] = logits
            
            # Teacher forcing
            if random.random() < teacher_forcing_ratio:
                # Use actual next token as next input
                decoder_input = tgt[:, t]
            else:
                # Use argmax of logits as next input
                decoder_input = logits.argmax(1)
                
        return outputs

if __name__ == "__main__":
    vocab_size = 8000
    batch_size = 4
    src_len = 20
    tgt_len = 15
    pad_idx = 0
    
    # Create dummy src (batch, src_len), tgt (batch, tgt_len) - random LongTensors
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
    
    # Add padding to mask checking logic implicitly
    src[:, -2:] = pad_idx 
    
    # src_len tensor
    src_len_tensor = torch.LongTensor([20, 18, 15, 12])
    
    # Build Encoder, Decoder, Seq2Seq
    encoder = Encoder(vocab_size=vocab_size, embedding_dim=256, hidden_dim=512, n_layers=2, dropout=0.3, pad_idx=pad_idx)
    decoder = Decoder(vocab_size=vocab_size, embedding_dim=256, hidden_dim=512, n_layers=2, dropout=0.3, pad_idx=pad_idx)
    seq2seq = Seq2Seq(encoder, decoder, pad_idx=pad_idx, device='cpu')
    
    # Run forward pass
    output = seq2seq(src, src_len_tensor, tgt, teacher_forcing_ratio=0.5)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, tgt_len - 1, vocab_size), "Output shape is incorrect"
    print("Seq2Seq Forward Pass OK")
