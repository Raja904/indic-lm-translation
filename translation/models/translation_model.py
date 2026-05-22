import torch
import torch.nn as nn
import math
import sys
import os

# Allow running directly or as a module using try/except imports
try:
    from part2_pretraining.bert.bert import BERTModel
    from part2_pretraining.gpt.gpt import GPTModel
    from shared_modules.rms_norm import RMSNorm
    from shared_modules.gqa import GroupedQueryAttention
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from part2_pretraining.bert.bert import BERTModel
    from part2_pretraining.gpt.gpt import GPTModel
    from shared_modules.rms_norm import RMSNorm
    from shared_modules.gqa import GroupedQueryAttention


class TranslationModel(nn.Module):
    """
    TranslationModel class for Hindi-Marathi NMT
    using pretrained BERT encoder and GPT decoder with a GQA cross-attention bridge.
    """
    def __init__(
        self,
        bert_checkpoint_path,
        gpt_checkpoint_path,
        vocab_size=8000,
        hidden_dim=768,
        pad_idx=0,
        device='cpu',
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        ffn_dim=4096,
        max_seq_len=256,
        dropout=0.1
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim

        # 1. Load pretrained BERTModel from checkpoint if path is provided and exists
        bert_vocab_size = vocab_size
        if bert_checkpoint_path is not None and os.path.exists(bert_checkpoint_path):
            checkpoint = torch.load(bert_checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', None))
            if state_dict is not None and 'token_embedding.weight' in state_dict:
                # Align vocab size dynamically with pretrained MLM checkpoint (usually vocab_size + 1 due to [MASK] token)
                bert_vocab_size = state_dict['token_embedding.weight'].shape[0]
                print(f"BERT checkpoint detected vocab_size={bert_vocab_size}")

        self.bert = BERTModel(
            vocab_size=bert_vocab_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_idx=pad_idx
        )

        if bert_checkpoint_path is not None and os.path.exists(bert_checkpoint_path):
            checkpoint = torch.load(bert_checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', None))
            if state_dict is not None:
                self.bert.load_state_dict(state_dict)
                print(f"Loaded BERTModel state dict from {bert_checkpoint_path}")
            else:
                print(f"Warning: Could not find model state dict in {bert_checkpoint_path}")

        # Remove MLM head — only keep encoder layers + embedding + final norm
        self.bert.mlm_head = None

        # 2. Load pretrained GPTModel from checkpoint if path is provided and exists
        gpt_vocab_size = vocab_size
        if gpt_checkpoint_path is not None and os.path.exists(gpt_checkpoint_path):
            checkpoint = torch.load(gpt_checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', None))
            if state_dict is not None and 'token_embedding.weight' in state_dict:
                gpt_vocab_size = state_dict['token_embedding.weight'].shape[0]
                print(f"GPT checkpoint detected vocab_size={gpt_vocab_size}")

        self.gpt = GPTModel(
            vocab_size=gpt_vocab_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_idx=pad_idx
        )

        if gpt_checkpoint_path is not None and os.path.exists(gpt_checkpoint_path):
            checkpoint = torch.load(gpt_checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', None))
            if state_dict is not None:
                self.gpt.load_state_dict(state_dict)
                print(f"Loaded GPTModel state dict from {gpt_checkpoint_path}")
            else:
                print(f"Warning: Could not find model state dict in {gpt_checkpoint_path}")

        # 3. Cross-attention bridge:
        # GroupedQueryAttention layer lets the GPT decoder attend to the BERT encoder outputs
        self.cross_attention = GroupedQueryAttention(
            hidden_dim=hidden_dim,
            n_heads=8,
            n_kv_heads=2,
            dropout=dropout
        )
        # RMSNorm(768) applied to decoder representations before cross-attention
        self.cross_attn_norm = RMSNorm(hidden_dim)

    def forward(self, src_ids, tgt_ids):
        # src_ids shape: (batch_size, src_len)
        # tgt_ids shape: (batch_size, tgt_len)

        # Compute attention masks based on pad index
        # src_mask shape: (batch_size, src_len)
        src_mask = (src_ids != self.pad_idx)
        # tgt_mask shape: (batch_size, tgt_len)
        tgt_mask = (tgt_ids != self.pad_idx)

        # 1. Run BERT encoder
        # bert_emb shape: (batch_size, src_len, hidden_dim)
        bert_emb = self.bert.token_embedding(src_ids)
        # bert_emb shape: (batch_size, src_len, hidden_dim)
        bert_emb = bert_emb * math.sqrt(self.bert.hidden_dim)

        # Pass through bidirectional BERT layers
        # bert_outputs shape: (batch_size, src_len, hidden_dim)
        bert_outputs = bert_emb
        for layer in self.bert.layers:
            # bert_outputs shape: (batch_size, src_len, hidden_dim)
            bert_outputs = layer(bert_outputs, mask=src_mask)

        # Final encoder norm
        # bert_outputs shape: (batch_size, src_len, hidden_dim)
        bert_outputs = self.bert.norm(bert_outputs)

        # 2. Run GPT layers (embedding + decoder layers)
        # gpt_emb shape: (batch_size, tgt_len, hidden_dim)
        gpt_emb = self.gpt.token_embedding(tgt_ids)
        # gpt_emb shape: (batch_size, tgt_len, hidden_dim)
        gpt_emb = gpt_emb * math.sqrt(self.gpt.hidden_dim)

        # Pass through causal GPT layers
        # gpt_hidden shape: (batch_size, tgt_len, hidden_dim)
        gpt_hidden = gpt_emb
        for layer in self.gpt.layers:
            # gpt_hidden shape: (batch_size, tgt_len, hidden_dim)
            gpt_hidden = layer(gpt_hidden, mask=tgt_mask)

        # Inject cross attention before the final LM head
        # Apply RMSNorm before cross-attention
        # normed_gpt_hidden shape: (batch_size, tgt_len, hidden_dim)
        normed_gpt_hidden = self.cross_attn_norm(gpt_hidden)

        # Cross attention: GPT decoder attends to BERT encoder outputs.
        # Pass src_mask to prevent decoder from attending to encoder padding tokens.
        # cross_attn_out shape: (batch_size, tgt_len, hidden_dim)
        cross_attn_out = self.cross_attention(
            x=normed_gpt_hidden,
            context=bert_outputs,
            mask=src_mask
        )

        # Residual connection
        # gpt_hidden shape: (batch_size, tgt_len, hidden_dim)
        gpt_hidden = cross_attn_out + gpt_hidden

        # Apply final RMSNorm (on the combined representation)
        # gpt_hidden shape: (batch_size, tgt_len, hidden_dim)
        gpt_hidden = self.gpt.norm(gpt_hidden)

        # 3. Pass through GPT LM head to get prediction logits
        # logits shape: (batch_size, tgt_len, gpt_vocab_size)
        logits = self.gpt.lm_head(gpt_hidden)

        # 4. Return logits
        return logits

    def count_parameters(self):
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Main test (no checkpoints needed, random weights)
    print("Testing TranslationModel with randomly initialized weights...")
    model = TranslationModel(bert_checkpoint_path=None, gpt_checkpoint_path=None, vocab_size=8000)
    
    src = torch.randint(0, 8000, (2, 32))
    tgt = torch.randint(0, 8000, (2, 28))
    
    out = model(src, tgt)
    
    # Assert shape matches (batch, tgt_len, vocab_size)
    assert out.shape == (2, 28, 8000), f"Expected (2, 28, 8000), but got {out.shape}"
    print("Translation Model Forward OK")
    print(f"Total model parameters: {model.count_parameters() / 1e6:.2f}M")
