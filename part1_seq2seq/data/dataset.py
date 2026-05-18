import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import os

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

class TranslationDataset(Dataset):
    def __init__(self, hi_path, mr_path, sp_model_path, max_len=100):
        super().__init__()
        
        # Load SentencePiece model
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        
        # Read hi and mr files line by line
        with open(hi_path, "r", encoding="utf-8") as f_hi, \
             open(mr_path, "r", encoding="utf-8") as f_mr:
            hi_lines = [line.strip() for line in f_hi if line.strip()]
            mr_lines = [line.strip() for line in f_mr if line.strip()]
            
        assert len(hi_lines) == len(mr_lines), "Mismatched number of lines between source and target files."
        
        self.pairs = []
        for hi_line, mr_line in zip(hi_lines, mr_lines):
            # Encode each line
            hi_ids = self.sp.encode(hi_line)
            mr_ids = self.sp.encode(mr_line)
            
            # Add <sos>=2 at start, <eos>=3 at end
            src_ids = [SOS_ID] + hi_ids + [EOS_ID]
            tgt_ids = [SOS_ID] + mr_ids + [EOS_ID]
            
            # Filter out pairs where either side exceeds max_len AFTER encoding
            if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
                self.pairs.append((
                    torch.LongTensor(src_ids),
                    torch.LongTensor(tgt_ids)
                ))
                
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    """
    batch: list of tuples (src_ids, tgt_ids)
    """
    src_list = [item[0] for item in batch]
    tgt_list = [item[1] for item in batch]
    
    # src_len: [batch_size]
    src_len = torch.LongTensor([len(s) for s in src_list])
    
    # tgt_len (original lengths): [batch_size]
    tgt_len = torch.LongTensor([len(t) for t in tgt_list])
    
    # Pad src to max src length in batch using pad_id=0
    # src: [batch_size, max_src_len]
    src = pad_sequence(src_list, padding_value=PAD_ID, batch_first=True)
    
    # Pad tgt to max tgt length in batch using pad_id=0
    # tgt: [batch_size, max_tgt_len]
    tgt = pad_sequence(tgt_list, padding_value=PAD_ID, batch_first=True)
    
    # tgt_in = tgt without last token (decoder input)
    # tgt_in: [batch_size, max_tgt_len - 1]
    tgt_in = tgt[:, :-1]
    
    # tgt_out = tgt without first token (what decoder must predict)
    # tgt_out: [batch_size, max_tgt_len - 1]
    tgt_out = tgt[:, 1:]
    
    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_len": src_len,
        "tgt_len": tgt_len - 1  # adjusted for shifted targets
    }


if __name__ == "__main__":
    # Note: We use tokenizer/spm_hi_mr.model because that is the name we used earlier.
    hi_path = "data/processed/train.hi"
    mr_path = "data/processed/train.mr"
    sp_path = "tokenizer/spm_hi_mr.model"
    
    if not os.path.exists(hi_path):
        print(f"Warning: Ensure you run this from the project root. Could not find {hi_path}")
    else:
        print("Loading dataset...")
        dataset = TranslationDataset(
            hi_path=hi_path,
            mr_path=mr_path,
            sp_model_path=sp_path,
            max_len=100
        )
        print(f"Dataset size after filtering: {len(dataset):,}")
        
        # Create DataLoader with batch_size=32 and collate_fn
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        
        # Get one batch
        batch = next(iter(dataloader))
        
        print("\nBatch shapes:")
        print(f"  src     : {batch['src'].shape}")
        print(f"  tgt_in  : {batch['tgt_in'].shape}")
        print(f"  tgt_out : {batch['tgt_out'].shape}")
        print(f"  src_len : {batch['src_len'].shape}")
        print(f"  tgt_len : {batch['tgt_len'].shape}")
        
        print("\nDecoded Sample Verification:")
        sp = dataset.sp
        for i in range(2):
            print(f"\n--- Sample {i+1} ---")
            
            # Remove padding and special tokens for clean decoding
            src_tokens = [t for t in batch['src'][i].tolist() if t not in (PAD_ID, SOS_ID, EOS_ID)]
            print(f"SRC (Hindi)         : {sp.decode(src_tokens)}")
            
            tgt_in_tokens = [t for t in batch['tgt_in'][i].tolist() if t not in (PAD_ID, SOS_ID, EOS_ID)]
            print(f"TGT_IN (Marathi)    : {sp.decode(tgt_in_tokens)}")
            
            tgt_out_tokens = [t for t in batch['tgt_out'][i].tolist() if t not in (PAD_ID, SOS_ID, EOS_ID)]
            print(f"TGT_OUT (Predicted) : {sp.decode(tgt_out_tokens)}")
