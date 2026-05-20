import argparse
import os
import torch
import sentencepiece as spm

# Import model definitions
from part1_seq2seq.models.encoder import Encoder
from part1_seq2seq.models.decoder import Decoder
from part1_seq2seq.models.seq2seq import Seq2Seq

# Special token IDs matching TranslationDataset
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# Default model configuration matching 'lstm_random_exp_a'
DEFAULT_CONFIG = {
    "vocab_size": 32000,
    "embedding_dim": 256,
    "hidden_dim": 512,
    "n_layers": 2,
    "dropout": 0.3,
    "sp_model": "tokenizer/spm_hi_mr.model",
}

def decode_greedy(model, src, src_len, sp_model, device, max_decode_len=100):
    """
    Greedy decoding to generate translation outputs autoregressively.
    """
    model.eval()
    with torch.no_grad():
        # Create a dummy target tensor filled with PAD_IDX
        dummy_tgt = torch.full((1, max_decode_len), PAD_IDX, dtype=torch.long, device=device)
        dummy_tgt[0, 0] = SOS_IDX
        
        # Autoregressively predict tokens (teacher_forcing_ratio=0.0)
        outputs = model(src, src_len, dummy_tgt, teacher_forcing_ratio=0.0)
        predictions = outputs.argmax(dim=2)
        
        predicted_tokens = []
        for token_id in predictions[0].tolist():
            if token_id == EOS_IDX:
                break
            if token_id not in [SOS_IDX, PAD_IDX]:
                predicted_tokens.append(token_id)
                
        decoded_string = sp_model.decode(predicted_tokens)
    return decoded_string

def main():
    parser = argparse.ArgumentParser(description="Translate Hindi to Marathi using trained Seq2Seq model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lstm_random_exp_a_best.pt", help="Path to checkpoint file")
    parser.add_argument("--text", type=str, default=None, help="Hindi sentence to translate (optional)")
    parser.add_argument("--sp_model", type=str, default=DEFAULT_CONFIG["sp_model"], help="Path to sentencepiece model")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer
    if not os.path.exists(args.sp_model):
        raise FileNotFoundError(f"SentencePiece model not found at: {args.sp_model}")
    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    print(f"Loaded tokenizer from {args.sp_model}")
    
    # 2. Build Model Architecture
    encoder = Encoder(
        vocab_size=DEFAULT_CONFIG["vocab_size"], 
        embedding_dim=DEFAULT_CONFIG["embedding_dim"], 
        hidden_dim=DEFAULT_CONFIG["hidden_dim"], 
        n_layers=DEFAULT_CONFIG["n_layers"], 
        dropout=DEFAULT_CONFIG["dropout"], 
        pad_idx=PAD_IDX
    )
    decoder = Decoder(
        vocab_size=DEFAULT_CONFIG["vocab_size"], 
        embedding_dim=DEFAULT_CONFIG["embedding_dim"], 
        hidden_dim=DEFAULT_CONFIG["hidden_dim"], 
        n_layers=DEFAULT_CONFIG["n_layers"], 
        dropout=DEFAULT_CONFIG["dropout"], 
        pad_idx=PAD_IDX
    )
    model = Seq2Seq(encoder, decoder, pad_idx=PAD_IDX, device=device)
    
    # 3. Load Checkpoint Weights
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
    print(f"Loading checkpoint weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load model weights (support both wrapped and unwrapped state_dicts)
    state_dict = checkpoint["model_state"]
    # If the checkpoint has 'module.' prefixes from DataParallel, strip them
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        clean_state_dict[name] = v
        
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()
    print("Model initialized and ready for inference!")
    
    def translate(sentence):
        sentence = sentence.strip()
        if not sentence:
            return ""
        
        # Encode string using sentencepiece
        hi_ids = sp.encode(sentence)
        # Format with SOS and EOS tokens
        src_ids = [SOS_IDX] + hi_ids + [EOS_IDX]
        
        # Convert to tensors
        src = torch.LongTensor(src_ids).unsqueeze(0).to(device)
        src_len = torch.LongTensor([len(src_ids)]).to(device)
        
        # Generate translation
        translation = decode_greedy(model, src, src_len, sp, device)
        return translation
        
    # 4. Run Inference
    if args.text:
        res = translate(args.text)
        print(f"\nHindi:   {args.text}")
        print(f"Marathi: {res}\n")
    else:
        print("\nEntering interactive mode. Type 'q' or 'quit' to exit.")
        while True:
            try:
                hi_text = input("Hindi:   ")
                if hi_text.lower() in ['q', 'quit']:
                    break
                if not hi_text.strip():
                    continue
                mr_text = translate(hi_text)
                print(f"Marathi: {mr_text}\n")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
