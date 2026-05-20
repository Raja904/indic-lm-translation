import argparse
import os
import torch
import sentencepiece as spm

# Import model definitions
from part1_seq2seq.models.encoder import Encoder
from part1_seq2seq.models.decoder import Decoder
from part1_seq2seq.models.seq2seq import Seq2Seq
from part1_seq2seq.inference.beam_search import decode_beam_search

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

def main():
    parser = argparse.ArgumentParser(description="Translate Hindi to Marathi using trained Seq2Seq model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lstm_random_exp_a_best.pt", help="Path to checkpoint file")
    parser.add_argument("--text", type=str, default=None, help="Hindi sentence to translate (optional)")
    parser.add_argument("--sp_model", type=str, default=DEFAULT_CONFIG["sp_model"], help="Path to sentencepiece model")
    
    # Beam Search Arguments
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for decoding")
    parser.add_argument("--alpha", type=float, default=0.6, help="Length penalty alpha factor")
    parser.add_argument("--disable_bigram_blocking", action="store_true", help="Disable no-repeat bigram blocking")
    parser.add_argument("--repetition_penalty", type=float, default=1.5, help="Repetition penalty for already-generated tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for decoding logits")
    parser.add_argument("--note", type=str, default="Beam Search Run", help="Description of this run to write in the output file")
    
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
        
        # Generate translation using beam search
        translation = decode_beam_search(
            model=model,
            src=src,
            src_len=src_len,
            sp_model=sp,
            device=device,
            beam_size=args.beam_size,
            max_decode_len=100,
            length_penalty_alpha=args.alpha,
            no_repeat_bigram=not args.disable_bigram_blocking,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            pad_idx=PAD_IDX,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX
        )
        return translation
        
    # 4. Run Inference
    output_file = "checkpoints/translations.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    config_header = (
        f"\n======================================================================\n"
        f"RUN NOTE:       {args.note}\n"
        f"Checkpoint:     {args.checkpoint}\n"
        f"Configuration:  Beam Size={args.beam_size}, Alpha={args.alpha}, "
        f"Repetition Penalty={args.repetition_penalty}, Temperature={args.temperature}, "
        f"Bigram Blocking={not args.disable_bigram_blocking}\n"
        f"======================================================================\n"
    )
    
    if args.text:
        res = translate(args.text)
        print(f"\nHindi:   {args.text}")
        print(f"Marathi: {res}\n")
        with open(output_file, "a", encoding="utf-8") as f_out:
            f_out.write(config_header)
            f_out.write(f"Hindi:   {args.text}\n")
            f_out.write(f"Marathi: {res}\n")
            f_out.write("----------------------------------------\n")
        print(f"Saved translation to: {output_file}")
    else:
        predefined_queries = [
            "यह एक बहुत अच्छा दिन है।",
            "मेरा नाम राजीव है।",
            "मैं स्कूल जा रहा हूँ।",
            "भारत एक महान देश है।",
            "आप कैसे हैं?"
        ]
        
        # Append config header and run results
        with open(output_file, "a", encoding="utf-8") as f_out:
            f_out.write(config_header)
            print("\n=== Running Predefined Test Queries ===")
            for query in predefined_queries:
                res = translate(query)
                print(f"Hindi:   {query}")
                print(f"Marathi: {res}")
                print("-" * 40)
                
                f_out.write(f"Hindi:   {query}\n")
                f_out.write(f"Marathi: {res}\n")
                f_out.write("----------------------------------------\n")
                
        print(f"\nSaved predefined translations to: {output_file}")
            
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
                
                # Append interactive translations to the file
                with open(output_file, "a", encoding="utf-8") as f_out:
                    f_out.write(f"Hindi:   {hi_text}\n")
                    f_out.write(f"Marathi: {mr_text}\n")
                    f_out.write("----------------------------------------\n")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
