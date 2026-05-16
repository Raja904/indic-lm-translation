import sentencepiece as spm
import argparse
import os

def train_tokenizer(input_files, model_prefix, vocab_size=32000):
    """
    Train a SentencePiece BPE tokenizer with a shared vocabulary.
    """
    print(f"Training SentencePiece BPE on {input_files}...")
    
    # Comma separated string of input files
    input_str = ",".join(input_files)
    
    spm.SentencePieceTrainer.train(
        input=input_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]'
    )
    
    print(f"Tokenizer saved with prefix: {model_prefix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Path to input text files")
    parser.add_argument("--prefix", type=str, default="tokenizer/spm_hi_mr", help="Model prefix")
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()
    
    train_tokenizer(args.input, args.prefix, args.vocab_size)
