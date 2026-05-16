"""
train_tokenizer.py — Train a SentencePiece BPE tokenizer.

This script trains a shared BPE model for Hindi and Marathi using the
preprocessed training data.

Usage:
    python train_tokenizer.py --input data/processed/train.hi data/processed/train.mr --vocab_size 32000
"""

import sentencepiece as spm
import argparse
import os

def train_tokenizer(input_files, model_prefix, vocab_size):
    print(f"Training SentencePiece BPE...")
    print(f"  Input files: {input_files}")
    print(f"  Vocab size : {vocab_size}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    
    # Comma-separated string of input files for SentencePiece
    input_str = ",".join(input_files)
    
    # Training parameters
    spm.SentencePieceTrainer.train(
        input=input_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,  # Good for languages with many characters like Hi/Mr
        num_threads=8,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]'
    )
    
    print(f"\nDone! Tokenizer saved:")
    print(f"  Model: {model_prefix}.model")
    print(f"  Vocab: {model_prefix}.vocab")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      nargs='+', required=True, help="List of training text files")
    parser.add_argument("--prefix",     default="tokenizer/spm_hi_mr", help="Output model prefix")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Total vocabulary size")
    args = parser.parse_args()
    
    train_tokenizer(args.input, args.prefix, args.vocab_size)
