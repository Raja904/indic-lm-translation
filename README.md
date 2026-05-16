# Multilingual NMT and Language Model Pretraining (Hindi ↔ Marathi)

This repository contains a clean, production-style setup for multilingual neural machine translation (NMT) and language model pretraining, specifically focusing on Hindi (Hi) and Marathi (Mr).

## Project Goals
1.  **Classical Seq2Seq NMT**: LSTM-based encoder-decoder with Bahdanau Attention.
2.  **Transformer Pretraining**:
    *   **BERT-style**: Masked Language Modeling (~110M parameters).
    *   **GPT-2 style**: Causal Language Modeling (~124M parameters).
3.  **Translation with Pretrained Models**: Using pretrained BERT and GPT representations for NMT.
4.  **Target Pair**: Hindi ↔ Marathi.

## Directory Structure
- `/data`: Raw, processed, and tokenized datasets.
- `/notebooks`: Exploration and debugging.
- `/tokenizer`: SentencePiece model training and saved models.
- `/checkpoints`: Saved model weights.
- `/logs`: Local and WandB logs.
- `/part1_seq2seq`: LSTM-based translation components.
- `/part2_pretraining`: BERT and GPT-2 pretraining.
- `/shared_modules`: Reusable components (RoPE, GQA, RMSNorm).
- `/translation`: Fine-tuning/Translation on top of pretrained models.
- `/configs`: YAML configuration files.
- `/utils`: Common utility functions (logging, checkpointing, etc.)
- `/scripts`: Preprocessing and tokenizer training scripts.
- `/reports`: Final reports and figures.
- `/assets`: Plots and figures.
- `/tests`: Sanity checks.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up WandB:
   ```bash
   wandb login
   ```

## Usage
- **Preprocessing**: `python scripts/preprocess.py`
- **Tokenizer**: `python scripts/train_tokenizer.py`
- **Part 1 (Seq2Seq)**: `python part1_seq2seq/train.py --config configs/part1/lstm_base.yaml`
- **Part 2 (Pretraining)**: `python part2_pretraining/bert/train.py --config configs/part2/bert_base.yaml`

## Platform Compatibility
This codebase is designed to run seamlessly on:
- Local CPU/GPU
- Google Colab
- Kaggle Kernels
(Managed via YAML configs and `utils/device.py`)
