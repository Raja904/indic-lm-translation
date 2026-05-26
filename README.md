# Indic NMT and Language Modeling (Hindi ↔ Marathi)

This repository contains the source code and configuration for a comprehensive study on Neural Machine Translation (NMT) and Language Model Pretraining for low-resource Indic languages, specifically focusing on Hindi and Marathi. 

The project is split into two primary phases:
1. **Classical Seq2Seq Translation:** Building an LSTM-based encoder-decoder with Bahdanau Attention.
2. **Transformer Pretraining:** Training BERT (Masked Language Modeling) and GPT (Causal Language Modeling) from scratch.

---

## 📁 Key Directory Structure

*   `data/`: Directory for datasets. **(Note: Place your raw parallel corpora here before running preprocessing).**
*   `part1_seq2seq/`: Contains the LSTM model architecture, training, and inference scripts.
    *   `models/`: Encoder, Decoder, and Attention definitions.
    *   `train.py`: Main entry point for training the Seq2Seq model.
    *   `translate.py`: Inference script featuring Beam Search and Repetition Blocking.
*   `part2_pretraining/`: Contains Transformer-based LM implementations.
    *   `bert/`: Scripts for BERT Masked Language Modeling.
    *   `gpt/`: Scripts for GPT Causal Language Modeling.
*   `tokenizer/`: Houses the scripts and saved models for the joint SentencePiece BPE tokenizer.
*   `configs/`: YAML configuration files controlling hyperparameters, paths, and model dimensions.
*   `checkpoints/`: Automatically created directory where model `.pt` files are saved during training.
*   `report/`: Contains the final Technical Report and Experiment Details PDFs.

---

## ⚙️ 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Raja904/indic-lm-translation.git
cd indic-lm-translation
pip install -r requirements.txt
```

*(Optional)* If you want to track your training metrics on Weights & Biases:
```bash
wandb login
```

---

## 📊 2. Dataset and Preprocessing

Before training any models, you must prepare the dataset and tokenizer.

1. **Download Data:** Place your raw parallel Hindi-Marathi dataset inside the `data/` directory.
2. **Preprocess Data:** Run the preprocessing script to clean the data (removes empty lines, filters extreme length ratios, and strips artifacts):
   ```bash
   python scripts/preprocess.py
   ```
3. **Train Tokenizer:** Train the joint SentencePiece BPE tokenizer (vocab size: 32,000) on the combined Hindi-Marathi text:
   ```bash
   python scripts/train_tokenizer.py
   ```
   *This will generate `spm_hi_mr.model` inside the `tokenizer/` directory.*

---

## 🚀 3. Part 1: Seq2Seq Translation

This phase trains a Bidirectional LSTM with Bahdanau Attention to translate from Hindi to Marathi.

### Training the Model
The training script natively supports multi-GPU setups (like Kaggle's dual T4 GPUs) using `DataParallel`.

```bash
python part1_seq2seq/train.py --config configs/part1/lstm_base.yaml
```
*Model weights will be saved automatically in the `checkpoints/` directory after each epoch.*

### Inference & Translation
To translate new sentences using your best trained checkpoint, use the translation script. It implements advanced decoding techniques including **Beam Search** and **Repetition Blocking**.

```bash
python part1_seq2seq/translate.py --checkpoint checkpoints/lstm_random_exp_a_best.pt --beam_size 4 --alpha 0.6
```

To translate a specific sentence directly from the command line:
```bash
python part1_seq2seq/translate.py --checkpoint checkpoints/lstm_random_exp_a_best.pt --text "यह एक बहुत अच्छा दिन है।"
```
*(If no text is provided, the script runs predefined queries and enters an interactive terminal mode).*

### Evaluation
To evaluate the model against the validation set (computes Corpus BLEU and CHRF scores):
```bash
python part1_seq2seq/evaluate.py --checkpoint checkpoints/lstm_random_exp_a_best.pt
```

---

## 🧠 4. Part 2: Transformer Pretraining

This phase focuses on pretraining Transformer models on the Indic corpus from scratch.

### BERT (Masked Language Modeling)
Train a bidirectional encoder using MLM objectives:
```bash
python part2_pretraining/bert/train.py --config configs/part2/bert_base.yaml
```

### GPT (Causal Language Modeling)
Train a unidirectional autoregressive decoder:
```bash
python part2_pretraining/gpt/train.py --config configs/part2/gpt_base.yaml
```

---

## 💻 Hardware Compatibility
This repository relies on PyTorch and is structured to run seamlessly across:
- Local machines (CPU / Single GPU)
- Google Colab
- Kaggle Kernels (Dual T4 GPU setup highly recommended for optimal training speed).
