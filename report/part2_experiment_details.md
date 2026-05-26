# Part II: Experiment Details

## 1. Setup and Environment
- **Platform:** Kaggle Notebooks
- **Hardware:** 2x T4 GPUs (Parallel)
- **Framework:** PyTorch with Mixed Precision (AMP)
- **Time Allocated:** ~12 Hours Total

## 2. Hyperparameters
| Parameter | BERT Encoder | GPT Decoder |
| :--- | :--- | :--- |
| **Vocab Size** | 16,001 | 16,000 |
| **Hidden Dim** | 768 | 768 |
| **Layers** | 12 | 12 |
| **Attention Heads** | 12 (4 KV heads for GQA) | 12 (4 KV heads for GQA) |
| **FFN Dim** | 4096 | 4096 |
| **Max Sequence** | 256 | 256 |
| **Batch Size** | 32 | 64 |
| **Learning Rate** | 1e-4 | 1e-4 |
| **Positional Encoding** | RoPE | RoPE |
| **Normalization** | RMSNorm | RMSNorm |

## 3. BERT Pretraining Logs (Masked Language Modeling)
- **Dataset:** `train.hi` and `val.hi`
- **Total Training Time:** ~2.5 Hours
- **Max Epoch Reached:** Epoch 6 (stopped early due to time constraints)
- **Checkpoint Strategy:** Saved mid-epoch every 2000 steps, and comprehensive checkpoint at the end of each epoch.

| Epoch | Train Loss (Final Step) | Validation Loss | Validation PPL |
| :---: | :---: | :---: | :---: |
| 0 | 4.840 | 4.731 | 113.41 |
| 1 | 4.241 | 4.230 | 68.74 |
| 2 | 3.980 | 3.978 | 53.41 |
| 3 | 3.466 | 3.826 | 45.88 |
| 4 | 3.273 | 2.534 | 12.60 |
| 5 | 2.704 | *Incomplete* | *Incomplete* |

## 4. GPT Pretraining Logs (Causal Language Modeling)
- **Dataset:** `train.mr` and `val.mr`
- **Total Training Time:** ~2.5 Hours
- **Max Epoch Reached:** Epoch 9 (crashed due to instability)
- **Checkpoint Utilized:** Epoch 8

| Epoch | Train Loss (Final Step) | Validation Loss | Validation PPL |
| :---: | :---: | :---: | :---: |
| 6 | 2.268 | 3.597 | 36.50 |
| 7 | 2.202 | 3.598 | 36.55 |
| 8 | 2.266 | 3.538 | 34.41 |

## 5. Bridged Translation (Zero-Shot Soft-Prompting)
- **Technique:** Concatenating BERT output embeddings with GPT token embeddings.
- **Trained Parameters during NMT:** None (Randomly initialized projection layer `768 -> 768`).
- **Inference Time:** ~2 hours (including metric evaluation and script compilation).
- **Final Metrics:** Corpus BLEU = 0.00, CHRF++ = 10.44.
