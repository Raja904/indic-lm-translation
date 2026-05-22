# Compute Constraints & Trade-offs

## 1. Hardware Environment
* **Platform**: Kaggle
* **GPU Accelerator**: T4 x2 GPUs (Parallel training enabled)
* **Training Time**: ~3-4 minutes per epoch for the parallel Indic corpus.

## 2. Experimental Trade-offs
* **Budget-Aware Engineering**: Because of the limited GPU hours allocated on Kaggle, training a large-scale Transformer model or running a 100+ epoch LSTM training sweep was not feasible.
* **Inference-Side Stabilization**: Instead of spending compute on retraining the model with new layers or embeddings, we focused on **inference-side optimization**. By implementing Beam Search, repetition blocking, and temperature control, we successfully corrected the model's structural repetition issues at zero training cost. This is a high-ROI engineering trade-off for low-compute environments.
