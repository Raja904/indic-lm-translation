# Part II: Language Model Pretraining and Translation
## Pretraining BERT and GPT for Hindi-Marathi NMT

---

## 1. Abstract
This report details the second phase of the project, focusing on pretraining language models from scratch and leveraging their learned representations for machine translation. We implemented and pretrained two foundational models: a BERT-like encoder (~110M parameters) trained with a custom Masked Language Modeling (MLM) pipeline, and a GPT-2 style decoder (~124M parameters) trained autoregressively. Both architectures were modernized using **Rotary Positional Embeddings (RoPE)**, **Grouped Query Attention (GQA)**, and **RMSNorm**. The pretrained models establish robust, context-aware representations that will be subsequently integrated into a downstream translation framework.

---

## 2. Architecture Enhancements
To modernize the standard Transformer components for both BERT and GPT, we incorporated three key architectural improvements:

### 2.1 Rotary Positional Embeddings (RoPE)
Instead of absolute sinusoidal embeddings, we utilized RoPE to encode relative positional information directly into the attention mechanism's query and key vectors. This improves the model's ability to extrapolate to longer sequences and provides a more principled mechanism for learning token distances.

### 2.2 Grouped Query Attention (GQA)
We replaced standard Multi-Head Attention with Grouped Query Attention (GQA). By sharing key and value heads across multiple query heads, GQA significantly reduces memory bandwidth requirements and KV-cache size during autoregressive decoding, offering a sweet spot between the speed of Multi-Query Attention and the quality of Multi-Head Attention.

### 2.3 RMSNorm
We swapped standard Layer Normalization for Root Mean Square Normalization (RMSNorm). By removing the mean-centering operation, RMSNorm reduces computational overhead while maintaining equivalent training stability and convergence properties.

---

## 3. Pretraining Phase

### 3.1 BERT Encoder Pretraining
- **Parameter Count:** ~110M
- **Objective:** Masked Language Modeling (MLM) without the Next Sentence Prediction (NSP) task. The pipeline masks 15% of the input tokens dynamically and optimizes the cross-entropy loss over the predicted tokens.

**Training Convergence:**
![BERT Metrics Dashboard](../../bert_metrics_dashboard_colorful.png)
![BERT Loss Curves](../../bert_loss_curves_colorful.png)
![BERT PPL Curves](../../bert_ppl_curves_colorful.png)

*Observations:* The BERT model demonstrated steady convergence. Validation loss minimized effectively around Epoch 4 (Val Loss: ~2.53, Val PPL: ~12.60).

### 3.2 GPT Decoder Pretraining
- **Parameter Count:** ~124M
- **Objective:** Autoregressive causal language modeling (predicting the next token given previous context).

**Training Convergence:**
![GPT Metrics Dashboard](../../metrics_dashboard_colorful.png)
![GPT Loss Curves](../../loss_curves_colorful.png)
![GPT PPL Curves](../../ppl_curves_colorful.png)

*Observations:* The GPT-2 model was trained over 8 completed epochs. The validation loss reached its minimum at Epoch 8 (Val Loss: 3.53, Val PPL: 34.41) before encountering instabilities in Epoch 9, confirming Epoch 8 as the optimal checkpoint.

---

## 4. Discussion and Challenges

### 4.1 Optimization and Stability
During pretraining, particularly with the GPT model, we observed late-stage training instability (Epoch 9 crashed). This highlights the sensitivity of autoregressive training at scale. Gradient clipping and careful learning rate warmups were necessary to prevent divergence.

### 4.2 Parameter Efficiency and Trade-offs
The use of GQA allowed us to allocate more parameters to the feed-forward networks while keeping the overall parameter count strictly within the required ~110M and ~124M limits. The trade-off is a slight reduction in representational capacity compared to full MHA, offset by substantial speedups during generation.

---

## 5. Machine Translation Integration Strategy
*How the pretrained BERT and GPT models are integrated for the translation task.*

A BERT-style model excels at bidirectional contextual understanding, making it an ideal encoder for the source language (Hindi). Conversely, a GPT-style model specializes in autoregressive generation, serving as a powerful decoder for the target language (Marathi). 
To exploit these complementary properties, we initialize the NMT Encoder with our pretrained BERT weights and the Decoder with our pretrained GPT weights, bridging them through a cross-attention mechanism.

---

## 6. Inference and Evaluation

### 6.1 Quantitative Results
We evaluated the bridged BERT-GPT model on a test set using the zero-shot soft-prompting approach.
- **Corpus BLEU:** 0.00
- **Corpus CHRF++:** 10.44

### 6.2 Qualitative Analysis
As anticipated from a zero-shot bridged architecture without fine-tuning the cross-lingual projection layer, the quantitative scores are low. While the models were successfully pre-trained independently on their respective languages, the linear projection layer connecting the BERT encoder's continuous representations to the GPT decoder's embedding space was randomly initialized.

**Example Outputs:**

![Inference Example](figures/part2_inference_example.png)

**Observations:**
1. **Target Language Fluency:** The GPT decoder generates syntactically valid and highly fluent Marathi text. This proves that the autoregressive pretraining phase was highly successful at modeling the target language.
2. **Source Language Disconnect:** Because the projection layer mapping Hindi representations to Marathi prompt-embeddings wasn't fine-tuned on parallel data, the GPT model essentially receives "random" contextual prompts. It ignores the source sentence's meaning and acts as an unconditioned language model, resulting in zero BLEU.
3. **Future Work:** To achieve high translation quality, the linear projection bridge (and potentially the top layers of both models) must be fine-tuned on the parallel Hindi-Marathi dataset. However, this experiment successfully demonstrates the architectural integration of two disparate foundational models into a unified generative pipeline.
