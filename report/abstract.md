# Abstract

This report presents our implementation and technical evaluation of an **IndicSeq2Seq** Neural Machine Translation (NMT) system optimized for translating **Hindi to Marathi**. Built on a bidirectional Recurrent Neural Network (RNN) framework with Bahdanau attention, the model is trained on a joint SentencePiece BPE tokenizer of 32k tokens. While the primary training was completed across 15 epochs, we performed a controlled **continuation training phase** ending at Epoch 16, resulting in a final validation loss of **6.487** and a significant BLEU score boost to **7.97** (CHRF: **32.41**).

The core technical contribution of this work lies in the optimization of the inference decoding pipeline. We address the classic NMT failure mode of repetitive token collapse under greedy argmax decoding by implementing a stabilized decoding system combining:
1. **Beam Search decoding** with configurable width.
2. **Logit temperature scaling** to smooth probability distributions.
3. **No-repeat bigram blocking** and general token repetition penalties.
4. **Length normalization** to avoid penalizing longer generated translations.

Our findings reveal that while validation loss plateaus early due to token-level cross-entropy limitations, sequence-level metric evaluation continues to improve through stabilized decoding, establishing a robust foundation for Indic translation tasks under constrained compute environments.
