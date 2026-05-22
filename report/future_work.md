# Future Work

To build upon the current IndicSeq2Seq baseline, the following research directions are proposed:

1. **Transformer Architecture Migration**: Replacing the recurrent LSTM cells with Self-Attention blocks to eliminate the fixed-dimension compression bottleneck, preventing semantic drift in long sentences.
2. **Back-translation for Data Augmentation**: Translating monolingual Marathi text into Hindi to double the size of the parallel training corpus, leading to more robust embeddings for named entities.
3. **Subword Vocabulary Optimization**: Training separate tokenizers instead of a shared vocabulary to evaluate if language-specific subword segmentation improves morphology learning.
4. **Copy Mechanism**: Integrating a pointer-generator network to copy proper nouns directly from the source sentence to the target translation, bypassing vocabulary alignment failures.
