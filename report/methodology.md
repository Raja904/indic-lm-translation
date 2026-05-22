# Methodology

Our IndicSeq2Seq translation pipeline is structured into three discrete phases: data preparation, model training, and post-processing decoding stabilization.

## 1. Dataset & Preprocessing
* **Corpus**: We utilize a parallel Hindi-Marathi corpus.
* **Cleaning**: Sentences are stripped of noisy web artifacts, excessive punctuation, and empty lines. Source-target length ratio constraints are enforced to remove severely misaligned sentence pairs.
* **Splitting**: The corpus is divided into a training partition and a validation partition used for monitoring over-fitting.
* **Qualitative Test Set**: For final evaluation, a balanced set of 45 sentences covering different linguistic difficulties (Simple, Medium, Long, Named Entities, Rare Words, Morphology, Numerals) was curated independently from the training set.

## 2. Tokenizer Strategy
* **SentencePiece BPE**: To handle the morphologically rich nature of Devanagari-based languages (Hindi and Marathi), we trained a joint **SentencePiece subword tokenizer** with a vocabulary size of **32,000**.
* **Vocabulary Sharing**: The Hindi and Marathi scripts share the Devanagari character set. Training a shared vocabulary tokenizer allows for high subword overlap, reducing out-of-vocabulary (OOV) tokens and enabling the sharing of embedding parameters for cognates.
