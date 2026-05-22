# Qualitative Failure Analysis

A rigorous scientific analysis of translation systems requires documenting failure cases to understand network bottleneck dynamics.

## Core observed failures:

### 1. Exposure Bias and Error Accumulation
Because the model is trained with teacher forcing (feeding target ground truth 50% of the time), it struggles when generating tokens autoregressively during inference. A single subword mistake early in the sentence alters the hidden state, causing subsequent attention weights to drift.

### 2. Semantic Drift
In longer sentences, the encoder's fixed-size hidden dimension bottleneck cannot compress the entire semantic sequence without loss of detail. Consequently, the decoder attention weights become "diffuse" towards the end, resulting in missing verbs or semantic drift.

### 3. Named Entity Mapping Failures
Proper nouns (e.g. *सचिन तेंदुलकर*, *राजीव*) are often mapped incorrectly. For example, *राजीव* (Rajeev) is translated as *मी* (I) or *तो* (He). Because named entities occur rarely in the training corpus, the model's word embeddings have weak representations for these subwords, forcing attention to fall back on high-frequency pronouns.

### 4. Recurrent Decoder Instability
Since LSTM cells process sequences sequentially, gradients and activations decay over long ranges. Unlike self-attention architectures, LSTMs cannot link distant tokens directly, making long-range coordination (such as subject-verb agreement) difficult.
