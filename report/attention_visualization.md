# Attention Visualization

To analyze the internal alignment behavior of the IndicSeq2Seq model, we extract the raw attention weight matrices from the Bahdanau attention module and plot them as heatmaps.

## Attention Heatmaps

### 1. Successful Translation (Simple Sentence)
![Successful Translation](figures/successful_translation.png)
* **Analysis**: The heatmap shows a sharp, strong diagonal alignment between Hindi words and Marathi translations (e.g. *भारत* mapping to *भारत*, *महान* to *महान*, *देश* to *देश*). This indicates that the attention weights are well-focused on the correct token mappings.

### 2. Semantic Drift Example (Long Sequence)
![Semantic Drift](figures/semantic_drift.png)
* **Analysis**: As the sequence length increases, the attention map becomes noticeably diffuse. The decoder's attention weights start to smear across multiple source tokens, leading to context loss and semantic drift towards the final tokens of the translation.

### 3. Repetition Failure (Greedy Collapse)
![Repetition Failure](figures/repetition_failure.png)
* **Analysis**: The attention matrix shows a cyclic pattern. The attention weights get stuck attending to the same source word (e.g. *दिन*) repeatedly. The model fails to progress along the source text, causing the decoder to generate repetitive tokens.

### 4. Named Entity Alignment Failure
![Named Entity Failure](figures/named_entity_failure.png)
* **Analysis**: The proper noun *सचिन तेंदुलकर* has weak, scattered attention weights. Rather than focusing sharply on the entity, the attention weights split between multiple unrelated words, leading to translation failure.

### 5. Long Sentence Alignment Dispersion
![Long Sentence Alignment](figures/long_sentence_alignment.png)
* **Analysis**: Shows high dispersion of attention. The diagonal pattern is weak, indicating that the RNN hidden states struggle to distinguish between different clauses in a long sentence.
