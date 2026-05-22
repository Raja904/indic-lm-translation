# Conclusion

In this work, we developed and analyzed an Indic Seq2Seq Neural Machine Translation system for Hindi-to-Marathi translation. By transitioning the evaluation and validation setup from greedy decoding to stabilized Beam Search with temperature scaling and repetition constraints, we successfully resolved severe recurrent repetition collapse.

Our evaluation shows that:
* Greedy decoding BLEU is highly constrained due to cyclic degeneration.
* Inference-side decoding stabilization yields a significant boost in translation quality (Final BLEU: **7.97**, CHRF: **32.41**).
* Model performance behavior demonstrates that sequence-level translation fluency continues to improve even after token-level cross-entropy loss plateaus.

This project demonstrates how rigorous post-processing optimization and analytical failure tracing can compensate for hardware constraints, achieving clean engineering quality and research-level interpretability.
