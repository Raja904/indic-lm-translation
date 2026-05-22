# Decoding Ablation Report

We evaluate the influence of four decoding strategies on the balanced qualitative evaluation set (45 sentences) using the final **Epoch 16** checkpoint.

## Decoding Metrics Comparison Table

| Decoding Strategy | BLEU | CHRF | Observations |
| :--- | :---: | :---: | :--- |
| Greedy | 5.10 | 42.09 | High rate of repetitive token degeneration. Frequently gets stuck in loops like 'दिवस दिवस'. |
| Beam Search | 5.43 | 45.27 | Sharp metric boost. Resolves simple sentence structures but remains prone to cyclic duplication in longer sequences. |
| Beam + Repetition Blocking | 5.69 | 44.63 | Eradicates word and bigram loops. Greatly improves fluency and semantic completeness. |
| Beam + Temp Scaling | 5.58 | 43.13 | Best overall qualitative results. Slightly softer probabilities prevent hallucination on named entities. |

## Key Discussion Points

1. **Greedy Decoding Instability**: In pure greedy decoding, the model is highly sensitive to local probability peaks. Once an incorrect token is selected, the autoregressive input feeds it back into the decoder, triggering self-reinforcing cyclic loops.
2. **Repetition Blocking Impact**: Introducing bigram blocking and token-level penalties mathematically overrides the model's cyclical biases, allowing it to move to subsequent parts of the source sentence.
3. **Temperature Scaling Impact**: Applying a temperature scale ($T=0.8$) sharpens distributions slightly or prevents near-equal probability classes from swapping randomly, leading to consistent, grammatically sound Marathi verbs.
