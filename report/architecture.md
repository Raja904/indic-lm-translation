# Architecture

The NMT system is modeled as a sequence-to-sequence network with Bahdanau (Additive) Attention.

## 1. Encoder
The Encoder is a **2-layer Bidirectional Long Short-Term Memory (LSTM)** network:
* Input: Subword token embeddings of size $d_{emb} = 256$.
* Hidden size: $d_{hid} = 512$ per direction.
* Output: Concatenated forward and backward hidden states, yielding a sequence representation of shape:
  $$\mathbf{h}_i = [\overrightarrow{\mathbf{h}}_i; \overleftarrow{\mathbf{h}}_i] \in \mathbb{R}^{1024}$$

## 2. Attention Mechanism
We implement Bahdanau attention to align decoder query states with encoder keys:
* Given decoder state $\mathbf{s}_{t-1} \in \mathbb{R}^{512}$ and encoder outputs $\mathbf{h}_i \in \mathbb{R}^{1024}$:
  $$e_{t,i} = \mathbf{v}_a^{	op} 	anh(\mathbf{W}_a \mathbf{h}_i + \mathbf{U}_a \mathbf{s}_{t-1})$$
  $$lpha_{t,i} = rac{\exp(e_{t,i})}{\sum_{j=1}^{T_x} \exp(e_{t,j})}$$
  $$\mathbf{c}_t = \sum_{i=1}^{T_x} lpha_{t,i} \mathbf{h}_i$$
where $\mathbf{v}_a$, $\mathbf{W}_a$, and $\mathbf{U}_a$ are learned linear projections.

## 3. Decoder
The Decoder is a **2-layer unidirectional LSTM** network:
* The input to the decoder LSTM at step $t$ is the concatenation of the target word embedding $\mathbf{y}_{t-1}$ and the context vector $\mathbf{c}_t$:
  $$\mathbf{x}_t = [\mathbf{y}_{t-1}; \mathbf{c}_t]$$
* Output linear projection translates LSTM hidden states back to target vocabulary logits.
