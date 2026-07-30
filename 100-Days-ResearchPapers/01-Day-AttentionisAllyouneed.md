# 100 Days of Research Papers

## Day 1 — Attention Is All You Need

**Paper:** Attention Is All You Need
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin
**Published:** 2017
**Research area:** Natural Language Processing, Sequence Modeling, Machine Translation

### Paper links

* [Read on arXiv](https://arxiv.org/abs/1706.03762)
* [Official Google Research page](https://research.google/pubs/attention-is-all-you-need/)
* [NeurIPS 2017 page](https://neurips.cc/virtual/2017/poster/9372)

---

## Why I Read This Paper

I read this paper to understand the original Transformer architecture instead of only using Transformer-based models as a black box.

Before the Transformer, sequence models mainly used recurrent architectures such as RNNs, LSTMs, and GRUs. These models processed tokens sequentially, which limited training parallelization and made long-distance relationships harder to learn.

The Transformer introduced a different idea:

> A sequence can be modeled using attention mechanisms without relying on recurrence or convolution.

This paper is important because its architecture became the foundation for later models such as BERT, GPT, T5, Vision Transformer, and many modern multimodal and language models.

---

## My Main Understanding

The Transformer contains two major parts:

1. An **encoder**, which creates contextual representations of the input sequence.
2. A **decoder**, which uses the encoder output and previously available target tokens to generate an output sequence.

The base Transformer configuration uses:

| Component                     | Value |
| ----------------------------- | ----: |
| Encoder layers                |     6 |
| Decoder layers                |     6 |
| Attention heads               |     8 |
| Model dimension               |   512 |
| Dimension per head            |    64 |
| Feed-forward hidden dimension |  2048 |

My memory shortcut is:

> **6 encoder layers, 6 decoder layers, 8 heads, 512 model dimensions, and 2048 feed-forward dimensions.**

---

## Input Embeddings and Positional Encoding

A sentence is first tokenized, and every token is converted into an embedding vector.

For the sentence:

```text
The boy is good
```

there are four token representations.

In Transformer Base, each token is represented using a vector of dimension 512. The input shape is therefore:

```text
4 × 512
```

Self-attention does not automatically understand token order. Therefore, positional encoding is added to each token embedding:

```text
Input representation = Token embedding + Positional encoding
```

The token embedding tells the model what the token represents, while positional encoding tells it where that token appears in the sequence.

---

## My Understanding of Query, Key, and Value

Every contextual token representation is projected independently into three different vectors:

```text
Q = XWQ
K = XWK
V = XWV
```

The three vectors are produced in parallel from the same input representation. Query does not generate Key, and Key does not generate Value.

My mental model is:

* **Query:** What information is this token looking for?
* **Key:** What kind of information does this token offer?
* **Value:** What information will this token actually send if it is selected?

A simple analogy is a library:

* The query is what I am searching for.
* The key is the label or index used to find a matching book.
* The value is the actual content of the book.

Queries and keys decide which tokens should communicate. Values contain the information that is transferred.

---

## Scaled Dot-Product Attention

The attention operation is:

```text
Attention(Q, K, V) = softmax(QKᵀ / √dk)V
```

The operation happens in four steps.

### Step 1: Compare queries and keys

```text
QKᵀ
```

This produces similarity scores between every query token and every key token.

For a sequence containing four tokens, one attention head produces a score matrix with shape:

```text
4 × 4
```

Each row represents one token asking how much information it should receive from every token in the sequence.

### Step 2: Scale the scores

The scores are divided by:

```text
√dk
```

In Transformer Base:

```text
dk = 64
√64 = 8
```

This scaling does not directly convert the scores into values between zero and one. Instead, it prevents the dot products from becoming too large as the vector dimension increases.

Without scaling, large scores could make softmax extremely sharp or saturated, creating very small gradients and making optimization harder.

### Step 3: Apply softmax

Softmax converts the scaled scores into normalized attention weights.

Every row sums to one.

A higher attention weight means that the current token will use more information from the corresponding value vector.

### Step 4: Mix the value vectors

The attention weights are multiplied by the value vectors:

```text
Attention weights × V
```

This gives every token a new contextual representation containing information collected from relevant tokens.

---

## Multi-Head Attention

Transformer Base uses eight attention heads.

The 512-dimensional representation is divided into eight heads:

```text
512 / 8 = 64 dimensions per head
```

For a sequence of length `T`, each head receives:

```text
Q, K, V shape = T × 64
```

The attention-score matrix for each head is:

```text
T × T
```

The number 64 is the feature dimension inside one head. It is not the shape of the final attention-score matrix.

All eight heads perform attention in parallel. Their outputs are concatenated:

```text
8 heads × 64 dimensions = 512 dimensions
```

Multiple heads allow the model to learn different types of relationships at the same time. One head might capture grammatical relationships, while another may capture semantic or positional relationships. These roles are learned automatically and are not manually assigned.

---

# Encoder Architecture

The encoder receives the complete source sequence.

Each of the six encoder layers contains two main sublayers:

1. Multi-head self-attention
2. Position-wise feed-forward neural network

Each sublayer is surrounded by:

* A residual connection
* Layer normalization

The overall flow of one encoder layer is:

```text
Input
  ↓
Multi-head self-attention
  ↓
Residual connection + Layer normalization
  ↓
Feed-forward neural network
  ↓
Residual connection + Layer normalization
```

This block is repeated six times.

---

## Encoder Self-Attention

In encoder self-attention, Query, Key, and Value all come from the encoder’s current representation:

```text
Q = Encoder representation
K = Encoder representation
V = Encoder representation
```

Every source token can attend to every other source token.

For example, in:

```text
The animal did not cross the street because it was tired.
```

the representation of `it` can attend strongly to `animal`.

The model does not use a fixed rule saying that `it` always refers to a particular noun. It learns contextual relationships from data.

The surrounding context can even change the relationship:

```text
The trophy did not fit in the suitcase because it was too big.
```

Here, `it` likely refers to the trophy.

```text
The trophy did not fit in the suitcase because it was too small.
```

Here, `it` likely refers to the suitcase.

The query for `it` is compared with the keys of all tokens. The context changes the learned attention scores, which changes the values that are collected.

---

## Residual Connections and Layer Normalization

After an attention or feed-forward sublayer, the original input is added back:

```text
Output = LayerNorm(Input + Sublayer(Input))
```

The addition is called a residual connection.

Residual connections help preserve existing information and make deeper networks easier to optimize.

The original 2017 Transformer uses what is commonly called post-layer normalization because normalization happens after the residual addition.

---

## Position-Wise Feed-Forward Network

After attention, every token is processed independently by the same feed-forward neural network:

```text
FFN(x) = ReLU(xW1 + b1)W2 + b2
```

In Transformer Base, the dimensional transformation is:

```text
512 → 2048 → 512
```

The first linear layer expands the token representation from 512 to 2048 dimensions.

ReLU is then applied:

```text
ReLU(x) = max(0, x)
```

This means:

* Positive values remain unchanged.
* Negative values become zero.

The second linear layer converts the representation from 2048 dimensions back to 512 dimensions.

An important correction to my original understanding is that the feed-forward network does not pass information from one token to the next token.

For a four-token sentence, the FFN is applied separately four times:

```text
FFN(token 1)
FFN(token 2)
FFN(token 3)
FFN(token 4)
```

The same weights are reused for every position, but the tokens are processed independently.

My mental model is:

> Attention is where tokens communicate with each other. The feed-forward network is where each token privately processes the information it has collected.

---

# Decoder Architecture

The decoder generates the target sequence.

Each of the six decoder layers contains three main sublayers:

1. Masked multi-head self-attention
2. Encoder–decoder attention, also called cross-attention
3. Position-wise feed-forward neural network

The flow is:

```text
Target embeddings + positional encoding
  ↓
Masked self-attention
  ↓
Residual connection + Layer normalization
  ↓
Cross-attention
  ↓
Residual connection + Layer normalization
  ↓
Feed-forward neural network
  ↓
Residual connection + Layer normalization
```

This complete block is repeated six times.

---

## Masked Self-Attention

The first decoder sublayer processes the target tokens that are currently available.

During training, the complete correct target sequence is known. However, the decoder must not be allowed to look at future target tokens.

A causal mask is therefore added to the attention scores.

For three target positions, a simplified mask is:

```text
[0   -∞  -∞]
[0    0  -∞]
[0    0   0]
```

This means:

* Position 1 can see only position 1.
* Position 2 can see positions 1 and 2.
* Position 3 can see positions 1, 2, and 3.

Future positions receive negative infinity before softmax. After softmax, their attention weight becomes zero.

The causal mask prevents the model from seeing the answer it is supposed to predict.

---

## Cross-Attention

Cross-attention connects the decoder to the encoder.

The sources of Query, Key, and Value are different:

```text
Q comes from the decoder.
K comes from the encoder output.
V comes from the encoder output.
```

The decoder query asks:

> Based on what I have generated so far, which source information do I need now?

For example, when generating the French word `garçon`, the decoder query may attend strongly to the encoder representation of the English word `boy`.

Suppose the target currently contains three positions and the source contains four positions.

For one head:

```text
Q shape = 3 × 64
K shape = 4 × 64
V shape = 4 × 64
```

The cross-attention score matrix has shape:

```text
3 × 4
```

There are three decoder positions attending to four encoder positions.

---

## Final Linear Layer and Softmax

After passing through all six decoder layers, the final decoder representation is projected into the vocabulary space.

If the vocabulary contains 30,000 possible tokens:

```text
512-dimensional decoder output
            ↓
Linear projection
            ↓
30,000 vocabulary logits
            ↓
Softmax
            ↓
Probability for every possible token
```

The model then predicts or samples the next token.

---

# Training Versus Inference

This was one of the most important concepts I clarified.

## Training

During training, the complete correct target sequence is already available.

Suppose the expected target is:

```text
Le garçon est bon
```

The decoder receives the shifted-right input:

```text
<BOS> Le garçon est
```

It learns to predict:

```text
Le garçon est bon
```

Because all correct previous target tokens are available, the GPU can calculate predictions for all target positions in parallel.

The causal mask controls which information each position can see. It does not force the GPU to calculate each position one at a time.

Therefore:

> Training is parallel across token positions.

The attention heads also run in parallel.

However, the six decoder layers remain sequential in depth:

```text
Layer 1 → Layer 2 → Layer 3 → ... → Layer 6
```

Layer 2 requires the output of Layer 1, so entire layers cannot normally run simultaneously.

## Inference

During inference, future target tokens do not exist yet.

The model begins with:

```text
<BOS>
```

It predicts:

```text
Le
```

The next input becomes:

```text
<BOS> Le
```

It then predicts:

```text
garçon
```

This process continues one generation step at a time.

Therefore:

> Inference is sequential across generated tokens.

The model cannot generate token `T + 1` until token `T` exists.

However, inside each individual generation step, the GPU still parallelizes:

* Attention heads
* Matrix multiplications
* Hidden dimensions
* Vocabulary calculations
* Multiple sequences in a batch

My final distinction is:

```text
Layers are sequential in depth.
Heads run in parallel within each layer.
Training token positions can be processed in parallel.
Autoregressive inference steps are sequential.
```

The next token is not passed from one decoder layer to another.

Instead, all six decoder layers process the currently available target positions. The output of the final decoder layer is used to predict the next token. That predicted token is then added for the next generation step.

---

# What I Initially Misunderstood

While studying the paper, I corrected several parts of my original mental model.

### Q, K, and V

I initially thought Query and Key might produce Value.

The correct understanding is:

```text
Q, K, and V are independently projected from the same input representation.
```

Queries and keys calculate matching scores. Values carry the information that is mixed.

### Attention scaling

I initially thought division by `√dk` was used to force attention scores into the range zero to one.

The correct understanding is:

```text
Scaling controls the magnitude of the logits.
Softmax converts those logits into normalized weights.
```

### Feed-forward processing

I initially imagined the FFN passing information from one token to the next.

The correct understanding is:

```text
The FFN processes every token independently using the same learned weights.
```

### Decoder layers and generated tokens

I initially thought one decoder layer might generate a token and pass it to the next decoder layer.

The correct understanding is:

```text
All decoder layers transform the currently available sequence.
Only the final decoder output is used to predict a new token.
```

The predicted token is used in the next generation time step, not as a newly generated token between decoder layers.

---

# My Final Mental Model

The encoder reads and understands the complete input sequence.

Inside every encoder layer:

1. Self-attention lets source tokens exchange information.
2. The feed-forward network transforms every token independently.
3. Residual connections and layer normalization stabilize the representations.

The decoder generates the target sequence.

Inside every decoder layer:

1. Masked self-attention examines only currently available target tokens.
2. Cross-attention retrieves relevant information from the encoder.
3. The feed-forward network transforms each target-token representation independently.
4. Residual connections and normalization preserve and stabilize information.

After the final decoder layer, a linear projection and softmax produce the next-token probability distribution.

My shortest summary is:

> Encoder self-attention asks, “How are the input tokens related?” Decoder masked attention asks, “What have I generated so far?” Cross-attention asks, “Which source information do I need now?” The feed-forward network asks, “How should I transform the information inside this token?”

---

## What I Learned

After completing this paper, I can explain:

* Why Transformers replaced recurrence with attention
* How embeddings and positional encodings represent an ordered sequence
* How Query, Key, and Value are created
* How scaled dot-product attention works
* Why multi-head attention uses several parallel attention spaces
* How encoder and decoder layers are structured
* Why the decoder requires causal masking
* How cross-attention connects decoder queries to encoder keys and values
* Why feed-forward networks operate independently across positions
* How training differs from autoregressive inference
* Which computations are sequential and which are parallel

---

## Self-Assessment

**Understanding:** 9/10
**Status:** Learned
**Needs spaced repetition:** Yes

Topics I should briefly review later:

* Why scaling by `√dk` improves optimization
* Why the attention-score shape is `T × T` per head
* The difference between feature dimensions and sequence dimensions
* The difference between decoder depth and generation time steps

---

## Final Reflection

This was not my first time reading “Attention Is All You Need,” but this was the first time I built a complete mental model of the architecture.

Previously, I knew the names of components such as Query, Key, Value, multi-head attention, encoder, and decoder. After this study, I understand what information moves through each component, where that information comes from, and which calculations happen in parallel or sequentially.

The most important lesson for me is that attention does not directly generate the next token. Attention first builds contextual representations. The decoder’s final representation is later projected into vocabulary probabilities to predict the next token.

This paper is now the architectural foundation for the remaining papers in my 100-day research journey.
