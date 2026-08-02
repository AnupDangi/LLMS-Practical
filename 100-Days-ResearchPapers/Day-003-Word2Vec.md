# AIResearch — Day 3 — Efficient Estimation of Word Representations in Vector Space

## Paper Information

**Title:** *Efficient Estimation of Word Representations in Vector Space*  
**Authors:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean  
**Year:** 2013  
**Core topic:** Efficient learning of distributed word representations  
**Main models:** Continuous Bag-of-Words (CBOW) and Continuous Skip-gram

**Paper Link:** https://arxiv.org/pdf/1301.3781
---

# 1. Why This Paper Matters

This paper showed that high-quality word vectors do not require a large nonlinear neural language model.

Earlier feedforward NNLMs and RNNLMs could learn useful representations, but they were expensive because of dense hidden-layer computation, recurrent-state computation, and full-vocabulary output prediction.

The authors proposed simpler **log-linear** architectures that remove the expensive nonlinear hidden layer.

> A simpler model may represent language less precisely per example, but it can train on far more data and learn better reusable word vectors.

This idea became the foundation of **Word2Vec**.

---

# 2. Core Goal

The main goal is not to build the strongest full language model. It is to learn vectors in which related words and relationships become geometrically useful.

Examples:

```text
big : bigger :: small : smaller
France : Paris :: Germany : Berlin
man : woman :: king : queen
```

The model learns these relationships from co-occurrence patterns, not dictionary definitions.

---

# 3. Earlier Feedforward NNLM

Architecture:

```text
N previous words
      ↓
One-hot word IDs
      ↓
Shared embedding/projection matrix
      ↓
Dense nonlinear hidden layer
      ↓
Probability over vocabulary
```

Its complexity is:

$$
Q = N \times D + N \times D \times H + H \times V
$$

where:

- $N$ = number of context words
- $D$ = embedding dimension
- $H$ = hidden-layer size
- $V$ = vocabulary size

| Term | Meaning |
|---|---|
| $N \times D$ | Retrieve and combine context embeddings |
| $N \times D \times H$ | Projection-to-hidden computation |
| $H \times V$ | Hidden-to-vocabulary output computation |

With ordinary softmax, the dominant term is usually:

$$
H \times V
$$

With hierarchical softmax, it becomes roughly:

$$
H \times \log_2(V)
$$

Then the hidden-layer term $N \times D \times H$ becomes the main bottleneck.

---

# 4. Earlier RNNLM

An RNNLM avoids a fixed context length by using a recurrent hidden state.

```text
Current word
     ↓
Hidden state h(t)
     ↓
Vocabulary prediction
     ↑
Previous hidden state h(t-1)
```

A simplified recurrence is:

$$
h_t = f(W_xx_t + W_hh_{t-1} + b)
$$

Its complexity is:

$$
Q = H \times H + H \times V
$$

With hierarchical softmax:

$$
Q \approx H^2 + H\log_2(V)
$$

The dominant term becomes:

$$
H^2
$$

The important conclusion is that even after reducing softmax cost, hidden computation remains expensive.

---

# 5. Parallel Training

The authors used **DistBelief** with:

- Multiple model replicas
- Different mini-batches on different machines
- Asynchronous gradient updates
- A centralized parameter server
- AdaGrad
- Many CPU cores

```text
Large corpus
   ↓
Many replicas process different mini-batches
   ↓
Each replica computes gradients
   ↓
Central parameter server updates parameters
   ↓
Replicas continue asynchronously
```

Important scale:

- **50 to 100 replicas**
- Mini-batch asynchronous gradient descent
- AdaGrad adaptive learning rates

---

# 6. Why the New Models Are Log-Linear

The new models remove the nonlinear hidden layer.

```text
Embedding lookup
      ↓
Simple linear projection
      ↓
Output prediction
```

This makes them less expressive than full neural language models, but much cheaper and easier to train on very large corpora.

The two models are:

1. Continuous Bag-of-Words
2. Continuous Skip-gram

---

# 7. Continuous Bag-of-Words

## Main idea

CBOW predicts the current middle word from surrounding context words.

```text
The cat ___ on the mat
```

Context:

```text
The, cat, on, the, mat
```

Target:

```text
sat
```

Prediction direction:

```text
Many context words → one middle word
```

## Architecture

```text
Context word IDs
      ↓
Shared embedding lookup
      ↓
Average or sum vectors
      ↓
Log-linear classifier
      ↓
Predict middle word
```

The context representation is:

$$
h = \frac{1}{N}\sum_{j=1}^{N}v_{w_j}
$$

where $N$ is the number of context words and $v_{w_j}$ is the embedding of context word $w_j$.

## Why “bag-of-words”

Because vectors are averaged, order is mostly lost.

```text
dog bites man
man bites dog
```

These contain the same words, so their averaged context representation may be similar even though meaning changes.

It is called **continuous** because it uses dense continuous vectors instead of discrete counts.

## Context size in the paper

The paper reports strong performance with:

- 4 history words
- 4 future words

So:

$$
N = 8
$$

## CBOW complexity

$$
Q = N \times D + D \times \log_2(V)
$$

The expensive $N \times D \times H$ hidden-layer term is gone.

---

# 8. Continuous Skip-Gram

## Main idea

Skip-gram reverses CBOW.

```text
CBOW: context → centre
Skip-gram: centre → context
```

Example:

```text
The cat sat on the mat
```

Centre word:

```text
sat
```

Training pairs:

```text
sat → The
sat → cat
sat → on
sat → the
```

## Architecture

```text
Current word ID
      ↓
Embedding lookup
      ↓
Continuous projection
      ↓
Predict one nearby word
      ↓
Repeat for every selected neighbour
```

## Objective

$$
\max
\frac{1}{T}
\sum_{t=1}^{T}
\sum_{\substack{-C \le j \le C \\ j \ne 0}}
\log P(w_{t+j}\mid w_t)
$$

where $C$ is the maximum context distance.

## Random radius

If:

$$
C = 5
$$

then the model samples:

$$
R \in \{1,2,3,4,5\}
$$

and uses $R$ words before and $R$ words after the centre word.

The number of classifications is:

$$
2R
$$

This causes nearby words to be sampled more often than distant words.

The paper later uses:

$$
C = 10
$$

in experiments.

## Skip-gram complexity

$$
Q = C\left(D + D\log_2(V)\right)
$$

Larger windows can improve representation quality, but increase cost because more context words must be predicted.

---

# 9. CBOW vs Skip-Gram

| Property | CBOW | Skip-gram |
|---|---|---|
| Input | Surrounding words | Centre word |
| Output | Middle word | Surrounding words |
| Direction | Many-to-one | One-to-many |
| Context operation | Average or sum | Separate predictions |
| Training speed | Faster | Slower |
| Main strength in paper | Syntax | Semantics |
| Typical rare-word behavior | Weaker | Better |

Mental shortcut:

```text
CBOW: neighborhood → missing word
Skip-gram: current word → neighborhood
```

---

# 10. Hierarchical Softmax

Ordinary softmax evaluates all vocabulary words:

$$
O(V)
$$

Hierarchical softmax stores words as leaves in a binary tree and predicts a word through a sequence of binary decisions:

$$
O(\log_2(V))
$$

For:

$$
V = 1{,}000{,}000
$$

we have approximately:

$$
\log_2(1{,}000{,}000) \approx 20
$$

So the model makes roughly 20 decisions rather than evaluating one million outputs.

## Huffman tree

The paper uses a Huffman tree:

- Frequent words get shorter paths
- Rare words get longer paths

This reduces average prediction cost.

---

# 11. Analogy Evaluation

The paper evaluates whether vector differences encode relationships.

Example:

```text
big : biggest :: small : ?
```

Compute:

$$
X = v_{\text{biggest}} - v_{\text{big}} + v_{\text{small}}
$$

Then find the closest word vector to $X$ using cosine similarity.

Expected answer:

```text
smallest
```

Another example:

$$
v_{\text{Paris}} - v_{\text{France}} + v_{\text{Italy}} \approx v_{\text{Rome}}
$$

---

# 12. Cosine Similarity

$$
\cos(v_a,v_b)
=
\frac{v_a^\top v_b}{\|v_a\|\|v_b\|}
$$

| Value | Meaning |
|---|---|
| Near 1 | Similar direction |
| Near 0 | Weak directional relation |
| Near -1 | Opposite direction |

Input question words are excluded from the final nearest-neighbour search.

---

# 13. Semantic-Syntactic Test Set

The benchmark contained:

- **5 semantic categories**
- **9 syntactic categories**
- **8,869 semantic questions**
- **10,675 syntactic questions**

Total:

$$
8{,}869 + 10{,}675 = 19{,}544
$$

questions.

## Semantic categories

1. Common capital city
2. All capital cities
3. Currency
4. City-in-state
5. Man-woman

## Syntactic categories

1. Adjective to adverb
2. Opposite
3. Comparative
4. Superlative
5. Present participle
6. Nationality adjective
7. Past tense
8. Plural nouns
9. Plural verbs

---

# 14. Strict Accuracy Metric

A question counted as correct only when the nearest predicted word exactly matched the expected word.

Therefore:

- Synonyms counted as mistakes
- Multi-word entities such as `New York` were excluded
- 100% accuracy was considered unlikely
- Morphological information was not explicitly provided

---

# 15. Training Data and Vocabulary

Large-scale setup:

- **6 billion Google News tokens**
- **1 million-word vocabulary**

Controlled experiments used:

- **30,000-word vocabulary**

---

# 16. Effect of Dimension and Data

Tested vector dimensions:

```text
50, 100, 300, 600
```

Training sizes:

```text
24M, 49M, 98M, 196M, 391M, 783M words
```

Important result:

- 50 dimensions + 24M words → **13.4%**
- 600 dimensions + 783M words → **50.4%**

Main lesson:

> More data and larger vectors both help, but each gives diminishing returns when increased alone. Increase both together.

---

# 17. Training Setup

For key experiments:

- Stochastic gradient descent
- Backpropagation
- 3 epochs
- Starting learning rate:

$$
0.025
$$

- Learning rate decreased linearly toward zero

Later results showed that one epoch over more unique data could match or beat three epochs over less data.

---

# 18. Architecture Comparison

Using 640-dimensional vectors:

| Model | Semantic | Syntactic | MSR |
|---|---:|---:|---:|
| RNNLM | 9% | 36% | 35% |
| NNLM | 23% | 53% | 47% |
| CBOW | 24% | 64% | 61% |
| Skip-gram | 55% | 59% | 56% |

Numbers to remember:

```text
CBOW: 24 semantic, 64 syntactic
Skip-gram: 55 semantic, 59 syntactic
```

Interpretation:

- CBOW was strongest on syntax
- Skip-gram was dramatically stronger on semantics

---

# 19. Public Vector Comparison

Using 300-dimensional vectors trained on 783M words:

| Model | Semantic | Syntactic | Total |
|---|---:|---:|---:|
| CBOW | 15.5% | 53.1% | 36.1% |
| Skip-gram | 50.0% | 55.9% | 53.3% |

Numbers to remember:

```text
CBOW total: 36.1%
Skip-gram total: 53.3%
```

---

# 20. One Epoch vs Three Epochs

| Model | Dim | Words | Epochs | Total | Time |
|---|---:|---:|---:|---:|---:|
| CBOW | 300 | 783M | 3 | 36.1% | 1 day |
| Skip-gram | 300 | 783M | 3 | 53.3% | 3 days |
| CBOW | 300 | 1.6B | 1 | 36.1% | 0.6 day |
| Skip-gram | 300 | 1.6B | 1 | 53.8% | 2 days |
| Skip-gram | 600 | 783M | 1 | 55.5% | 2.5 days |

Main lesson:

> One pass over more unique data can be as good as or better than repeating the same smaller corpus.

---

# 21. Large-Scale Distributed Results

Using 6B words:

| Model | Dim | Semantic | Syntactic | Total | Training |
|---|---:|---:|---:|---:|---:|
| NNLM | 100 | 34.2% | 64.5% | 50.8% | 14 days × 180 cores |
| CBOW | 1000 | 57.3% | 68.9% | 63.7% | 2 days × 140 cores |
| Skip-gram | 1000 | 66.1% | 65.1% | 65.6% | 2.5 days × 125 cores |

Numbers to remember:

```text
CBOW total: 63.7%
Skip-gram total: 65.6%
```

Best semantic score:

$$
66.1\%
$$

Best syntactic score:

$$
68.9\%
$$

---

# 22. Microsoft Sentence Completion Challenge

The benchmark contained:

- **1,040 sentences**
- One missing word per sentence
- Five candidate choices

| Model | Accuracy |
|---|---:|
| 4-gram | 39.0% |
| Average LSA similarity | 49.0% |
| Log-bilinear model | 54.8% |
| RNNLM | 55.4% |
| Skip-gram | 48.0% |
| Skip-gram + RNNLM | 58.9% |

Important result:

$$
58.9\%
$$

Development accuracy:

$$
59.2\%
$$

Test accuracy:

$$
58.7\%
$$

Skip-gram alone was not the best sentence model, but it added complementary information to the RNNLM.

---

# 23. Learned Relationships

Examples:

```text
Paris - France + Italy ≈ Rome
big - bigger + small ≈ larger
Miami - Florida + Baltimore ≈ Maryland
Microsoft - Windows + Google ≈ Android
Microsoft - Ballmer + Apple ≈ Jobs
```

The examples in Table 8 would score only about:

$$
60\%
$$

under the strict exact-match metric.

So the results were impressive, but not perfect.

---

# 24. Multiple Examples Improve Accuracy

Using ten examples of the same relationship instead of one improved accuracy by about:

$$
10\%
$$

absolute.

The method was to average several relation vectors before searching for the answer.

---

# 25. Applications

The paper discusses or suggests:

- Machine translation
- Information retrieval
- Question answering
- Sentiment analysis
- Paraphrase detection
- Knowledge-base completion
- Fact verification
- Out-of-list word detection
- Semantic relation discovery

---

# 26. Follow-Up Work

The authors later released:

- Multi-threaded C++ code
- CBOW and Skip-gram implementations
- Training speeds of billions of words per hour
- More than **1.4 million named-entity vectors**
- Training on more than **100 billion words**

Numbers to remember:

```text
1.4 million vectors
100+ billion words
Billions of words per hour
```

---

# 27. Main Limitations

## Static embeddings

One vector per word, regardless of context.

```text
bank → financial institution
bank → riverbank
```

## Weak word order

Basic CBOW averages context vectors and loses much of sequence order.

## Out-of-vocabulary words

A word outside the vocabulary has no vector.

## Corpus bias

Embeddings reproduce patterns and biases from training data.

## Similarity is not full understanding

Geometric closeness reflects statistical usage, not human-like conceptual understanding.

## Strict evaluation

Correct synonyms may be counted as wrong.

---

# 28. Connection to Modern AI

| Word2Vec | Modern LLM |
|---|---|
| Word-level vocabulary | Subword tokens |
| Static embedding | Contextual token representation |
| Local context | Long context |
| Shallow log-linear model | Deep Transformer |
| Dot-product scoring | Dot products in attention and output layers |
| Analogy benchmark | Broad benchmark suites |
| CPU-scale training | Distributed accelerator training |

The lasting principle is:

> Useful representations can emerge from solving a predictive task over massive unlabeled data.

---

# 29. Important Numbers to Remember

## Test set

```text
5 semantic categories
9 syntactic categories
8,869 semantic questions
10,675 syntactic questions
19,544 total questions
```

## Training

```text
6B Google News tokens
1M vocabulary
30K reduced vocabulary
Learning rate 0.025
3 epochs
50–100 replicas
```

## Dimensions

```text
50, 100, 300, 600 tested
640 in architecture comparison
1000 in distributed training
```

## Architecture comparison

```text
CBOW: 24% semantic, 64% syntactic
Skip-gram: 55% semantic, 59% syntactic
```

## 300-dimensional comparison

```text
CBOW total: 36.1%
Skip-gram total: 53.3%
```

## Large distributed comparison

```text
CBOW total: 63.7%
Skip-gram total: 65.6%
```

## Sentence completion

```text
Skip-gram: 48.0%
RNNLM: 55.4%
Combined: 58.9%
```

## Follow-up scale

```text
1.4M vectors
100B+ training words
Billions of words per hour
```

---

# 30. Formula Sheet

## Feedforward NNLM

$$
Q = ND + NDH + HV
$$

## RNNLM

$$
Q = H^2 + HV
$$

## CBOW

$$
Q = ND + D\log_2(V)
$$

## Skip-gram

$$
Q = C\left(D + D\log_2(V)\right)
$$

## CBOW context average

$$
h = \frac{1}{N}\sum_{j=1}^{N}v_{w_j}
$$

## Skip-gram objective

$$
\max
\frac{1}{T}
\sum_{t=1}^{T}
\sum_{\substack{-C \le j \le C \\ j \ne 0}}
\log P(w_{t+j}\mid w_t)
$$

## Softmax probability

$$
P(w_o\mid w_i)
=
\frac{\exp({v'_{w_o}}^\top v_{w_i})}
{\sum_{w=1}^{V}\exp({v'_w}^\top v_{w_i})}
$$

## Cosine similarity

$$
\cos(v_a,v_b)
=
\frac{v_a^\top v_b}{\|v_a\|\|v_b\|}
$$

## Analogy vector

$$
X = v_B - v_A + v_C
$$

Example:

$$
v_{\text{biggest}} - v_{\text{big}} + v_{\text{small}}
\approx
v_{\text{smallest}}
$$

---

# 31. Final Mental Model

```text
Earlier NNLMs
    ↓
Useful but expensive hidden layers
    ↓
Word2Vec removes nonlinear hidden computation
    ↓
CBOW predicts centre from context
    ↓
Skip-gram predicts context from centre
    ↓
Hierarchical softmax reduces vocabulary cost
    ↓
Train on much more data
    ↓
Learn semantic and syntactic vector structure
```

Remember:

```text
CBOW
context → centre
faster, strong syntax

Skip-gram
centre → context
slower, stronger semantics
```

---

# 32. Final Summary

This paper demonstrated that simple log-linear prediction models can learn high-quality word embeddings more efficiently than larger feedforward and recurrent neural language models.

CBOW predicts the current word from surrounding words. Skip-gram predicts surrounding words from the current word. Both remove the expensive nonlinear hidden layer and use hierarchical softmax to reduce vocabulary prediction cost.

The paper introduced a large analogy benchmark with 5 semantic categories, 9 syntactic categories, and 19,544 total questions. CBOW performed strongly on syntactic relationships, while Skip-gram was much stronger on semantic relationships.

The main scientific conclusion was:

> Model simplicity enabled training on much more data, and scale produced better reusable word representations.

Word2Vec became a foundational step toward modern self-supervised representation learning.
