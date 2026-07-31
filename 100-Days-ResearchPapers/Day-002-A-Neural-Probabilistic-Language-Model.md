# Day 2 — A Neural Probabilistic Language Model

## Paper Information

**Title:** A Neural Probabilistic Language Model  
**Authors:** Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin  
**Published:** 2003, *Journal of Machine Learning Research*  
**Official paper:** https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

---

## Why This Paper Matters

This paper is an important bridge between traditional statistical language models and modern neural language models.

Before this approach, language models mainly used n-gram probability tables. Those models could remember short combinations of words, but they had difficulty generalizing to unseen sequences and could not naturally represent similarities such as:

- `cat` and `dog`
- `walking` and `running`
- `room` and `bedroom`

The paper proposed learning a continuous vector for every word and training those vectors jointly with a neural network that predicts the next word.

Many ideas used in modern language models are already present here:

- learned word embeddings
- next-word prediction
- softmax over a vocabulary
- maximum-likelihood training
- backpropagation through embeddings
- perplexity as an evaluation metric
- shared parameters across word positions

---

# 1. The Main Problem: Curse of Dimensionality

Suppose a vocabulary contains 100,000 words.

The number of possible 10-word sequences is:

$$
100{,}000^{10}=10^{50}
$$

A training corpus can contain only a tiny fraction of all possible sequences.

A model therefore cannot learn language by storing a separate probability for every possible sentence. This explosive growth in possible combinations is the **curse of dimensionality**.

The goal is to create a model that can learn from observed sentences and generalize to related sentences that were never seen during training.

For example, learning from:

> The cat is walking in the bedroom.

should also help the model understand sentences such as:

> A dog was running in a room.

---

# 2. Language Modeling

A language model assigns a probability to a sequence of words:

$$
P(w_1, w_2, \dots, w_T)
$$

Using the chain rule:

$$
P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P\!\left(w_t \mid w_1, \dots, w_{t-1}\right)
$$

This means that the probability of a sentence is built from a series of next-word predictions.

For example:

$$
P(\text{The cat is sleeping})
$$

is decomposed into:

$$
P(\text{The})
P(\text{cat}\mid\text{The})
P(\text{is}\mid\text{The cat})
P(\text{sleeping}\mid\text{The cat is})
$$

---

# 3. Why N-Gram Models Were Limited

An n-gram model predicts a word using only the previous $n-1$ words:

$$
P(w_t \mid w_1, \dots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \dots, w_{t-1})
$$

A trigram model uses only two previous words.

N-gram models have two major limitations.

## 3.1 Short context

Longer contexts create too many possible combinations. Most traditional systems therefore used only a few previous words.

## 3.2 No natural semantic similarity

A basic n-gram table treats `cat` and `dog` as unrelated symbols. Learning a sequence containing `cat` does not naturally improve the probability of a similar sequence containing `dog`.

## Back-off and smoothing

When an exact trigram is unseen, a model may use a shorter context:

$$
P(w_t \mid w_{t-2}, w_{t-1}) \rightarrow P(w_t \mid w_{t-1}) \rightarrow P(w_t)
$$

This prevents unseen sequences from receiving zero probability, but it does not create semantic understanding.

Without smoothing, an unseen sequence may receive:

$$
P=0
$$

and therefore:

$$
\log(0)=-\infty
$$

---

# 4. The Paper’s Core Solution

The proposed model has three central ideas:

1. Give every word a learned continuous vector.
2. Use the vectors of previous words as input to a neural network.
3. Learn the word vectors and the prediction network together.

The complete flow is:

```text
Previous word indices
        ↓
Embedding lookup in matrix C
        ↓
Concatenate context embeddings into x
        ↓
Hidden neural layer with tanh
        ↓
One score for every vocabulary word
        ↓
Softmax probability distribution
        ↓
Next-word prediction
```

---

# 5. Word Indices and the Embedding Matrix

Every word is assigned an integer index.

Example:

```text
The       → 12
cat       → 47
is        → 8
sleeping  → 106
```

The index is only an identifier. It does not contain meaning.

The model stores learned word vectors in a matrix:

$$
C\in\mathbb{R}^{|V|\times m}
$$

where:

- $|V|$ is the vocabulary size
- $m$ is the embedding dimension
- row $i$ contains the vector for vocabulary word $i$

The embedding of word $i$ is:

$$
C(i)\in\mathbb{R}^{m}
$$

## Important clarification: no nearest-row search

The model does not search for a nearby or semantically similar row.

If `cat` has index 47, the model directly retrieves:

$$
C(47)
$$

Similarity appears later because training moves words used in similar contexts toward useful nearby regions of the vector space.

---

# 6. Shared Parameters Across Context Positions

The same embedding matrix $C$ is used at every context position.

If `cat` appears as $w_{t-1}$ in one example and $w_{t-3}$ in another, the model uses the same vector:

$$
C(\text{cat})
$$

This parameter sharing:

- reduces the number of parameters
- creates a consistent representation for each word
- allows learning from one position to help all other positions
- improves generalization

---

# 7. Understanding the Time Indices

Suppose the sentence is:

```text
The cat is sleeping
```

and the model is predicting `sleeping`.

Then:

$$
w_t=\text{sleeping}
$$

$$
w_{t-1}=\text{is}
$$

$$
w_{t-2}=\text{cat}
$$

$$
w_{t-3}=\text{The}
$$

The notation describes positions relative to the target word. It does not mean the model uses invalid negative array indices.

---

# 8. Constructing the Context Vector $x$

The embeddings of the previous words are concatenated:

$$
x=
\left(
C(w_{t-1}),
C(w_{t-2}),
\ldots,
C(w_{t-n+1})
\right)
$$

Suppose there are three context words and every embedding has four dimensions:

$$
C(\text{is})=[0.1,0.2,-0.4,0.7]
$$

$$
C(\text{cat})=[0.8,-0.1,0.5,0.2]
$$

$$
C(\text{The})=[-0.2,0.3,0.1,0.4]
$$

Then:

$$
x=
[
0.1,0.2,-0.4,0.7,
0.8,-0.1,0.5,0.2,
-0.2,0.3,0.1,0.4
]
$$

Its dimension is:

$$
\dim(x)=(n-1)m
$$

## Important clarification

$x$ is created **before** applying the hidden layer or tanh.

It contains only the concatenated context embeddings.

---

# 9. Hidden-Layer Representation

The hidden layer computes:

$$
a=\tanh(d+Hx)
$$

where:

- $x$ is the concatenated context vector
- $H$ is the input-to-hidden weight matrix
- $d$ is the hidden-layer bias
- $a$ is the hidden activation
- $\tanh$ introduces nonlinearity

The difference between $x$ and $a$ is:

- $x$: raw concatenated word embeddings
- $a$: learned nonlinear representation of the complete context

The hidden layer can learn interactions between words rather than treating every embedding dimension independently.

---

# 10. Output Scores

The paper calculates vocabulary scores using:

$$
y=b+Wx+U\tanh(d+Hx)
$$

Since:

$$
a=\tanh(d+Hx)
$$

this can also be written as:

$$
y=b+Wx+Ua
$$

## Meaning of each term

### $b$: output bias

Each vocabulary word receives a baseline score. Frequent words can learn larger baseline biases.

### $Wx$: optional direct connection

This creates a direct linear path from the context embeddings to vocabulary scores.

It helps the model learn simpler relationships quickly.

When direct connections are disabled:

$$
W=0
$$

### $U\tanh(d+Hx)$: nonlinear path

This path processes the context through a nonlinear hidden layer before producing vocabulary scores.

It allows the network to model more complex relationships between context words.

The final vector $y$ contains one unnormalized score, or logit, for every vocabulary word.

---

# 11. Softmax

The model converts logits into probabilities:

$$
P(w_t = i \mid \text{context}) = \frac{e^{y_i}}{\sum_{j} e^{y_j}}
$$

Softmax guarantees:

$$
P_i>0
$$

and:

$$
\sum_i P_i=1
$$

The result is a valid probability distribution over the full vocabulary.

---

# 12. Why the Output Layer Is Expensive

The model must calculate a score for every vocabulary word.

With a vocabulary of nearly 18,000 words, every training example requires almost 18,000 output computations.

The expensive part includes:

- computing every output logit
- exponentiating the logits
- summing across the vocabulary
- normalizing the probabilities
- computing output gradients

In one reported architecture, approximately 99.7% of computation occurred in the output layer.

---

# 13. Numerically Stable Softmax

Large logits can cause numerical overflow.

The paper subtracts the maximum logit before exponentiation:

$$
Q = \max_{j} y_j
$$

$$
P_i = \frac{e^{y_i - Q}}{\sum_{j} e^{y_j - Q}}
$$

Subtracting the same constant from every logit does not change the final softmax probabilities.

This is still standard practice in modern deep-learning libraries.

---

# 14. Training Objective

The model maximizes penalized log-likelihood:

$$
L = \frac{1}{T}\sum_{t} \log P(w_t \mid w_{t-1}, \dots, w_{t-n+1}) + R(\theta)
$$

In plain language:

> Increase the probability assigned to the correct next word while discouraging unnecessarily large parameters.

A conceptual regularization term is:

$$
R(\theta)=-\lambda\|\theta\|_2^2
$$

The paper applies weight decay to neural-network weights and the embedding matrix, but not to biases.

---

# 15. Gradient Updates

The paper uses stochastic gradient ascent:

$$
\theta \leftarrow \theta + \epsilon \nabla_{\theta} \log P(w_t \mid \text{context})
$$

where:

- $\theta$ contains all learned parameters
- $\epsilon$ is the learning rate
- the gradient indicates how parameters should change to increase the correct-word probability

Modern systems often minimize negative log-likelihood with gradient descent:

$$
\mathcal{L} = -\log P(w_t \mid \text{context})
$$

These are mathematically equivalent:

```text
Maximize log-likelihood
        =
Minimize negative log-likelihood
```

The learned parameter set includes:

$$
\theta=(b,d,W,U,H,C)
$$

Therefore, the embedding matrix $C$ is trained jointly with the rest of the network.

---

# 16. How Semantic Similarity Emerges

The model is never directly told that `cat` and `dog` are similar.

Suppose the training corpus repeatedly contains:

```text
The cat is sleeping.
The dog is sleeping.
A cat was running.
A dog was running.
```

Because `cat` and `dog` appear in similar contexts and support similar predictions, their embedding rows receive similar gradient updates.

Over time:

$$
C(\text{cat})\approx C(\text{dog})
$$

This similarity is a learned consequence of the prediction objective.

The model learns usage patterns, not dictionary definitions.

---

# 17. Perplexity

The paper evaluates language models with perplexity:

$$
\operatorname{PPL} = \exp\left(-\frac{1}{T}\sum_{t}\log P(w_t \mid \text{context})\right)
$$

Lower perplexity is better.

Perplexity measures how surprised the model is by the actual next words.

A lower value means the model generally assigns higher probability to the words that truly occur.

---

# 18. Parallel Training

The full-vocabulary output layer made training expensive, so the authors explored parallel computation.

## Data parallelism

Different processors worked on different training examples while sharing model parameters.

An early synchronized implementation used locks and was slow. The authors later allowed asynchronous updates, which introduced small amounts of noise but greatly reduced waiting.

## Parameter parallelism

The output vocabulary was divided among processors.

Each processor calculated scores and gradients for a subset of output words. The processors then shared:

- the softmax normalization sum
- gradients for the hidden layer
- gradients for the input embedding representation

This was effective because most computation occurred in the output layer.

---

# 19. Combining the Neural Model with a Trigram

The authors combined the neural model and a trigram model:

$$
P_{\text{mixture}} = \alpha P_{\text{neural}} + (1 - \alpha) P_{\text{trigram}}
$$

This helped because the models made different kinds of errors:

- the trigram was strong on frequent exact local patterns
- the neural model generalized through continuous representations

Their combination often achieved lower perplexity than either model alone.

---

# 20. Experimental Findings

The experiments used:

- the Brown corpus
- the Associated Press News corpus

The main findings were:

- the neural model achieved lower perplexity than strong n-gram baselines
- larger context windows helped the neural model
- hidden units were useful
- combining the neural and trigram models improved performance
- large neural language models were computationally expensive but feasible
- the output vocabulary layer was the main bottleneck

---

# 21. Limitations

## Fixed context window

The model only uses a fixed number of previous words.

## Full-vocabulary output cost

Every prediction requires a score for every vocabulary word.

## One vector per word

A word such as `bank` receives one embedding even though it has multiple meanings.

## Fixed vocabulary

Words outside the vocabulary are difficult to represent.

## Static lookup embeddings

The initial vector for a word is the same in every sentence. Context affects the hidden representation, but the lookup vector itself is not contextual.

---

# 22. Future Work Proposed by the Authors

The paper suggested several directions that later became major research topics:

- recurrent networks for longer contexts
- hierarchical output structures
- evaluating only selected output words
- importance sampling
- linguistic and semantic prior knowledge
- multiple vectors for polysemous words
- energy-based language models
- better methods for unknown words
- faster parallel training

---

# 23. Connection to Modern Language Models

| 2003 neural model | Modern language model |
|---|---|
| Word-level indices | Subword tokens |
| Static embedding lookup | Contextual token representations |
| Fixed context window | Long context windows |
| Concatenated embeddings | Self-attention |
| One tanh hidden layer | Many Transformer layers |
| Full softmax | Highly optimized vocabulary projection |
| Next-word prediction | Next-token prediction |
| Gradient training | Large-scale distributed optimization |

Modern models are much larger and more powerful, but they retain the central principle:

> Learn token representations and the language-prediction function jointly.

---

# 24. Important Clarifications from This Study

1. A word index is only an ID; it is not a semantic representation.
2. The embedding lookup directly selects a row from $C$; it does not search for the nearest vector.
3. Similarity between words emerges through similar training updates.
4. $C$ is shared across every context position.
5. $x$ is the concatenation of context embeddings.
6. $x$ exists before the tanh hidden layer.
7. $a=\tanh(d+Hx)$ is the nonlinear hidden representation.
8. $Wx$ is an optional direct linear path to the output.
9. The output layer, not merely the softmax formula, is the computational bottleneck.
10. Embeddings and neural-network parameters are learned together.
11. Lower perplexity means better next-word prediction.
12. The model replaces sparse probability tables with a smooth learned function.

---

# 25. Final Mental Model

```text
Context words
    ↓
Convert each word to an integer index
    ↓
Use each index to retrieve its row from C
    ↓
Concatenate the retrieved vectors into x
    ↓
Compute a = tanh(d + Hx)
    ↓
Combine the direct and nonlinear paths
    ↓
Compute y = b + Wx + Ua
    ↓
Apply softmax over the vocabulary
    ↓
Obtain the probability of every next word
    ↓
Compare with the correct next word
    ↓
Backpropagate the error
    ↓
Update embeddings and neural-network parameters
```

---

# Final Summary

Traditional n-gram models relied on short discrete contexts and large probability tables. They could not naturally transfer knowledge between semantically or grammatically similar words.

This paper introduced a neural language model that learns a continuous vector for every word. The vectors of previous words are concatenated and transformed by a nonlinear hidden layer. The network then assigns a score to every vocabulary word and uses softmax to produce the next-word probability distribution.

The model learns the embeddings and prediction network together by maximizing the correct next-word log-likelihood.

Its key strength is generalization: one observed sentence can improve predictions for many related unseen sentences because similar words can acquire similar continuous representations.

Its primary weakness is computational cost, especially the need to calculate scores for the entire output vocabulary.

This paper established a foundation that later influenced word2vec, neural machine translation, BERT, GPT, and modern large language models.
