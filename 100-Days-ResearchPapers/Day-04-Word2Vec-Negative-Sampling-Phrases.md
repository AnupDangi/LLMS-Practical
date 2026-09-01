# Day 4 — Distributed Representations of Words and Phrases and Their Compositionality

> **AIResearch — 100 Days of Research Papers**  
> **Status:** ✅ Completed  
> **Day:** 4  
> **Authors:** Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean  
> **Year:** 2013  
> **Venue:** NeurIPS 2013  
> **Official Paper:** https://arxiv.org/abs/1310.4546  
> **PDF:** https://arxiv.org/pdf/1310.4546  
> **NeurIPS:** https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html

---

## 1. Why This Paper Matters

Day 3 introduced CBOW and Skip-gram. Day 4 asks how to make Skip-gram much more scalable and expressive.

The paper develops four major ideas:

1. **Negative Sampling**
2. **Frequent-word Subsampling**
3. **Phrase Representations**
4. **Additive Compositionality**

A compact summary is:

$$
\boxed{\text{Skip-gram} + \text{better objective} + \text{better sampling} + \text{phrase modeling}}
$$

---

## 2. Full-Softmax Skip-gram

For center word $w_I$ and context word $w_O$:

$$
P(w_O \mid w_I)
=
\frac{
\exp({v'_{w_O}}^\top v_{w_I})
}{
\sum_{w=1}^{V}
\exp({v'_w}^\top v_{w_I})
}
$$

The expensive part is the denominator:

$$
\sum_{w=1}^{V}\exp({v'_w}^\top v_{w_I})
$$

With vocabulary size $V$, the output-side work scales roughly as:

$$
\boxed{O(V)}
$$

or, including embedding dimension $D$:

$$
\boxed{O(VD)}
$$

Softmax uses exponentials, but the computational scaling with vocabulary size is **linear, not exponential**.

---

## 3. Negative Sampling

Instead of asking:

> Which word among the whole vocabulary is the correct context?

Negative Sampling asks:

> Is this particular center-context pair genuine or artificial?

For a genuine pair:

$$
(\text{cat},\text{milk})\rightarrow1
$$

we sample fake pairs such as:

$$
(\text{cat},\text{parliament})\rightarrow0
$$

$$
(\text{cat},\text{aircraft})\rightarrow0
$$

If $k=5$:

$$
\boxed{1\text{ positive}+5\text{ negatives}}
$$

replace exhaustive comparison against all $V$ words.

Conceptually:

$$
\boxed{O(V)\rightarrow O(k)}
$$

for the output comparison part.

### Objective

$$
\boxed{
\log \sigma({v'_{w_O}}^\top v_{w_I})
+
\sum_{i=1}^{k}
\mathbb{E}_{w_i\sim P_n(w)}
\left[
\log \sigma(-{v'_{w_i}}^\top v_{w_I})
\right]
}
$$

where:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

For real pairs:

$$
{v'_{w_O}}^\top v_{w_I}\uparrow
$$

For negative pairs:

$$
{v'_{w_i}}^\top v_{w_I}\downarrow
$$

Mental model:

$$
\boxed{\text{Pull real pairs together; push sampled noise pairs apart.}}
$$

---

## 4. Two Embedding Matrices

Word2Vec learns:

$$
W_{\text{input}}
$$

and:

$$
W_{\text{output}}
$$

A word therefore has an input vector $v_w$ and an output/context vector $v'_w$.

For:

$$
(\text{cat},\text{milk})
$$

training uses:

$$
v_{\text{cat}}
$$

and:

$$
v'_{\text{milk}}
$$

---

## 5. Negative-Sampling Distribution

Negative words are sampled from:

$$
\boxed{P_n(w)\propto U(w)^{3/4}}
$$

where $U(w)$ is unigram frequency.

Normalized:

$$
P_n(w)=
\frac{
U(w)^{3/4}
}{
\sum_{j} U(w_j)^{3/4}
}
$$

### Why $3/4$?

Suppose:

$$
f_A=10,000,\qquad f_B=100
$$

Raw ratio:

$$
100:1
$$

After the $3/4$ power:

$$
10,000^{3/4}=1000
$$

$$
100^{3/4}\approx31.62
$$

so:

$$
\boxed{100:1\rightarrow31.6:1}
$$

The distribution is flattened: frequent words remain more likely negatives, but dominate less strongly.

A negative sample is **not** a semantic opposite. It is simply a pair treated as artificial/not observed.

---

## 6. Frequent-Word Subsampling

Very frequent words such as `the`, `of`, `a`, `is`, and `in` produce enormous amounts of redundant training signal.

The paper therefore probabilistically discards some occurrences of frequent words.

A common discard form is:

$$
\boxed{
P_{\text{discard}}(w)
=
1-\sqrt{\frac{t}{f(w)}}
}
$$

with a characteristic threshold:

$$
\boxed{t\approx10^{-5}}
$$

The key behavior is:

$$
f(w)\uparrow
\Rightarrow
P_{\text{discard}}(w)\uparrow
$$

The word is **not removed from the vocabulary**; only some occurrences are skipped during training.

Core lesson:

$$
\boxed{\text{token count}\neq\text{information content}}
$$

Less redundant data can make training faster and improve the effective learning signal.

### Subsampling vs. Negative Sampling

| Property | Frequent-word Subsampling | Negative Sampling |
|---|---|---|
| Operates on | Real corpus tokens | Training pairs |
| Happens | Before pair generation | After positive pair generation |
| Removes real observations | Yes, probabilistically | No |
| Creates artificial examples | No | Yes |
| Main purpose | Reduce redundancy | Avoid full softmax |
| Key parameter | $t$ | $k$ |

Pipeline:

$$
\boxed{
\text{Raw corpus}
\rightarrow
\text{Subsampling}
\rightarrow
\text{Skip-gram pairs}
\rightarrow
\text{Negative Sampling}
}
$$

---

## 7. Phrase Representations

Some meanings belong to multiword expressions:

- `New York`
- `Air Canada`
- `Boston Globe`
- `Toronto Maple Leafs`
- `machine learning`

The corpus can be rewritten as:

```text
New_York
Air_Canada
Boston_Globe
```

Then Skip-gram learns:

$$
v_{\text{New\_York}}
$$

directly.

---

## 8. Phrase Detection Score

The paper uses a score of the form:

$$
\boxed{
\operatorname{score}(w_i,w_j)
=
\frac{
\operatorname{count}(w_i\,w_j)-\delta
}{
\operatorname{count}(w_i)\,\operatorname{count}(w_j)
}
}
$$

The question is:

> Do these two words occur together unusually often relative to how frequently they occur individually?

Raw bigram count is insufficient because frequent function-word combinations can have huge counts without being meaningful phrases.

The discount $\delta$ reduces accidental high scores from rare co-occurrences.

---

## 9. Phrase Learning Is Not BPE

Phrase learning:

$$
\boxed{
\text{New}+\text{York}
\rightarrow
\text{New\_York}
}
$$

combines full words into **larger semantic units**.

BPE instead learns reusable subword pieces:

$$
\boxed{
\text{word/string}
\rightarrow
\text{subword units}
}
$$

Conceptually:

```text
unbelievable
→ un + believ + able
```

So:

$$
\boxed{\text{Phrase learning: words}\rightarrow\text{larger units}}
$$

whereas:

$$
\boxed{\text{BPE: strings}\rightarrow\text{smaller reusable units}}
$$

---

## 10. Additive Compositionality

Word2Vec representations exhibit approximately linear semantic relations.

For example:

$$
v(\text{Madrid})
-
v(\text{Spain})
+
v(\text{France})
\approx
v(\text{Paris})
$$

and conceptually:

$$
v(\text{Germany})+v(\text{capital})
\approx
v(\text{Berlin})
$$

Why can this work?

Embeddings encode statistical context. Vector operations can approximately combine contextual constraints.

Therefore:

$$
\boxed{
\text{distributional regularity}
\rightarrow
\text{geometric regularity}
}
$$

But:

$$
\boxed{
\text{vector arithmetic}
\neq
\text{symbolic reasoning}
}
$$

Word2Vec is not executing explicit logical rules.

---

## 11. Phrase Analogy Evaluation and Scale

Important numbers:

- Phrase analogy benchmark:  
  $$
  \boxed{3218}
  $$

- Very large phrase-training setup:  
  $$
  \boxed{\sim33\text{ billion words}}
  $$

- Large embedding dimensions in some experiments:  
  $$
  \boxed{\sim1000}
  $$

- Strong phrase analogy performance in a large setup:  
  $$
  \boxed{\sim72\%}
  $$

- Negative samples for smaller datasets:  
  $$
  \boxed{5\text{--}20}
  $$

- For very large datasets:  
  $$
  \boxed{2\text{--}5}
  $$

---

## 12. Full Softmax vs. Hierarchical Softmax vs. Negative Sampling

| Method | Main Question | Approximate Output Cost |
|---|---|---:|
| Full Softmax | Which vocabulary word? | $O(V)$ |
| Hierarchical Softmax | Which binary-tree path? | $O(\log V)$ |
| Negative Sampling | Is this pair real or fake? | $O(k)$ |

Hierarchical Softmax still defines probabilities over vocabulary words.

Negative Sampling primarily optimizes:

$$
\boxed{\text{representation quality}}
$$

rather than an ordinary normalized vocabulary distribution.

---

## 13. Complete Training Pipeline

```text
                         RAW CORPUS
                             |
                             v
                  FREQUENT-WORD SUBSAMPLING
                             |
                             v
                      PHRASE DETECTION
                             |
                             v
                   TOKENIZED TRAINING TEXT
                             |
                             v
                         SKIP-GRAM
                             |
                      POSITIVE PAIR
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
          genuine context        k negative words
                                sampled from
                                  U(w)^(3/4)
                 |                       |
                 +-----------+-----------+
                             |
                             v
                    NEGATIVE SAMPLING
                             |
                             v
                      EMBEDDING UPDATE
                             |
                             v
                    WORD + PHRASE VECTORS
                             |
                             v
              ANALOGIES / COMPOSITIONALITY
```

---

## 14. Most Important Equations

### Full Skip-gram Softmax

$$
P(w_O \mid w_I)
=
\frac{
\exp({v'_{w_O}}^\top v_{w_I})
}{
\sum_{w=1}^{V}\exp({v'_w}^\top v_{w_I})
}
$$

### Negative Sampling

$$
\boxed{
\log \sigma({v'_{w_O}}^\top v_{w_I})
+
\sum_{i=1}^{k}
\mathbb{E}_{w_i\sim P_n(w)}
\left[
\log \sigma(-{v'_{w_i}}^\top v_{w_I})
\right]
}
$$

### Noise Distribution

$$
\boxed{P_n(w)\propto U(w)^{3/4}}
$$

### Frequent-Word Discard Probability

$$
\boxed{
P_{\text{discard}}(w)
=
1-\sqrt{\frac{t}{f(w)}}
}
$$

### Phrase Score

$$
\boxed{
\operatorname{score}(w_i,w_j)
=
\frac{
\operatorname{count}(w_i\,w_j)-\delta
}{
\operatorname{count}(w_i)\,\operatorname{count}(w_j)
}
}
$$

### Analogy Structure

$$
\boxed{
v(B)-v(A)+v(C)\approx v(D)
}
$$

---

## 15. Numbers to Remember

| Number | Meaning |
|---:|---|
| $\frac{3}{4}$ | Negative-sampling unigram exponent |
| $5\text{--}20$ | Typical negative samples for smaller datasets |
| $2\text{--}5$ | Negative samples that may suffice for very large datasets |
| $10^{-5}$ | Frequent-word subsampling threshold |
| $3218$ | Phrase analogy questions |
| $\sim33B$ | Words in a very large phrase-training setup |
| $\sim1000$ | Embedding dimensions in some large-scale experiments |
| $\sim72\%$ | Strong phrase analogy performance in a large setup |

Mnemonic:

$$
\boxed{
3/4
\rightarrow
5\text{--}20
\rightarrow
10^{-5}
\rightarrow
3218
}
$$

---

## 16. Day 3 vs. Day 4

### Day 3

Introduced:

$$
\boxed{\text{CBOW}}
$$

and:

$$
\boxed{\text{Skip-gram}}
$$

### Day 4

Improves the system through:

$$
\boxed{\text{Negative Sampling}}
$$

$$
\boxed{\text{Frequent-word Subsampling}}
$$

$$
\boxed{\text{Phrase Representations}}
$$

and demonstrates:

$$
\boxed{\text{Compositional Vector Structure}}
$$

Summary:

$$
\boxed{
\text{Day 3 introduces the architectures}
\rightarrow
\text{Day 4 makes them faster and more expressive}
}
$$

---

## 17. Deeper Lessons

### 1. You Do Not Always Need the Full Prediction Problem

A cheaper surrogate objective can be sufficient when the goal is useful representations:

$$
\boxed{O(V)\rightarrow O(k)}
$$

### 2. More Data Is Not Automatically Better Data

$$
\boxed{\text{token count}\neq\text{information content}}
$$

This idea later connects to modern LLM data filtering, deduplication, sampling, data mixtures, and quality weighting.

### 3. Meaning Can Live Above the Word Level

`New York`, `machine learning`, and `Air Canada` may deserve their own vectors.

### 4. Semantic Relations Can Become Geometry

Statistical regularities in language can create approximately linear relations in vector space.

---

## 18. Three-Sentence Summary

The paper makes Skip-gram more scalable by replacing expensive full-softmax training with Negative Sampling, where each observed center-context pair is contrasted with a small number of noise words sampled from a unigram distribution raised to the $3/4$ power. It also subsamples extremely frequent words to remove redundant training signal and learns vectors for statistically meaningful multiword expressions such as `New_York`. The resulting word and phrase embeddings exhibit useful relational geometry, showing how distributional statistics can become structured continuous representations.

---

## 19. Mastery Checklist

- [x] Why full softmax is expensive
- [x] Why its vocabulary scaling is approximately $O(V)$
- [x] Positive Skip-gram pairs
- [x] Negative samples
- [x] Meaning of $k$
- [x] Negative Sampling objective
- [x] Positive vs. negative dot-product behavior
- [x] Two embedding matrices
- [x] Why $U(w)^{3/4}$ is used
- [x] How $3/4$ flattens the frequency distribution
- [x] Frequent-word subsampling
- [x] Why subsampling differs from Negative Sampling
- [x] Why redundant observations can be removed
- [x] Phrase representations
- [x] Phrase scoring
- [x] Role of $\delta$
- [x] Why phrase learning is not BPE
- [x] Additive compositionality
- [x] Why vector arithmetic is not symbolic reasoning
- [x] Full training pipeline

---

## 20. Final Recall Map

$$
\boxed{
\text{Full Softmax}
\rightarrow
\text{Negative Sampling}
}
$$

$$
\boxed{
1\text{ positive}+k\text{ negatives}
}
$$

$$
\boxed{
P_n(w)\propto U(w)^{3/4}
}
$$

$$
\boxed{
\text{Frequent-word redundancy}
\rightarrow
\text{Subsampling}
}
$$

$$
\boxed{
\text{New York}
\rightarrow
\text{New\_York}
}
$$

$$
\boxed{
\text{Distributional statistics}
\rightarrow
\text{Vector geometry}
}
$$

---

## 21. References

1. Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean.  
   **Distributed Representations of Words and Phrases and Their Compositionality.**  
   NeurIPS 2013.  
   https://arxiv.org/abs/1310.4546

2. Official PDF:  
   https://arxiv.org/pdf/1310.4546

3. NeurIPS Proceedings:  
   https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html

---

## 22. AIResearch Progress

```text
Day 1 — Attention Is All You Need
✅ Completed

Day 2 — A Neural Probabilistic Language Model
✅ Completed

Day 3 — Efficient Estimation of Word Representations in Vector Space
✅ Completed

Day 4 — Distributed Representations of Words and Phrases and Their Compositionality
✅ Completed

Day 5 — GloVe: Global Vectors for Word Representation
🟡 Next
```
