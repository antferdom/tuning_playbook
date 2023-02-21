# Embedding Representations and Semantic Search

**Target:** Learning representations text representations with Deep Learning

![*Turning text into embeddings.*](/Users/antonio/Downloads/5b9a1f6-Embeddings_Visual_1.svg)

**Hypothetical Implementation**

```python
import datacrunch
import numpy as np

client = datacrunch.Client("YOUR_API_KEY")

# get the embeddings
phrases = ["i love soup", "soup is my favorite", "london is far away"]
(soup1, soup2, london) = client.embed(phrases).embeddings

# compare them
def calculate_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

calculate_similarity(soup1, soup2) # 0.9 - very similar!
calculate_similarity(soup1, london) # 0.3 - not similar!
```

## Applications

1. Build systems that route queries using semantic similarity rather than employing metadata.
2. **Data Analysis:** The embeddings can be visualized using projection techniques such as **PCA**, allowing inspection of vast amounts of text and finding helpful patterns inhabiting them. We can construct groups using **clustering.**
3. **Matchmaking:** Semantic search over an embedding database.
4. Entity recognition, sentiment classification and content toxicity filtering.

# Natural Langauge Processing Tasks

*semantic textual similarity (STS)*

Derive semantically meaningful sentence embeddings that can be compared using cosine-similarity or any other algebraic similarity metric. For example, in a [metric space](https://ncatlab.org/nlab/show/metric+space) small distances suggest high relatedness and large distances suggest low similarity.

## How Embeddings are Obtained

Given a text whose lengths surpass an upper-bound, for example 1024 tokens, we split the input text in manageable chunks of valid length. We compute an embedding representation per each chunk and we finish doing a composition operation that generates a single embedding representation of the whole input text independently of its length.

This composition operation can be the average or any other operation with similar semantics.

## Articles

**Reimers** and Gurevych

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

# Contrastive Learning