# Search and ranking

A frozen fixture document. Do not edit: evaluation cases assert on the exact
rank of passages retrieved from this corpus, so changing the text changes the
measurements.

## Ranking algorithms

BM25 scores a passage by term frequency, inverse document frequency and a
length normalisation factor. Rare terms contribute more than common ones.
The saturation parameter k1 controls how quickly repeated terms stop adding
score, and b controls how strongly longer passages are penalised.

## Combining result lists

When two rankers disagree, their scores are usually on incompatible scales.
Fusing on position instead of magnitude avoids the problem entirely. Each
candidate accumulates a contribution from every list it appears in, so a
candidate both rankers liked outranks one that only a single ranker favoured.
