# RAG architecture

The knowledge assistant answers questions about this project using the
project's own documentation as its only source.

## Pipeline

```
docs on disk
   │  ingestion.py    discover (allow-list) → load → title → content hash
   ▼
Document
   │  chunking.py     split on markdown headings → pack to a size budget
   ▼
Chunk  (stable id: {doc_id}::{position})
   │
   ├── embeddings.py     hashing vectoriser → VectorIndex (cosine)
   └── keyword_search.py BM25 → KeywordIndex
   │
   │  hybrid.py        Reciprocal Rank Fusion of both rankings
   ▼
FusedChunk
   │  reranking.py     heuristic rerank → top-k
   ▼
RerankedChunk  (the evidence)
   │  answering.py     three gates: evidence score → schema → citations
   ▼
RagAnswer  (grounded answer with verified citations, or an abstention)
```

## Ingestion

The corpus is an **explicit allow-list** of documentation files, not a
directory walk. `BLOCKED_PATTERNS` additionally refuses anything matching
`.env`, `secrets`, `credentials` or `.venv`, so a credential file cannot be
ingested even if someone adds it to the list.

Each document records a `content_hash`, an `ingested_at` timestamp, its source
path and its title. Document ids are derived from the path rather than
randomly generated, so re-ingesting unchanged content produces identical ids
and chunk ids stay comparable across runs. Two files with identical content
are ingested once, with the duplicate reported in `skipped`.

## Chunking strategy

**Split on markdown headings, then pack paragraphs to a 1200-character
budget, with 120 characters of overlap when a section must be split.**

The corpus is technical documentation where a heading names the topic of
everything beneath it. Keeping a section intact means a retrieved chunk is
self-contained, and the heading becomes the citation's section label. Fixed
character windows would routinely cut a table or a list in half.

Each chunk carries `chunk_id`, `doc_id`, `source_path`, `section`,
`position`, `content_hash` and `ingested_at` — everything a citation needs.

## The two retrievers

### Lexical-vector retrieval

The default embedder, `HashingEmbedder`, is a **hashing vectoriser over word
unigrams and bigrams**, L2-normalised and compared with cosine similarity.

Stopwords are removed before embedding. In a count-based vector, function
words like "what", "is" and "for" are frequent enough to dominate the cosine
similarity, so a chunk of ordinary prose outranks the chunk that answers the
question. They are kept for BM25, where IDF already discounts them.

It is **not a trained semantic model**, and the code does not call it one. It
cannot match "how do I stop the model inventing sources?" to a passage about
"citation validation" unless the words overlap. What it does provide is a real
vector-space retriever with the right interface, latency profile and fusion
behaviour, running offline with no API key.

The `Embedder` protocol exists so a trained model can replace it as a
configuration change rather than a rewrite.

### Keyword retrieval (BM25)

Standard BM25 with `k1=1.5`, `b=0.75`. It excels at exact technical terms —
`max_iterations`, `GAME_DEBUG`, `submit_guess` — because a rare term has high
IDF and pins the one chunk containing it. It is blind to paraphrase.

The two retrievers are included together because they **fail differently**.
That is the entire argument for hybrid retrieval.

## Hybrid ranking: Reciprocal Rank Fusion

BM25 scores are unbounded and corpus-dependent; cosine similarity is bounded
in [0, 1]. Adding them compares incomparable units and lets whichever scale is
larger dominate silently.

Reciprocal Rank Fusion discards magnitudes and uses only each retriever's
ordering:

```
RRF(d) = Σ over retrievers  1 / (k + rank_r(d))        k = 60
```

- **Scale-free** — only ranks are compared, so no normalisation is needed.
- **Agreement wins** — a chunk both retrievers ranked accumulates two terms
  and beats a chunk ranked first by one and missed by the other.
- **Diminishing returns** — the gap between ranks 1 and 2 is larger than
  between 9 and 10, matching how much rank position actually means.

`k` damps the top ranks: larger `k` favours broad agreement, smaller `k`
sharpens the advantage of being ranked first.

## Reranking

`HeuristicReranker` scores three transparent signals: **term coverage** (what
fraction of distinct query terms appear), **proximity** (whether matches
cluster together), and **section-title match**.

A production reranker is a **cross-encoder** that reads query and passage
jointly and scores relevance directly. This is not that, and does not claim to
be. The heuristics correlate with relevance often enough to be useful and cost
nothing. `PassthroughReranker` exists to measure whether reranking helps at
all, by comparing against pure retrieval order.

Ties fall back to the fused score, so reranking never overrides retrieval
where its own signals are indifferent.

## Citation validation

The model cites evidence as `[1]`, `[2]` — indexes into the numbered evidence
block it was given, not free-form source names. That is what makes validation
mechanical: an index either refers to a passage that was retrieved, or it does
not.

`validate_citations` checks every marker against the evidence list.
`format_evidence` produces the numbering. The two functions are the halves of
one contract and must stay in step.

An answer containing a marker outside the evidence range is **rejected**, not
displayed with a broken footnote. A fabricated citation therefore cannot reach
the user.

## Abstention: three gates

1. **Evidence gate.** If nothing clears `MIN_EVIDENCE_SCORE` (0.25), the
   system abstains *without calling the model* — cheaper, and it removes any
   possibility of an unsupported answer.
2. **Schema gate.** The reply is validated against the `GroundedAnswer`
   Pydantic model. The model may also set `sufficient_evidence: false`, which
   is honoured.
3. **Citation gate.** Every marker must resolve.

Abstention is a first-class outcome, not an error. A wrong answer about your
own architecture is worse than "I could not find that."

## Evaluation

Retrieval metrics (do the right chunks come back?):

- **Recall@K** — of the chunks marked relevant, what fraction appear in the
  top K. Measures whether the evidence was found at all.
- **Precision@K** — of the top K returned, what fraction are relevant.
  Measures how much noise the generator must read past.
- **MRR** — mean of `1 / rank of the first relevant chunk`. Measures whether
  the best evidence is near the top, which matters because generation only
  sees top-k.

Generation metrics (is the answer trustworthy?):

- **Citation validity rate** — answers whose every marker resolved.
- **Grounded answer rate** — answered, with all citations valid.
- **Abstention correctness** — abstained exactly when it should have.

These are reported separately and never averaged together. Retrieval can be
perfect while generation is poor, and the opposite; one number would hide
both. Relevance labels are assigned by source document, so recall and
precision measure *whether the right document was found*, which is coarser
than passage-level ground truth and is stated as such.

## Observability

Ingestion duration, retrieval latency, rerank latency, candidate counts,
citation-validation failures and abstentions are recorded through the Phase 6
tracker, under the `retrieval` component.

## Limitations

- The default embedder is lexical, not semantic. Paraphrase without word
  overlap will miss.
- **Lexical scoring cannot distinguish "mentions" from "explains".** Two
  evaluation cases record this: a query about Reciprocal Rank Fusion ranks the
  README (which mentions it once) above the RAG document (which explains it),
  and a query about P95 latency ranks a design-decision entry that uses
  `p95_latency_ms` as an example above the document defining the formula. In
  both cases the right document is still retrieved, at document-rank 2, so the
  generator sees it. The cases are tagged `known-limitation` and pin the
  current rank, so a further regression still fails. A trained embedding model
  is the fix; weight-tuning the heuristic reranker to pass these two queries
  would be overfitting to the test set.
- The heuristic reranker is not a trained cross-encoder.
- Relevance labels are document-level, not passage-level.
- The corpus is six documents. Metrics on a corpus this small are
  demonstrations of method, not stable measurements.
- The index is rebuilt in memory at startup. There is no persistence and no
  incremental re-indexing of changed files beyond content-hash detection.
