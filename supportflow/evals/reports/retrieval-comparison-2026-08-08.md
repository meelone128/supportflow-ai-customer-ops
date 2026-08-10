# Retrieval comparison — 2026-08-08

## Scope

The same 9 versioned, de-identified retrieval cases were evaluated against the current four-document demonstration policy corpus. The Bailian run used `text-embedding-v4`; no customer ticket data was sent.

| Retriever | Hit@1 | MRR | Average latency | P95 latency | Actual mode |
| --- | ---: | ---: | ---: | ---: | --- |
| Local TF-IDF | 1.000 | 1.000 | 0.3 ms | 1 ms | `local_tfidf` |
| Bailian semantic | 0.889 | 0.944 | 260.9 ms | 1408 ms | `bailian_semantic` |

## Decision

Do not claim that adding embeddings automatically improves this product. On this small corpus, the cases use policy terms that closely match their source documents, so the lexical baseline is both more accurate and materially faster.

Keep the Bailian adapter as a configurable retrieval option and rerun the experiment after expanding the corpus with paraphrased, cross-document queries. A future hybrid or reranking change must improve the versioned evaluation set before becoming the default.

## Reproduction

```powershell
python -m supportflow.run_retrieval_experiment
python -m supportflow.run_retrieval_experiment --with-bailian
```
