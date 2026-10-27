# RAG Search Engine

A learning project I built while going through the AI Engineering course on Boot.dev.

The idea was to explore how modern search engines work, from basic keyword matching to semantic search, hybrid retrieval, and retrieval-augmented generation (RAG).

The project uses a movie dataset, so most examples involve searching for films using descriptions, keywords, natural-language questions, or even images.

## What's inside

Over the course of the project, I implemented and experimented with:

- **Keyword search** using BM25
- **Semantic search** using text embeddings
- **Hybrid search** combining both approaches with Reciprocal Rank Fusion (RRF)
- **Re-ranking** using cross-encoders and LLMs
- **Query rewriting and expansion**
- **RAG** to generate answers based on retrieved documents
- **LLM-based evaluation** of search results
- **Multimodal search** using CLIP to match images with movie descriptions

Most of these features are accessible through separate Python CLI scripts.

## Tech

Python, uv, Sentence Transformers, BM25, CLIP, Pillow, and LLM APIs through OpenRouter.

## Running the project

Install dependencies:

```bash
uv sync
```

For example, to run a hybrid search with cross-encoder re-ranking:

```bash
uv run cli/hybrid_search_cli.py rrf-search "dinosaur" --rerank-method cross_encoder
```

Or to generate an answer using RAG:

```bash
uv run cli/augmented_generation_cli.py rag "movies about dinosaurs"
```

Some features require an API key and access to an LLM provider.

## A few takeaways

The most interesting part for me was seeing how different retrieval methods can produce quite different results for the same query.

Keyword search is useful when the wording matters, while semantic search can find relevant documents even when they don't share the same keywords. Combining them helps, but doesn't automatically guarantee better results.

I also got to experiment with re-ranking, evaluation metrics
