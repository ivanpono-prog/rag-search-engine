from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
import os
import json

from inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from sentence_transformers import CrossEncoder


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

        # Initialize semantic search
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # Initialize keyword search
        self.idx = InvertedIndex()

        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float, limit: int = 5
    ) -> list[dict]:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        candidate_limit = limit * 500

        bm25_results = self._bm25_search(query, candidate_limit)
        semantic_results = self.semantic_search.search_chunks(
            query, candidate_limit
        )

        bm25_scores = normalize([score for doc_id, score in bm25_results])
        semantic_scores = normalize([r["score"] for r in semantic_results])

        combined = {}

        for (doc_id, _), score in zip(bm25_results, bm25_scores):
            doc = self.idx.docmap[doc_id]

            combined[doc_id] = {
                **doc,
                "bm25_score": score,
                "semantic_score": 0.0,
            }

        for result, score in zip(semantic_results, semantic_scores):
            doc_id = result["id"]

            if doc_id not in combined:
                combined[doc_id] = {
                    **result,
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }

            combined[doc_id]["semantic_score"] = score

        for result in combined.values():
            result["hybrid_score"] = hybrid_score(
                result["bm25_score"],
                result["semantic_score"],
                alpha,
            )

        return sorted(
            combined.values(),
            key=lambda r: r["hybrid_score"],
            reverse=True,
        )[:limit]


    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        candidate_limit = limit * 500

        bm25_results = self._bm25_search(query, candidate_limit)
        semantic_results = self.semantic_search.search_chunks(query, candidate_limit)

        combined = {}

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            doc = self.idx.docmap[doc_id]
            combined[doc_id] = {
                **doc,
                "bm25_rank": rank,
                "semantic_rank": None,
            }

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]

            if doc_id not in combined:
                combined[doc_id] = {
                    **self.idx.docmap[doc_id],
                    "bm25_rank": None,
                    "semantic_rank": None,
                }

            combined[doc_id]["semantic_rank"] = rank

        for result in combined.values():
            result["rrf_score"] = (
                rrf_score(result["bm25_rank"], k)
                + rrf_score(result["semantic_rank"], k)
            )

        return sorted(
            combined.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )[:limit]



def normalize(scores):
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if minimum == maximum:
        return [1.0] * len(scores)

    return [(score - minimum) / (maximum - minimum) for score in scores]


def hybrid_score(bm25_score, semantic_score, alpha):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    if rank is None:
        return 0.0
    return 1 / (k + rank)

def correct_spelling(query: str) -> str:
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content.strip()

def rewrite_query(query: str) -> str:
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror - scary, animation - cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content.strip()

def expand_query(query: str) -> str:
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "horror grizzly terrifying film"
- "action movie with bear" -> "thriller chase fight adventure"
- "comedy with bear" -> "funny humor lighthearted"

User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    additional_terms = response.choices[0].message.content

    if not additional_terms:
        return query

    return f"{query} {additional_terms.strip()}"

def rerank_individual(query: str, documents: list[dict]) -> list[dict]:
    import time

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - doc.get("description", doc.get("document", ""))

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        raw_score = response.choices[0].message.content

        try:
            doc["rerank_score"] = max(0.0, min(10.0, float(raw_score.strip())))
        except (ValueError, AttributeError):
            doc["rerank_score"] = 0.0

        time.sleep(3)

    return sorted(
        documents,
        key=lambda doc: doc["rerank_score"],
        reverse=True,
    )

def rerank_batch(query: str, documents: list[dict]) -> list[dict]:
    if not documents:
        return []

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    doc_list = []

    for doc in documents:
        description = doc.get("description", doc.get("document", ""))

        doc_list.append(
            f"ID: {doc['id']}\n"
            f"Title: {doc['title']}\n"
            f"Description: {description[:300]}"
        )

    doc_list_str = "\n\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    content = response.choices[0].message.content
    ranked_ids = json.loads(content)

    rank_map = {
        int(doc_id): rank
        for rank, doc_id in enumerate(ranked_ids, start=1)
    }

    for doc in documents:
        doc["rerank_rank"] = rank_map.get(
            int(doc["id"]),
            len(documents) + 1,
        )

    return sorted(
        documents,
        key=lambda doc: doc["rerank_rank"],
    )

def rerank_cross_encoder(
    query: str,
    documents: list[dict],
) -> list[dict]:

    if not documents:
        return []

    pairs = []

    for doc in documents:
        description = doc.get("description", doc.get("document", ""))

        pairs.append([
            query,
            f"{doc.get('title', '')} - {description}"
        ])

    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L2-v2"
    )

    scores = cross_encoder.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["cross_encoder_score"] = float(score)

    return sorted(
        documents,
        key=lambda doc: doc["cross_encoder_score"],
        reverse=True,
    )

def evaluate_results(query: str, results: list[dict]) -> list[int]:
    if not results:
        return []

    formatted_results = []

    for i, doc in enumerate(results, start=1):
        title = doc.get("title", "")
        description = doc.get("document", doc.get("description", ""))

        formatted_results.append(
            f"{i}. {title} - {description}"
        )

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents.
Return a valid JSON list, nothing else.

For example:
[2, 0, 3, 2, 0, 1]"""

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    content = response.choices[0].message.content

    if content is None:
        raise ValueError("LLM returned an empty response")

    scores = json.loads(content)

    if (
        not isinstance(scores, list)
        or len(scores) != len(results)
        or any(type(score) is not int or score not in (0, 1, 2, 3) for score in scores)
    ):
        raise ValueError("LLM returned invalid evaluation scores")

    return scores
