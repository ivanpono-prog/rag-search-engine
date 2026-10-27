from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode([text])[0]

        return embedding

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        texts = []

        for doc in documents:
            self.document_map[doc["id"]] = doc

            text = f"{doc['title']}: {doc['description']}"
            texts.append(text)

        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc

        cache_path = "cache/movie_embeddings.npy"

        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        scores = []

        for i, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, embedding)
            document = self.documents[i]
            scores.append((similarity, document))

        scores.sort(key=lambda item: item[0], reverse=True)

        results = []
        for score, document in scores[:limit]:
            results.append({
                "score": float(score),
                "title": document["title"],
                "description": document["description"],
            })

        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(documents):
            description = doc.get("description", "")

            if not description or not description.strip():
                continue

            chunks = semantic_chunk(
                description,
                max_chunk_size=4,
                overlap=1
            )

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                })

        if all_chunks:
            self.chunk_embeddings = self.model.encode(
                all_chunks,
                show_progress_bar=True
            )
        else:
            self.chunk_embeddings = np.empty(
                (0, self.model.get_sentence_embedding_dimension())
            )

        self.chunk_metadata = chunk_metadata

        os.makedirs("cache", exist_ok=True)

        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks)
                },
                f,
                indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self, documents: list[dict]
    ) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        embeddings_path = "cache/chunk_embeddings.npy"
        metadata_path = "cache/chunk_metadata.json"

        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            embeddings = np.load(embeddings_path)

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if (
                len(embeddings) == len(metadata["chunks"])
                and all(
                    0 <= chunk["movie_idx"] < len(documents)
                    for chunk in metadata["chunks"]
                )
            ):
                self.chunk_embeddings = embeddings
                self.chunk_metadata = metadata["chunks"]
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("Chunk embeddings not loaded")

        query_embedding = self.generate_embedding(query)

        chunk_scores = []

        for embedding, metadata in zip(
            self.chunk_embeddings, self.chunk_metadata
        ):
            score = cosine_similarity(query_embedding, embedding)

            chunk_scores.append({
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": score,
            })

        movie_scores = {}

        for chunk in chunk_scores:
            movie_idx = chunk["movie_idx"]
            score = chunk["score"]

            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        sorted_movies = sorted(
            movie_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]

            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(float(score), 3),
            })

        return results

def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()

    with open("data/movies.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = data["movies"]

    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_chunk(text, max_chunk_size=4, overlap=0):
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("Invalid overlap")

    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    step = max_chunk_size - overlap

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + max_chunk_size]).strip()

        if chunk:
            chunks.append(chunk)

        if i + max_chunk_size >= len(sentences):
            break

    return chunks
