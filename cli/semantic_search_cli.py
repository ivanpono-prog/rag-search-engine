import argparse
import re
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, ChunkedSemanticSearch, semantic_chunk
from lib.semantic_search import SemanticSearch
from search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for text"
    )
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser(
        "embed_query",
        help="Generate an embedding for a search query"
    )

    embed_query_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    subparsers.add_parser(
        "verify_embeddings",
        help="Verify the movie embeddings"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic similarity"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5)

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Overlapping words between chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into sentence-based chunks"
    )

    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0
    )

    subparsers.add_parser(
        "embed_chunks",
        help="Generate or load chunk embeddings"
    )

    chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search movies using chunk embeddings"
    )

    chunked_parser.add_argument("query", type=str)
    chunked_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()



    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            search = SemanticSearch()
            documents = load_movies()
            search.load_or_create_embeddings(documents)

            results = search.search(args.query, args.limit)

            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}")
                print()

        case "chunk":
            words = args.text.split()
            chunk_size = args.chunk_size
            overlap = args.overlap

            if chunk_size <= 0:
                parser.error("--chunk-size must be greater than zero")

            if overlap < 0 or overlap >= chunk_size:
                parser.error("--overlap must be non-negative and smaller than chunk size")

            chunks = []
            step = chunk_size - overlap

            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)

                if i + chunk_size >= len(words):
                    break

            print(f"Chunking {len(args.text)} characters")

            for number, chunk in enumerate(chunks, start=1):
                print(f"{number}. {chunk}")

        case "semantic_chunk":
            text = args.text
            max_chunk_size = args.max_chunk_size
            overlap = args.overlap

            if max_chunk_size <= 0:
                parser.error("--max-chunk-size must be greater than zero")

            if overlap < 0 or overlap >= max_chunk_size:
                parser.error("--overlap must be non-negative and smaller than max-chunk-size")

            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            sentences = [sentence for sentence in sentences if sentence]

            chunks = []
            step = max_chunk_size - overlap

            for i in range(0, len(sentences), step):
                chunk = " ".join(sentences[i:i + max_chunk_size])
                chunks.append(chunk)

                if i + max_chunk_size >= len(sentences):
                    break

            print(f"Semantically chunking {len(text)} characters")

            for number, chunk in enumerate(chunks, start=1):
                print(f"{number}. {chunk}")

        case "embed_chunks":
            documents = load_movies()
            search = ChunkedSemanticSearch()
            embeddings = search.load_or_create_chunk_embeddings(documents)

            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            documents = load_movies()

            search = ChunkedSemanticSearch()
            search.load_or_create_chunk_embeddings(documents)

            results = search.search_chunks(args.query, args.limit)

            for i, result in enumerate(results, start=1):
                print(
                    f"\n{i}. {result['title']} "
                    f"(score: {result['score']:.4f})"
                )
                print(f"   {result['document']}...")


if __name__ == "__main__":
    main()
