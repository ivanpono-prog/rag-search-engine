import argparse
import logging
from lib.hybrid_search import (
    HybridSearch,
    correct_spelling,
    rewrite_query,
    expand_query,
    rerank_individual,
    rerank_batch,
    rerank_cross_encoder,
    evaluate_results,
)

def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="DEBUG: %(message)s",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Search using weighted BM25 and semantic scores"
    )

    weighted_parser.add_argument("query", type=str)
    weighted_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_parser.add_argument("--limit", type=int, default=5)


    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Search using Reciprocal Rank Fusion"
    )
    rrf_parser.add_argument("query", type=str)
    rrf_parser.add_argument("-k", type=int, default=60)
    rrf_parser.add_argument("--limit", type=int, default=5)
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        default=None,
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        default=None,
        help="LLM re-ranking method",
    )
    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate search results using an LLM",
    )



    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize a list of scores"
    )

    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="*"
    )

    args = parser.parse_args()

    match args.command:

        case "normalize":
            scores = args.scores

            if not scores:
                return

            min_score = min(scores)
            max_score = max(scores)

            if min_score == max_score:
                normalized = [1.0 for score in scores]
            else:
                normalized = [
                    (score - min_score) / (max_score - min_score)
                    for score in scores
                ]

            for score in normalized:
                print(f"* {score:.4f}")

        case "weighted-search":
            from search_utils import load_movies

            documents = load_movies()
            search = HybridSearch(documents)

            results = search.weighted_search(
                args.query,
                args.alpha,
                args.limit,
            )

            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"   BM25: {result['bm25_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(f"   {result.get('document', result.get('description', ''))[:100]}...")


        case "rrf-search":
            from search_utils import load_movies

            documents = load_movies()
            search = HybridSearch(documents)

            query = args.query
            logging.debug(f"Original query: {query}")

            if args.enhance == "spell":
                enhanced_query = correct_spelling(query)

            elif args.enhance == "rewrite":
                enhanced_query = rewrite_query(query)

            elif args.enhance == "expand":
                enhanced_query = expand_query(query)

            else:
                enhanced_query = query

            if args.enhance:
                print(
                    f"Enhanced query ({args.enhance}): "
                    f"'{query}' -> '{enhanced_query}'\n"
                )

            query = enhanced_query

            candidate_limit = args.limit

            if args.rerank_method:
                candidate_limit = args.limit * 5

            logging.debug(f"Enhanced query: {query}")
            results = search.rrf_search(
                query,
                args.k,
                candidate_limit,
            )
            logging.debug(f"RRF results: {results}")

            if args.rerank_method == "individual":
                print(
                    f"Re-ranking top {args.limit} results "
                    f"using individual method..."
                )
                results = rerank_individual(query, results)

            elif args.rerank_method == "batch":
                print(
                    f"Re-ranking top {args.limit} results "
                    f"using batch method..."
                )
                results = rerank_batch(query, results)

            elif args.rerank_method == "cross_encoder":
                print(
                    f"Re-ranking top {args.limit} results "
                    f"using cross_encoder method..."
                )

                results = rerank_cross_encoder(query, results)

            results = results[:args.limit]

            logging.debug(f"Final results: {results}")


            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc['title']}")

                if args.rerank_method == "individual":
                    print(f"   Re-rank Score: {doc['rerank_score']:.3f}/10")

                elif args.rerank_method == "batch":
                    print(f"   Re-rank Rank: {doc['rerank_rank']}")

                elif args.rerank_method == "cross_encoder":
                    print(
                        f"   Cross Encoder Score: "
                        f"{doc['cross_encoder_score']:.3f}"
                    )

                print(f"   RRF Score: {doc['rrf_score']:.3f}")
                print(
                    f"   BM25 Rank: {doc['bm25_rank']}, "
                    f"Semantic Rank: {doc['semantic_rank']}"
                )
                print(f"   {doc.get('description', doc.get('document', ''))[:100]}...")

            if args.evaluate:
                scores = evaluate_results(query, results)

                for i, (doc, score) in enumerate(zip(results, scores), start=1):
                    print(f"{i}. {doc['title']}: {score}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
