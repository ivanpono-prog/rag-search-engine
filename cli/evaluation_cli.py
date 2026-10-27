import argparse
import json

from lib.hybrid_search import HybridSearch
from search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)

    documents = load_movies()
    search = HybridSearch(documents)

    print(f"k={limit}")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant = test_case["relevant_docs"]

        results = search.rrf_search(query, k=60, limit=limit)

        retrieved = [result["title"] for result in results]

        relevant_found = sum(
            1 for title in retrieved if title in relevant
        )

        precision = relevant_found / limit
        recall = len(set(retrieved) & set(relevant)) / len(relevant) if relevant else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant)}")


if __name__ == "__main__":
    main()
