import json
import argparse
import string
import math
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex
from search_utils import tokenize_text, tokenize_term, load_stopwords
from constants import BM25_K1, BM25_B

stemmer = PorterStemmer()


def matches_query(query: str, title: str, stopwords: list[str]) -> bool:
    translator = str.maketrans("", "", string.punctuation)

    query_tokens = query.lower().translate(translator).split()
    title_tokens = title.lower().translate(translator).split()

    query_tokens = [token for token in query_tokens if token not in stopwords]
    title_tokens = [token for token in title_tokens if token not in stopwords]

    query_tokens = [stemmer.stem(token) for token in query_tokens]
    title_tokens = [stemmer.stem(token) for token in title_tokens]

    for token in query_tokens:
        for title_token in title_tokens:
            if token in title_token:
                return True

    return False


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()

def bm25_idf_command(term):
    index = InvertedIndex()
    index.load()

    token = tokenize_term(term, index.stopwords)
    return index.get_bm25_idf(token)

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    index = InvertedIndex()
    index.load()

    token = tokenize_term(term, index.stopwords)
    return index.get_bm25_tf(doc_id, token, k1, b)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    subparsers.add_parser("build", help="Build the inverted index")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

    idf_parser = subparsers.add_parser("idf", help="Calculate inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term to calculate IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )

    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1,
        help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B,
        help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search",
        help="Search movies using full BM25 scoring"
    )

    bm25search_parser.add_argument(
        "query", type=str, help="Search query"
    )

    bm25search_parser.add_argument(
        "--limit", type=int, default=5,
        help="Maximum number of results"
    )

    args = parser.parse_args()



    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError:
                print("Error: Index not found. Run the build command first.")
                return

            tokens = tokenize_text(args.query, index.stopwords)
            results = set()

            for token in tokens:
                docs = index.get_documents(token)
                results.update(docs)

            for number, doc_id in enumerate(sorted(results)[:5], start=1):
                movie = index.docmap[doc_id]
                print(f"{number}. {movie['title']} (ID: {doc_id})")

        case "build":
            build_command()

        case "tf":
            index = InvertedIndex()
            index.load()

            token = tokenize_term(args.term, index.stopwords)
            frequency = index.get_tf(args.doc_id, token)

            print(frequency)

        case "idf":
            index = InvertedIndex()
            index.load()

            token = tokenize_term(args.term, index.stopwords)


            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(token))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            index = InvertedIndex()
            index.load()

            token = tokenize_term(args.term, index.stopwords)

            tf = index.get_tf(args.doc_id, token)

            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(token))

            tf_idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1)) * tf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            index = InvertedIndex()
            index.load()

            results = index.bm25_search(args.query, args.limit)

            for number, (doc_id, score) in enumerate(results, start=1):
                movie = index.docmap[doc_id]
                print(f"{number}. ({doc_id}) {movie['title']} - Score: {score:.2f}")

        case _:
            parser.print_help()



if __name__ == "__main__":
    main()
