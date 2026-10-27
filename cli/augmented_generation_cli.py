import argparse
from openai import OpenAI
from dotenv import load_dotenv
import os

from search_utils import load_movies
from lib.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    rag_parser = subparsers.add_parser(
        "rag",
        help="Perform RAG (search + generate answer)"
    )

    rag_parser.add_argument(
        "query",
        type=str,
        help="Search query for RAG"
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize multiple search results"
    )

    summarize_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to summarize"
    )

    citations_parser = subparsers.add_parser(
        "citations",
        help="Generate answers with source citations"
    )

    citations_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results"
    )

    question_parser = subparsers.add_parser(
        "question",
        help="Answer questions about movies"
    )

    question_parser.add_argument(
        "question",
        type=str,
        help="Question about movies"
    )

    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            documents = load_movies()
            search = HybridSearch(documents)

            results = search.rrf_search(
                query,
                k=60,
                limit=5
            )

            docs = "\n\n".join(
                f"Title: {doc['title']}\n"
                f"Description: {doc.get('description', doc.get('document', ''))}"
                for doc in results
            )

            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {docs}

            Answer:"""

            load_dotenv()

            client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content
            print("Search Results:")

            for doc in results:
                print(f"- {doc['title']}")

            print("\nRAG Response:")
            print(answer)

        case "summarize":
            query = args.query

            documents = load_movies()
            search = HybridSearch(documents)

            results = search.rrf_search(
                query,
                k=60,
                limit=args.limit
            )

            docs = "\n\n".join(
                f"Title: {doc['title']}\n"
                f"Description: {doc.get('description', doc.get('document', ''))}"
                for doc in results
            )

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results into a single response.

            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the movies.

            This should be tailored to Webflyx users. Webflyx is a movie streaming service.

            Query: {query}

            Search results:
            {docs}

            Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""

            load_dotenv()

            client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            summary = response.choices[0].message.content

            print("Search Results:")

            for doc in results:
                print(f"- {doc['title']}")

            print("\nLLM Summary:")
            print(summary)

        case "citations":
            query = args.query

            documents = load_movies()
            search = HybridSearch(documents)

            results = search.rrf_search(
                query,
                k=60,
                limit=args.limit
            )

            formatted_docs = []

            for i, doc in enumerate(results, start=1):
                formatted_docs.append(
                    f"[{i}] {doc['title']}\n"
                    f"{doc.get('description', doc.get('document', ''))}"
                )

            documents = "\n\n".join(formatted_docs)

            prompt = f"""Answer the query below and give information based on the provided documents.

            The answer should be tailored to users of Webflyx, a movie streaming service.
            If not enough information is available to provide a good answer, say so, but give the best answer possible.

            Query: {query}

            Documents:
            {documents}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources in the format [1], [2], etc. when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the provided documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""

            load_dotenv()

            client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

            print("Search Results:")

            for doc in results:
                print(f"- {doc['title']}")

            print("\nLLM Answer:")
            print(answer)

        case "question":
            question = args.question

            documents = load_movies()
            search = HybridSearch(documents)

            results = search.rrf_search(
                question,
                k=60,
                limit=args.limit
            )

            context = "\n\n".join(
                f"Title: {doc['title']}\n"
                f"Description: {doc.get('description', doc.get('document', ''))}"
                for doc in results
            )

            prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

            Question: {question}

            Documents:
            {context}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""

            load_dotenv()

            client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

            print("Search Results:")

            for doc in results:
                print(f"- {doc['title']}")

            print("\nAnswer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
