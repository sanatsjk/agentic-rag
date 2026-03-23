"""Script to fetch papers from arxiv and build the FAISS index."""

import argparse
from src.arxiv_fetcher import fetch_and_store
from src.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Build the paper index from arxiv")
    parser.add_argument(
        "--query",
        type=str,
        default="retrieval augmented generation LLM",
        help="Arxiv search query",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum number of papers to fetch",
    )
    args = parser.parse_args()

    print(f"Fetching papers for query: '{args.query}'")
    papers, chunks = fetch_and_store(args.query, args.max_results)
    print(f"Fetched {len(papers)} papers, created {len(chunks)} chunks")

    print("Building FAISS index...")
    vs = VectorStore()
    vs.build_index(chunks)
    print(f"Index built with {len(chunks)} vectors")
    print("Done! Files saved in data/")


if __name__ == "__main__":
    main()
