import arxiv
import json
import os
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_papers(query: str, max_results: int = 50) -> list[dict]:
    """Fetch papers from arxiv and return structured data."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        paper = {
            "id": result.entry_id.split("/")[-1],
            "title": result.title,
            "abstract": result.summary,
            "authors": [a.name for a in result.authors],
            "published": result.published.isoformat(),
            "categories": result.categories,
            "url": result.entry_id,
        }
        papers.append(paper)

    return papers


def chunk_paper(paper: dict, chunk_size: int = 500) -> list[dict]:
    """Split a paper's content into chunks for embedding."""
    chunks = []

    # Title + abstract as first chunk
    chunks.append({
        "paper_id": paper["id"],
        "paper_title": paper["title"],
        "chunk_type": "abstract",
        "text": f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}",
    })

    # Split abstract into smaller chunks if it's long
    abstract = paper["abstract"]
    words = abstract.split()
    if len(words) > chunk_size:
        for i in range(0, len(words), chunk_size // 2):  # 50% overlap
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 50:
                continue
            chunks.append({
                "paper_id": paper["id"],
                "paper_title": paper["title"],
                "chunk_type": "abstract_part",
                "text": f"From paper '{paper['title']}':\n{' '.join(chunk_words)}",
            })

    return chunks


def fetch_and_store(query: str, max_results: int = 50) -> tuple[list[dict], list[dict]]:
    """Fetch papers and create chunks. Returns (papers, chunks)."""
    DATA_DIR.mkdir(exist_ok=True)

    papers = fetch_papers(query, max_results)

    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunk_paper(paper))

    # Save to disk
    with open(DATA_DIR / "papers.json", "w") as f:
        json.dump(papers, f, indent=2)

    with open(DATA_DIR / "chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    return papers, all_chunks
