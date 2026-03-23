import json
from pathlib import Path
from smolagents import tool
from src.vector_store import VectorStore

DATA_DIR = Path(__file__).parent.parent / "data"

# Shared vector store instance
_vector_store = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        vs = VectorStore()
        try:
            vs.load_index()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Vector database not found. Please build the index first by running "
                "'python build_index.py' or using the 'Build Index' section in the UI."
            )
        _vector_store = vs
    return _vector_store


def reset_vector_store():
    global _vector_store
    _vector_store = None


def get_papers_db() -> list[dict]:
    papers_path = DATA_DIR / "papers.json"
    if not papers_path.exists():
        raise FileNotFoundError(
            "Papers database not found. Please build the index first by running "
            "'python build_index.py' or using the 'Build Index' section in the UI."
        )
    with open(papers_path) as f:
        return json.load(f)


# --- Tool definitions for smolagents ---

@tool
def search_papers(query: str, top_k: int = 5) -> str:
    """Semantic search across all indexed research papers. Returns the most relevant text chunks with paper titles and scores. Use this when you need to find papers or information about a broad topic.

    Args:
        query: The search query to find relevant papers/content.
        top_k: Number of results to return (default 5).
    """
    try:
        vs = get_vector_store()
    except FileNotFoundError as e:
        return f"Error: {e}"
    results = vs.search(query, top_k=top_k)
    if not results:
        return "No results found."
    output = []
    for i, r in enumerate(results, 1):
        output.append(
            f"[{i}] Paper: {r['paper_title']} (ID: {r['paper_id']})\n"
            f"    Score: {r['score']:.3f}\n"
            f"    Content: {r['text'][:500]}"
        )
    return "\n\n".join(output)


@tool
def get_paper_metadata(paper_id: str) -> str:
    """Get metadata (title, authors, date, abstract, URL) for a specific paper by its ID. Use this when you need details about a specific paper.

    Args:
        paper_id: The arxiv paper ID (e.g., '2312.10997').
    """
    try:
        papers = get_papers_db()
    except FileNotFoundError as e:
        return f"Error: {e}"
    for p in papers:
        if p["id"] == paper_id:
            return (
                f"Title: {p['title']}\n"
                f"Authors: {', '.join(p['authors'])}\n"
                f"Published: {p['published']}\n"
                f"Categories: {', '.join(p['categories'])}\n"
                f"URL: {p['url']}\n"
                f"Abstract: {p['abstract']}"
            )
    return f"Paper with ID '{paper_id}' not found."


@tool
def search_within_paper(paper_id: str, query: str) -> str:
    """Search within a specific paper's chunks for relevant content. Use this when you already know which paper to look in and want to find specific information within it.

    Args:
        paper_id: The arxiv paper ID to search within.
        query: What to search for within the paper.
    """
    try:
        vs = get_vector_store()
    except FileNotFoundError as e:
        return f"Error: {e}"
    results = vs.search_by_paper(paper_id, query)
    if not results:
        return f"No content found for paper '{paper_id}'."
    output = []
    for i, r in enumerate(results, 1):
        output.append(
            f"[{i}] Score: {r['score']:.3f}\n"
            f"    Content: {r['text'][:500]}"
        )
    return "\n\n".join(output)


@tool
def list_all_papers() -> str:
    """List all indexed papers with their titles and IDs. Use this to see what papers are available in the knowledge base.

    Args:
    """
    try:
        papers = get_papers_db()
    except FileNotFoundError as e:
        return f"Error: {e}"
    output = []
    for p in papers:
        output.append(f"- {p['title']} (ID: {p['id']})")
    return "\n".join(output)


ALL_TOOLS = [search_papers, get_paper_metadata, search_within_paper, list_all_papers]
