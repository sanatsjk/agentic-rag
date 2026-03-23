"""
Agentic RAG — Research Paper Assistant.

Uses smolagents with Llama-3.3-70B to search and reason over arxiv papers.
Deployed on Hugging Face Spaces with Gradio.
"""

import os
from dotenv import load_dotenv
import gradio as gr
from src.agent import run_agent_streaming

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")


def answer_question(message: str, history: list) -> str:
    """Agentic RAG pipeline: search, reason, and answer."""
    if not message.strip():
        return "Please enter a question."

    if not HF_TOKEN:
        return "HF_TOKEN env var is not set."

    # Check that the vector database exists before running the agent
    index_path = os.path.join(os.path.dirname(__file__), "data", "faiss.index")
    if not os.path.exists(index_path):
        return (
            "Vector database not found. Please build the index first using the "
            "'Build Index' section below before asking questions."
        )

    # Run the agent and collect steps
    reasoning_log = []
    final_answer = ""

    for step in run_agent_streaming(message, HF_TOKEN):
        if step["type"] in ("tool_call", "tool_result"):
            reasoning_log.append(step["content"])
        elif step["type"] == "answer":
            final_answer = step["content"]

    # Append reasoning as collapsible section
    if reasoning_log:
        refs = "\n".join(f"{i}. {line}" for i, line in enumerate(reasoning_log, 1))
        reasoning = f"\n\n<details><summary>Show Agent Reasoning</summary>\n\n{refs}\n\n</details>"
        return final_answer + reasoning

    return final_answer


def build_index(query: str, max_results: int):
    """Build the paper index from arxiv."""
    from src.arxiv_fetcher import fetch_and_store
    from src.vector_store import VectorStore
    from src.tools import reset_vector_store

    reset_vector_store()

    papers, chunks = fetch_and_store(query, int(max_results))
    vs = VectorStore()
    vs.build_index(chunks)
    return f"Indexed {len(papers)} papers ({len(chunks)} chunks) for query: '{query}'"


# --- Gradio UI ---

with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=answer_question,
        title="Research Paper Assistant",
        description=(
            "Ask questions about AI/ML research papers! This app uses Agentic RAG — "
            "an LLM agent that searches, reasons over, and synthesizes answers from "
            "arxiv papers indexed in a FAISS vector store."
        ),
        examples=[
            "What are the latest approaches to improving RAG accuracy?",
            "List all papers in the knowledge base",
            "How does chunking strategy affect retrieval quality?",
            "Compare different retrieval methods proposed in recent papers",
        ],
    )

    with gr.Accordion("Build Index", open=False):
        gr.Markdown("Fetch papers from arxiv and build the search index. Do this once before chatting.")
        query_input = gr.Textbox(
            value="retrieval augmented generation LLM",
            label="Arxiv Search Query",
        )
        max_results_input = gr.Slider(
            minimum=10, maximum=100, value=50, step=10, label="Max Papers"
        )
        run_build_btn = gr.Button("Build", variant="primary")
        build_output = gr.Textbox(label="Status", interactive=False)

        run_build_btn.click(build_index, [query_input, max_results_input], build_output)


if __name__ == "__main__":
    demo.launch()
