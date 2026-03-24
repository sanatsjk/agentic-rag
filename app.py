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

    yield "**Building index...** please wait."

    try:
        reset_vector_store()
        papers, chunks = fetch_and_store(query, int(max_results))
        vs = VectorStore()
        vs.build_index(chunks)
        yield f"**Done** — indexed {len(papers)} papers ({len(chunks)} chunks)."
    except Exception as e:
        yield f"**Failed** — {e}"


# --- Gradio UI ---

CUSTOM_CSS = """
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.main-header p {
    opacity: 0.7;
    font-size: 0.95rem;
    max-width: 640px;
    margin: 0 auto;
}
.setup-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 1.25rem;
}
.build-status p {
    font-size: 0.85rem;
    margin: 0.25rem 0 0;
}
.step-badge {
    display: inline-block;
    background: var(--color-accent);
    color: white;
    border-radius: 50%;
    width: 24px; height: 24px;
    text-align: center; line-height: 24px;
    font-weight: 700; font-size: 0.8rem;
    margin-right: 6px;
}
footer { display: none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    radius_size=gr.themes.sizes.radius_md,
)

_GRADIO_MAJOR = int(gr.__version__.split(".")[0])

_blocks_kw = dict(title="Research Paper Assistant")
_launch_kw: dict = {}
if _GRADIO_MAJOR >= 6:
    _launch_kw.update(theme=THEME, css=CUSTOM_CSS)
else:
    _blocks_kw.update(theme=THEME, css=CUSTOM_CSS)

with gr.Blocks(**_blocks_kw) as demo:

    # ── Header ──
    gr.HTML(
        """
        <div class="main-header">
            <h1>Research Paper Assistant</h1>
            <p>Search, reason over, and get answers from arxiv papers — powered by Agentic RAG.</p>
        </div>
        """
    )

    # ── Setup section (collapsible) ──
    with gr.Accordion("Setup — Build Paper Index", open=False, elem_classes="setup-card"):
        gr.Markdown(
            "Fetch papers from **arxiv** and build a FAISS vector index. "
            "You only need to do this once per topic."
        )
        with gr.Row(equal_height=True):
            query_input = gr.Textbox(
                value="retrieval augmented generation LLM",
                label="Arxiv Search Query",
                placeholder="e.g. transformer architecture attention mechanism",
                scale=4,
            )
            max_results_input = gr.Slider(
                minimum=10, maximum=100, value=50, step=10,
                label="Max Papers",
                scale=2,
            )
            run_build_btn = gr.Button("Build Index", variant="primary", scale=0, min_width=120, size="sm")
        build_status = gr.Markdown("")

        run_build_btn.click(
            build_index, [query_input, max_results_input], build_status,
        )

    # ── Chat ──
    _chat_type = {"type": "messages"} if _GRADIO_MAJOR < 6 else {}
    gr.ChatInterface(
        fn=answer_question,
        **_chat_type,
        examples=[
            ["What are the latest approaches to improving RAG accuracy?"],
            ["List all papers in the knowledge base"],
            ["How does chunking strategy affect retrieval quality?"],
            ["Compare different retrieval methods proposed in recent papers"],
        ],
        chatbot=gr.Chatbot(
            **_chat_type,
            height=480,
            placeholder=(
                "<div style='text-align:center; opacity:0.55; padding:2rem 0'>"
                "<p style='font-size:1.1rem; margin-bottom:0.5rem'><strong>Ask anything about your indexed papers</strong></p>"
                "<p style='font-size:0.85rem'>Try one of the example prompts below, or type your own question.</p>"
                "</div>"
            ),
        ),
    )

    # ── How to Use (bottom accordion) ──
    with gr.Accordion("How to Use", open=False):
        gr.Markdown(
            """
**<span class="step-badge">1</span> Build the Index** — Open the **Setup** panel above, enter a topic query, and click **Build Index**.
The app fetches papers from arxiv and stores them in a local vector database.

**<span class="step-badge">2</span> Ask Questions** — Type a question in the chat. The AI agent searches the indexed papers,
reasons over them step-by-step, and returns a synthesized answer with citations.

**<span class="step-badge">3</span> Inspect Reasoning** — Expand *Show Agent Reasoning* in any response to see the full research trace.

> Rebuild the index at any time with a different query to explore a new research area.
"""
        )


if __name__ == "__main__":
    demo.launch(**_launch_kw)
