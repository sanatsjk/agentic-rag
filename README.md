---
title: Research Paper Assistant
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
---

# 📚 Research Paper Assistant

An **agentic RAG** system that searches and reasons over arxiv research papers using [smolagents](https://huggingface.co/docs/smolagents) and Llama-3.3-70B via the HF Inference API.

## What makes it agentic?

Unlike simple RAG (retrieve → generate), this system uses an **autonomous agent loop**:

1. **Plans** what information it needs
2. **Searches** the paper index with semantic queries
3. **Drills down** into specific papers for details
4. **Self-evaluates** whether it has enough context
5. **Iterates** with refined searches if needed
6. **Synthesizes** a final cited answer

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your HF token
cp .env.example .env
# Edit .env with your Hugging Face token

# Build the paper index
python build_index.py --query "retrieval augmented generation LLM" --max-results 50

# Run the app
python app.py
```

## Deploying to HF Spaces

This repo syncs to Hugging Face Spaces via GitHub Actions. Set these secrets in your GitHub repo:

- `HF_TOKEN` — Your Hugging Face access token
- `HF_SPACE_ID` — Your space ID (e.g., `username/research-paper-assistant`)

## Tech Stack

- **Agent Framework**: [smolagents](https://huggingface.co/docs/smolagents) (ToolCallingAgent)
- **LLM**: Llama-3.3-70B-Instruct (via HF Inference API / Together)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Data Source**: arxiv API
- **Frontend**: Gradio
- **Deployment**: Hugging Face Spaces
