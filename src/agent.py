import os
from smolagents import ToolCallingAgent, LiteLLMModel
from smolagents.memory import ActionStep, FinalAnswerStep
from src.tools import ALL_TOOLS

MODEL_ID = "groq/llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are a research paper assistant with access to a knowledge base of arxiv papers.
Your job is to answer user questions by searching through the indexed papers.

## How to work

1. **Plan** your approach: think about what information you need and which tools to use.
2. **Search broadly** first with `search_papers` to find relevant papers.
3. **Drill down** into specific papers with `search_within_paper` or `get_paper_metadata`.
4. **Synthesize** your findings into a clear, cited answer.
5. **Self-check**: if your retrieved context doesn't fully answer the question, search again with different queries.

## Rules
- Always cite papers by title and ID when referencing them.
- If you can't find relevant information, say so honestly.
- When comparing papers, retrieve information from each one separately.
- Prefer concrete details from the papers over general knowledge.
"""


def create_agent(hf_token: str) -> ToolCallingAgent:
    """Create a smolagents ToolCallingAgent with the paper tools."""
    model = LiteLLMModel(
        model_id=MODEL_ID,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )

    agent = ToolCallingAgent(
        tools=ALL_TOOLS,
        model=model,
        max_steps=10,
    )
    agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT

    return agent


def run_agent(user_message: str, hf_token: str) -> str:
    """Run the agentic RAG loop and return the final answer."""
    agent = create_agent(hf_token)
    result = agent.run(user_message)
    return str(result)


def run_agent_streaming(user_message: str, hf_token: str):
    """Run the agentic RAG loop, yielding intermediate steps and final answer."""
    agent = create_agent(hf_token)

    for step in agent.run(user_message, stream=True):
        if isinstance(step, ActionStep):
            if step.tool_calls:
                for tc in step.tool_calls:
                    yield {
                        "type": "tool_call",
                        "content": f"Using **{tc.name}**({tc.arguments})",
                    }
            if step.observations:
                yield {
                    "type": "tool_result",
                    "content": f"Got {len(step.observations)} chars of results",
                }

        elif isinstance(step, FinalAnswerStep):
            yield {"type": "answer", "content": str(step.output)}
