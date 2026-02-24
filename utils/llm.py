"""
Centralised LLM accessor — all calls routed through OpenRouter via LangChain ChatOpenAI.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")


def get_llm(temperature: float = 0.0, streaming: bool = False) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance pointed at OpenRouter.
    All seven pipelines call this function — single place to swap models.
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        streaming=streaming,
        default_headers={
            "HTTP-Referer": "https://emb-rag-demo.local",
            "X-Title": "EMB RAG Demo",
        },
    )
