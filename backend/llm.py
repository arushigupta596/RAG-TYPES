# backend/llm.py
# Single module that creates LLM clients — import from here, nowhere else
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from backend.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GENERATION_MODEL,
    GRADING_MODEL,
    EMBEDDING_MODEL,
)

_HEADERS = {
    "HTTP-Referer": "https://rag-showcase.demo",
    "X-Title": "RAG Architecture Showcase",
}


def get_generation_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=GENERATION_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        streaming=streaming,
        temperature=0.0,
        default_headers=_HEADERS,
    )


def get_grading_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=GRADING_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.0,
        default_headers=_HEADERS,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """OpenRouter proxies OpenAI embedding endpoints."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
    )
