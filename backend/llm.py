# backend/llm.py
# Single module that creates LLM clients — import from here, nowhere else
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from backend.config import (
    get_openrouter_api_key,
    get_openrouter_base_url,
    get_generation_model,
    get_grading_model,
    get_embedding_model,
)

_HEADERS = {
    "HTTP-Referer": "https://rag-showcase.demo",
    "X-Title": "RAG Architecture Showcase",
}


def get_generation_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=get_generation_model(),
        openai_api_key=get_openrouter_api_key(),
        openai_api_base=get_openrouter_base_url(),
        streaming=streaming,
        temperature=0.0,
        default_headers=_HEADERS,
    )


def get_grading_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=get_grading_model(),
        openai_api_key=get_openrouter_api_key(),
        openai_api_base=get_openrouter_base_url(),
        temperature=0.0,
        default_headers=_HEADERS,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """OpenRouter proxies OpenAI embedding endpoints."""
    return OpenAIEmbeddings(
        model=get_embedding_model(),
        openai_api_key=get_openrouter_api_key(),
        openai_api_base=get_openrouter_base_url(),
        default_headers=_HEADERS,
    )
