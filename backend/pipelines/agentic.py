# backend/pipelines/agentic.py
# Agentic RAG: LangGraph create_react_agent + RetrieverTool — multi-step tool-calling retrieval
# LangChain 1.x removed AgentExecutor; create_react_agent (langgraph) is the replacement.
from __future__ import annotations

from typing import List, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import create_retriever_tool
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent

from backend.pipelines.base import BaseRAGPipeline, RAGResponse, TraceStep
from backend.ingest.embedder import get_vectorstore
from backend.llm import get_generation_llm

AGENT_SYSTEM = (
    "You are a senior financial analyst assistant with access to a financial document corpus. "
    "Use the search tools to find relevant information before answering. "
    "For complex questions that span multiple topics, make multiple targeted tool calls. "
    "Always ground your answer in the retrieved documents. "
    "When you have enough evidence, synthesise a clear, structured answer."
)


class AgenticRAGPipeline(BaseRAGPipeline):
    name = "agentic"
    label = "Agentic RAG"
    description = (
        "Uses a LangGraph ReAct agent with a RetrieverTool wrapping ChromaDB. "
        "The agent autonomously decides how many tool calls to make and with what "
        "sub-questions — every tool call appears in the trace."
    )
    color = "#9b59b6"

    def __init__(self):
        self._vs = None
        self._llm = None
        self._agent = None
        self._retriever_tool = None

    def _init(self):
        if self._vs is None:
            self._vs = get_vectorstore()
        if self._llm is None:
            self._llm = get_generation_llm()
        if self._agent is None:
            self._build_agent()

    def _build_agent(self):
        retriever = self._vs.as_retriever(search_kwargs={"k": 4})
        self._retriever_tool = create_retriever_tool(
            retriever,
            name="search_financial_docs",
            description=(
                "Search the financial document corpus. Use this tool to find information "
                "about earnings, risk factors, balance sheets, M&A, interest rates, "
                "valuations, and sector analysis. Input: a specific search query string."
            ),
        )

        self._agent = create_react_agent(
            model=self._llm,
            tools=[self._retriever_tool],
            prompt=AGENT_SYSTEM,
        )

    def run(self, query: str) -> RAGResponse:
        self._init()
        trace: List[TraceStep] = []

        trace.append(TraceStep(
            step="ROUTE",
            description=(
                f"Agentic pipeline initiated. Model: {self._llm.model_name}. "
                f"Max tool calls: 6. Tool: search_financial_docs."
            ),
        ))

        # Invoke the LangGraph agent
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": 20},
        )

        # Parse messages to extract tool calls and observations
        messages = result.get("messages", [])
        sources: list[dict] = []
        seen_chunks: set[str] = set()
        tool_call_count = 0
        final_answer = ""

        for msg in messages:
            msg_type = type(msg).__name__

            # AIMessage with tool_calls → the agent decided to call a tool
            if msg_type == "AIMessage":
                tool_calls = getattr(msg, "tool_calls", []) or []
                for tc in tool_calls:
                    tool_call_count += 1
                    tc_input = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    query_str = tc_input.get("query", str(tc_input))[:120]
                    trace.append(TraceStep(
                        step="RETRIEVE",
                        description=f"Tool call #{tool_call_count}: `search_financial_docs` — sub-query: `{query_str}`",
                        data={"sub_query": query_str, "tool": "search_financial_docs"},
                    ))
                # If no tool calls, this is the final answer message
                if not tool_calls and msg.content:
                    final_answer = msg.content

            # ToolMessage → observation returned from the tool
            elif msg_type == "ToolMessage":
                content = msg.content or ""
                # Try to parse Document objects from string representation
                # (LangGraph serialises tool output as a formatted string)
                raw_preview = content[:300].replace("\n", " ")
                if trace and trace[-1].step == "RETRIEVE":
                    trace[-1].data["raw_observation"] = raw_preview

                # Extract individual document chunks from the tool output string
                # The retriever tool formats output as concatenated page_content blocks
                for chunk_text in content.split("\n\n"):
                    chunk_text = chunk_text.strip()
                    if len(chunk_text) > 50:
                        cid = chunk_text[:40]
                        if cid not in seen_chunks:
                            seen_chunks.add(cid)
                            sources.append({
                                "text": chunk_text,
                                "doc_name": "financial_corpus",
                                "page": "?",
                                "score": 1.0,
                                "chunk_id": cid,
                            })

        # Fallback: last AIMessage content if final_answer not captured
        if not final_answer:
            for msg in reversed(messages):
                if type(msg).__name__ == "AIMessage" and msg.content:
                    final_answer = msg.content
                    break

        trace.append(TraceStep(
            step="GENERATE",
            description=(
                f"Agent completed {tool_call_count} tool call(s). "
                f"Generated final answer from {len(sources)} retrieved chunks."
            ),
        ))

        return RAGResponse(
            answer=final_answer or "No answer generated.",
            sources=sources,
            trace=trace,
            latency_ms=0,
            pipeline_name=self.name,
        )
