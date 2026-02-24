# backend/pipelines/self_rag_prompts.py

ISREL_PROMPT = """
Evaluate whether this document chunk is relevant to the question.
Return JSON only: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}

Question: {query}
Chunk: {chunk}
"""

ISSUP_PROMPT = """
Evaluate whether this answer is fully supported by the provided context.
Return JSON only: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}

Context: {context}
Answer: {answer}
"""

CRAG_GRADE_PROMPT = """
Evaluate the overall quality and relevance of the following set of document chunks
for answering the given question.
Return JSON only: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}

Question: {query}
Documents:
{docs_text}
"""

ADAPTIVE_ROUTE_PROMPT = """
Classify the complexity of the following financial question into exactly one tier:
- SIMPLE: single factual lookup, definition, or specific number
- MEDIUM: requires explanation or analysis of 1-2 documents
- COMPLEX: multi-hop reasoning, cross-document comparison, or synthesis

Return JSON only: {{"tier": "SIMPLE"|"MEDIUM"|"COMPLEX", "reason": "<one sentence>"}}

Question: {query}
"""
