from backend.pipelines.agentic import AgenticRAGPipeline
from backend.pipelines.graph_rag import GraphRAGPipeline
from backend.pipelines.self_rag import SelfRAGPipeline
from backend.pipelines.crag import CRAGPipeline
from backend.pipelines.adaptive import AdaptiveRAGPipeline
from backend.pipelines.hyde import HyDEPipeline
from backend.pipelines.fusion import FusionRAGPipeline

ALL_PIPELINES = [
    AgenticRAGPipeline,
    GraphRAGPipeline,
    SelfRAGPipeline,
    CRAGPipeline,
    AdaptiveRAGPipeline,
    HyDEPipeline,
    FusionRAGPipeline,
]

# Keyed by pipeline.name (short slug)
PIPELINE_MAP: dict[str, type] = {p.name: p for p in ALL_PIPELINES}

# Ordered display list with metadata
PIPELINE_OPTIONS = [
    {
        "name": p.name,
        "label": p.label,
        "description": p.description,
        "color": p.color,
    }
    for p in ALL_PIPELINES
]
