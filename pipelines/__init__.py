from .naive_rag import NaiveRAGPipeline
from .contextual_compression import ContextualCompressionPipeline
from .multi_query import MultiQueryPipeline
from .hyde import HyDEPipeline
from .hybrid_rag import HybridRAGPipeline
from .graph_rag import GraphRAGPipeline
from .reranking import RerankingRAGPipeline

ALL_PIPELINES = [
    NaiveRAGPipeline,
    ContextualCompressionPipeline,
    MultiQueryPipeline,
    HyDEPipeline,
    HybridRAGPipeline,
    GraphRAGPipeline,
    RerankingRAGPipeline,
]

PIPELINE_MAP = {p.name: p for p in ALL_PIPELINES}
