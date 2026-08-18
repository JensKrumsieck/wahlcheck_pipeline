from llama_index.core import VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from huggingface_hub.constants import HF_HUB_CACHE
from wahlcheck_ai.config import K_PER_VARIANT

TOP_K = 8
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
K_FUSED = 25

_RERANKER: SentenceTransformerRerank | None = None


def get_reranker() -> SentenceTransformerRerank:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = SentenceTransformerRerank(
            model=RERANK_MODEL,
            top_n=8,
            keep_retrieval_score=True,
            cache_folder=HF_HUB_CACHE,
        )
    return _RERANKER


def build_fusion_retriever(
    vector_index: VectorStoreIndex, bm25_retriever: BM25Retriever
) -> QueryFusionRetriever:
    vector_retriever = vector_index.as_retriever(similarity_top_k=K_PER_VARIANT)
    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        llm=MockLLM(),
        mode="reciprocal_rerank",  # type: ignore
        similarity_top_k=K_FUSED,
        num_queries=1,
        use_async=False,
    )


def retrieve_candidates(
    vector_index: VectorStoreIndex,
    bm25_retriever: BM25Retriever,
    variants: dict[str, str],
) -> list[NodeWithScore]:
    """RRF-fuse dense+BM25 per query variant, keep best score per node across variants."""
    fusion = build_fusion_retriever(vector_index, bm25_retriever)
    best: dict[str, NodeWithScore] = {}
    for query_text in variants:
        if not query_text:
            continue
        for result in fusion.retrieve(query_text):
            node_id = result.node.node_id
            if node_id not in best or (result.score or 0) > (best[node_id].score or 0):
                best[node_id] = result
    return sorted(best.values(), key=lambda result: -(result.score or 0))[:K_FUSED]
