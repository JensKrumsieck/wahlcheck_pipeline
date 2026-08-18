import json

from wahlcheck_ai import expansion
from wahlcheck_ai.config import DOCUMENTS_DIR
from llama_index.core.schema import NodeWithScore
from wahlcheck_ai.embed import build_party_index
from glob import glob
from pprint import pprint
from wahlcheck_ai.retrieve import TOP_K, get_reranker, retrieve_candidates


def main() -> None:
    theses = expansion.expand_queries("openwebui:GPT-OSS-120B")
    for file in glob("*.pdf", root_dir=DOCUMENTS_DIR):
        filename = DOCUMENTS_DIR / file
        vector_index, bm25_retriever = build_party_index(filename)
        for these in theses:
            thesis = these["these"]["these"]
            queries = [
                thesis,
                " ".join(these["topics"]),
                " ".join(these["measures"]),
                " ".join(these["goals"]),
                " ".join(these["implications"]),
                these["opposing"],
            ]
        nodes = retrieve_candidates(vector_index, bm25_retriever, queries)  # type: ignore
        reranker = get_reranker()
        reranker.top_n = TOP_K
        retrievals = []
        for these in theses:
            thesis = these["these"]["these"]
            top = reranker.postprocess_nodes(
                nodes,
                query_str=thesis,  # type: ignore
            )
            retrievals.append([_slim_window(result) for result in top])
            
        with open("test.json", "w", encoding="utf-8") as f:
            json.dump(retrievals, f, ensure_ascii=False)


def _slim_window(result: NodeWithScore) -> dict:
    metadata = result.node.metadata
    return {
        "id": result.node.node_id,
        "metadata": metadata,
        "fused_score": metadata.get("retrieval_score"),
        "rerank_score": result.score,
    }


if __name__ == "__main__":
    main()
