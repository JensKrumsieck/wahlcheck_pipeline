from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from huggingface_hub.constants import HF_HUB_CACHE
from llama_index.core.schema import TextNode
from wahlcheck_ai.config import INDEX_DIR, K_PER_VARIANT
from wahlcheck_ai.extract import extract
import Stemmer

_EMBED_MODEL: HuggingFaceEmbedding | None = None
_EMBEDDINGS_MODEL: str = "BAAI/bge-m3"
_METADATA_KEYS = ["pages", "heading_path", "sentence_ids", "tokens", "text"]


def get_embed_model() -> HuggingFaceEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=_EMBEDDINGS_MODEL, cache_folder=HF_HUB_CACHE
        )
    return _EMBED_MODEL


def build_party_index(filename: Path, force: bool = False):
    party = filename.stem
    party_dir = INDEX_DIR / party

    if party_dir.exists() and not force:
        vector = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(party_dir / "vector")),
            embed_model=get_embed_model(),
        )
        bm25 = BM25Retriever.from_persist_dir(str(party_dir / "bm25"))
        return vector, bm25

    doc = extract(filename, force)
    nodes = _build_nodes(doc.blocks)
    vector_index = VectorStoreIndex(
        nodes, embed_model=get_embed_model(), show_progress=True
    )
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,  # type: ignore
        stemmer=Stemmer.Stemmer("german"),
        language="german",
        similarity_top_k=K_PER_VARIANT,
    )

    party_dir.mkdir(parents=True, exist_ok=True)
    vector_index.storage_context.persist(persist_dir=str(party_dir / "vector"))
    bm25_retriever.persist(str(party_dir / "bm25"))

    return vector_index, bm25_retriever


def _build_nodes(windows: list) -> list[TextNode]:
    return [
        TextNode(
            id_=str(window["id"]),
            text=window["embedding_text"],
            metadata={
                "pages": window["pages"],
                "sentence_ids": window["sentence_ids"],
                "headings": window["headings"],
                "text": window["text"],
            },
            excluded_embed_metadata_keys=_METADATA_KEYS,
            excluded_llm_metadata_keys=_METADATA_KEYS,
        )
        for window in windows
    ]
