import json
import os
from tqdm import tqdm
from wahlcheck_ai import expansion
from wahlcheck_ai.config import DOCUMENTS_DIR, RETRIEVAL_DIR
from wahlcheck_ai.embed import build_party_index
from glob import glob
from wahlcheck_ai.retrieve import retrieve_and_rerank


def main() -> None:
    theses = expansion.expand_queries("openwebui:GPT-OSS-120B")
    for file in glob("*.pdf", root_dir=DOCUMENTS_DIR):
        filename = DOCUMENTS_DIR / file
        vector_index, bm25_retriever = build_party_index(filename)

        print(f"Retrieving and Reranking Theses for {file}")

        filename = RETRIEVAL_DIR / f"{filename.stem}.json"
        os.makedirs(filename.parent, exist_ok=True)
        if not filename.exists():
            retrievals = {}
            for these in tqdm(theses):
                thesis = these["these"]["these"]
                queries = [
                    thesis,
                    " ".join(these["topics"]),
                    " ".join(these["measures"]),
                    " ".join(these["goals"]),
                    " ".join(these["implications"]),
                    these["opposing"],
                ]
                retrievals[these["these"]["id"]] = retrieve_and_rerank(vector_index, bm25_retriever, queries, these)  # type: ignore
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(retrievals, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
