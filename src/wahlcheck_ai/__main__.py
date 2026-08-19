from wahlcheck_ai import expansion
from wahlcheck_ai.config import DOCUMENTS_DIR
from wahlcheck_ai.embed import build_party_index
from glob import glob
from wahlcheck_ai.prompts import rate
from wahlcheck_ai.retrieve import retrieve_for


def main() -> None:
    theses = expansion.expand_queries("openwebui:GPT-OSS-120B")
    for file in glob("*.pdf", root_dir=DOCUMENTS_DIR):
        filename = DOCUMENTS_DIR / file
        vector_index, bm25_retriever = build_party_index(filename)
        retrievals = retrieve_for(filename, theses, vector_index, bm25_retriever)

        thesis_id = theses[0]["these"]["id"]
        chosen_ones = [
            retrieval
            for retrieval in retrievals[thesis_id]
            if retrieval["rerank_score"] > 0.05
        ]
        #rate.rate(theses[0]["these"]["these"], chosen_ones, "openwebui:GPT-OSS-120B")


if __name__ == "__main__":
    main()
