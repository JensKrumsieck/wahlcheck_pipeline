from wahlcheck_ai import expansion
from wahlcheck_ai.config import DOCUMENTS_DIR
from wahlcheck_ai.embed import build_party_index
from glob import glob
from wahlcheck_ai.output import write_antworten
from wahlcheck_ai.rate import rating
from wahlcheck_ai.retrieve import retrieve_for


def main() -> None:
    print(r"""
 _    _       _     _      _               _    
| |  | |     | |   | |    | |             | |   
| |  | | __ _| |__ | | ___| |__   ___  ___| | __
| |/\| |/ _` | '_ \| |/ __| '_ \ / _ \/ __| |/ /
\  /\  / (_| | | | | | (__| | | |  __/ (__|   < 
 \/  \/ \__,_|_| |_|_|\___|_| |_|\___|\___|_|\_\
                                                
                                                """)
    model = "openwebui:GPT-OSS-120B"
    theses = expansion.expand_queries(model)
    for file in glob("*.pdf", root_dir=DOCUMENTS_DIR):
        filename = DOCUMENTS_DIR / file
        vector_index, bm25_retriever = build_party_index(filename)
        retrievals = retrieve_for(filename, theses, vector_index, bm25_retriever)
        rating(filename, theses, retrievals, model)
        write_antworten(filename, theses, model)


if __name__ == "__main__":

    main()
