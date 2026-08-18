import json
import re
from wahlcheck_ai.config import GLOSSARY_JSON, THESES_JSON


def load_theses() -> list[dict]:
    with open(THESES_JSON, encoding="utf-8") as f:
        categories = json.load(f)

    theses = []
    for category, items in categories.items():
        slug = re.split(r"[,\s]", category, maxsplit=1)[0].lower()
        for i, text in enumerate(items, start=1):
            theses.append(
                {"id": f"{slug}-{i:02d}", "category": category, "these": text}
            )
    return theses


def load_glossary() -> list[dict]:
    with open(GLOSSARY_JSON, encoding="utf-8") as f:
        items = json.load(f)
    return items
