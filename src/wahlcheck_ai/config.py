import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
DOCUMENTS_DIR = ROOT / "documents"
BUILD_DIR = ROOT / "build"

EXTRACTED_DIR = BUILD_DIR / "extract"
INDEX_DIR = BUILD_DIR / "index"
RETRIEVAL_DIR = BUILD_DIR / "retrieval"
RATING_DIR = BUILD_DIR / "rating"
ANTWORTEN_DIR = BUILD_DIR / "antworten"

INPUT_DIR = ROOT / "input"
THESES_JSON = INPUT_DIR / "fragen.json"
GLOSSARY_JSON = INPUT_DIR / "glossar.json"
ANTWORT_SCHEMA_JSON = INPUT_DIR / "antwort.schema.json"

OPENWEBUI_BASE_URL = os.environ.get("OPENWEBUI_BASE_URL")
OPENWEBUI_API_KEY = os.environ.get("OPENWEBUI_API_KEY")

K_PER_VARIANT = 25
