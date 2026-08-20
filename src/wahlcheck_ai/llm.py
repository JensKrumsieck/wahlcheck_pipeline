import json

from openai import OpenAI
from ollama import chat as _ollama_chat
from wahlcheck_ai.config import OPENWEBUI_API_KEY, OPENWEBUI_BASE_URL

OPENWEBUI_PREFIX = "openwebui:"
_OPENWEBUI_CLIENT: OpenAI | None = None


def _openwebui_client() -> OpenAI:
    global _OPENWEBUI_CLIENT
    if _OPENWEBUI_CLIENT is None:
        if not OPENWEBUI_BASE_URL or not OPENWEBUI_API_KEY:
            raise RuntimeError(
                "OPENWEBUI_BASE_URL / OPENWEBUI_API_KEY not set - add them to .env "
                "(see .env.example) to use a `openwebui:<model>` model name."
            )
        _OPENWEBUI_CLIENT = OpenAI(
            base_url=OPENWEBUI_BASE_URL, api_key=OPENWEBUI_API_KEY
        )
    return _OPENWEBUI_CLIENT


def _parse_json(content: str) -> dict:
    """Strips a markdown code fence if the model wrapped its JSON in one -
    Ollama's format= never does this, but not every OpenWebUI backend honors
    response_format as strictly."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    return json.loads(text)


def chat_json(
    model: str,
    system_prompts: list[str],
    user_prompt: str,
    schema: dict,
    num_ctx: int = 8192*2,
) -> dict:
    messages = [
        {"role": "system", "content": system_prompt} for system_prompt in system_prompts
    ]
    messages.append({"role": "user", "content": user_prompt})

    if model.startswith(OPENWEBUI_PREFIX):
        real_model = model.removeprefix(OPENWEBUI_PREFIX)
        response = _openwebui_client().chat.completions.create(
            model=real_model,
            messages=messages,  # type: ignore
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            },
        )
        return _parse_json(response.choices[0].message.content)  # type: ignore

    response = _ollama_chat(
        model=model,
        messages=messages,
        format=schema,
        think=False,
        options={"num_ctx": num_ctx},
    )
    return _parse_json(response.message.content)  # type: ignore
