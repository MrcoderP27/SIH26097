import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def load_env_file():
    """
    Simple .env loader using only Python standard library.

    This avoids requiring python-dotenv.
    """

    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"

    if not env_file.exists():
        return

    with open(env_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip()

            if (
                len(value) >= 2
                and value.startswith('"')
                and value.endswith('"')
            ):
                value = value[1:-1]

            if (
                len(value) >= 2
                and value.startswith("'")
                and value.endswith("'")
            ):
                value = value[1:-1]

            os.environ.setdefault(key, value)


load_env_file()


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)


def call_ollama(messages):
    """
    Call Ollama when a JSON response is required.

    Used for structured tasks such as:
    - beneficiary profile extraction
    - conversation analysis
    - structured AI output
    """

    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urlopen(request, timeout=120) as response:

        response_data = response.read().decode("utf-8")

    data = json.loads(response_data)

    if "message" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    content = data["message"].get("content", "")

    if not content:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    try:
        return json.loads(content)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            f"Ollama did not return valid JSON: {content}"
        ) from exc


def call_ollama_text(messages):
    """
    Call Ollama when a normal text response is required.

    Unlike call_ollama(), this function does NOT force
    JSON output.

    Used for:
    - natural Hindi responses
    - follow-up questions
    - conversational assistant responses
    """

    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9
        }
    }

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urlopen(request, timeout=120) as response:

        response_data = response.read().decode("utf-8")

    data = json.loads(response_data)

    if "message" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    content = data["message"].get("content", "")

    if not content:
        raise RuntimeError(
            "Ollama returned an empty text response."
        )

    return content.strip()