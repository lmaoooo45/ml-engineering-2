import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

LOCAL_API = "http://127.0.0.1:11434"
LOCAL_HTTP = build_opener(ProxyHandler({}))
MODELS = {
    "gemma": "gemma4-e2b-local:q4_k_m",
    "ministral": "ministral3-3b-local:q4_k_m",
    "qwen": "qwen3.5-2b-local:q4_k_m",
}


def request_json(path, payload=None, timeout=60):
    if os.environ.get("MOVIE_OLLAMA_BRIDGE") == "stdio":
        from movie_assistant.bridge import rpc
        return rpc(path, payload)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(LOCAL_API + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with LOCAL_HTTP.open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"Нет ответа от локального Ollama: {error}") from error


def chat(model, messages, *, options, response_format=None, tools=None, keep_alive=0, think=None):
    payload = {
        "model": model,
        "messages": messages,
        "options": options,
        "stream": False,
        "keep_alive": keep_alive,
    }
    if response_format is not None:
        payload["format"] = response_format
    if tools is not None:
        payload["tools"] = tools
    if think is not None:
        payload["think"] = think
    return request_json("/api/chat", payload)
