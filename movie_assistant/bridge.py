import json
import os
import sys

MAX_MESSAGE_BYTES = 1024 * 1024


def emit_tool_event(name, seconds, status, movie_count, attempt):
    if os.environ.get("MOVIE_OLLAMA_BRIDGE") == "stdio":
        print(json.dumps({"type": "tool_event", "tool": name, "seconds": seconds,
                          "status": status, "movie_count": movie_count, "attempt": attempt}), flush=True)


def rpc(endpoint, payload):
    message = json.dumps({"type": "rpc", "endpoint": endpoint, "payload": payload})
    if len(message.encode()) > MAX_MESSAGE_BYTES:
        raise RuntimeError("Запрос к посреднику слишком большой")
    print(message, flush=True)
    line = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1)
    if not line or len(line) > MAX_MESSAGE_BYTES:
        raise RuntimeError("Посредник закрыл соединение или вернул слишком большой ответ")
    response = json.loads(line)
    if not response["ok"]:
        raise RuntimeError("Посредник отклонил запрос или модель не ответила")
    return response["result"]
