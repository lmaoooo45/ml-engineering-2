from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from filelock import Timeout
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from movie_assistant.gateway import isolated_request, sandbox_settings
from movie_assistant.ollama_client import MODELS, request_json
from movie_assistant.routing import DECISION_MODEL
from movie_assistant.metrics import read_summary, prometheus_text
from movie_assistant.settings import storage_root


class ChatRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    profile: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    session_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    user_id: int | None = Field(default=None, ge=0, lt=2**63)


def create_app(transport=isolated_request):
    app = FastAPI(title="Киноассистент")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    def summary():
        try:
            return read_summary(storage_root())
        except (OSError, Timeout) as error:
            raise HTTPException(503, "Журнал временно недоступен") from error

    @app.get("/monitor")
    def monitor():
        return summary()

    @app.get("/metrics", response_class=Response)
    def metrics():
        return Response(prometheus_text(summary()),
                        media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/ready")
    def ready():
        try:
            sandbox_settings()
            installed = {model["name"] for model in request_json("/api/tags")["models"]}
            if not (set(MODELS.values()) | {DECISION_MODEL}) <= installed:
                raise ValueError("Установлены не все модели")
            if not transport("probe")["passed"]:
                raise ValueError("Проверка песочницы не пройдена")
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            raise HTTPException(503, str(error)[:300]) from error
        return {"status": "ready"}

    @app.post("/chat")
    def chat(body: ChatRequest):
        try:
            result = transport("chat", body.model_dump())
        except TimeoutError as error:
            raise HTTPException(504, "Истёк срок выполнения запроса") from error
        except ValueError as error:
            raise HTTPException(400, str(error)[:300]) from error
        except (OSError, RuntimeError, KeyError) as error:
            raise HTTPException(503, str(error)[:300]) from error
        if result["status"] == "error":
            raise HTTPException(503, result["text"])
        return result

    return app


app = create_app()
