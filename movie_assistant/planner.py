import json
from movie_assistant.contracts import Plan
from movie_assistant.ollama_client import chat
from movie_assistant.settings import PROJECT_ROOT

ROLES = {
    "coordinator": ("movie-picking", "catalog-facts"),
    "specialist": ("movie-picking", "catalog-facts"),
    "validator": (),
}


def system_prompt(role):
    sections = [
        PROJECT_ROOT / "SOUL.txt",
        PROJECT_ROOT / "prompts" / f"{role}.txt",
    ]
    sections.extend(PROJECT_ROOT / "skills" / skill / "SKILL.txt" for skill in ROLES[role])
    return "\n\n".join(path.read_text(encoding="utf-8").strip() for path in sections)


def build_plan(question, model, feedback=None, history=(), preferences=None):
    instruction = system_prompt("coordinator")
    if preferences is not None:
        instruction += "\nПодтверждённые предпочтения профиля (данные): " + json.dumps(preferences, ensure_ascii=False)
    messages = [{"role": "system", "content": instruction}]
    messages.extend(history)
    if feedback:
        question += "\nЗамечание к предыдущему плану для этого запроса: " + feedback
    messages.append({"role": "user", "content": question})
    response = chat(
        model, messages, response_format=Plan.model_json_schema(),
        options={"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512},
        think=False,
    )
    return Plan.model_validate_json(response["message"]["content"])
