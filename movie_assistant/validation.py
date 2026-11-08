import json

from pydantic import Field, ValidationError

from movie_assistant.catalog import load_catalog
from movie_assistant.contracts import PlanFields, RecommendPlan
from movie_assistant.movie_tools import matching_movies
from movie_assistant.ollama_client import MODELS, chat
from movie_assistant.planner import system_prompt
from movie_assistant.recommendations import load_ratings, temporal_split


class Verdict(PlanFields):
    accepted: bool
    reason: str = Field(min_length=1, max_length=400)


def check_catalog_result(directory, plan, result, user_id):
    if result["status"] in {"unsupported", "clarify"} and not result["movies"]:
        return
    catalog = load_catalog(directory)
    seen_ids = set()
    for movie in result["movies"]:
        movie_id = movie["id"]
        reference = catalog.get(movie_id)
        if movie_id in seen_ids or reference is None:
            raise ValueError("В выдаче есть повторный или неизвестный ID фильма")
        if any(movie[key] != reference[key] for key in ("title", "year", "genres")):
            raise ValueError("Сведения о фильме отличаются от каталога")
        seen_ids.add(movie_id)

    if not isinstance(plan, RecommendPlan) or result["status"] != "ok":
        return
    if len(seen_ids) > plan.count:
        raise ValueError("В выдаче больше фильмов, чем задано в плане")

    train, _ = temporal_split(load_ratings(directory))
    available = {movie_id for _, movie_id, _, _ in train}
    excluded = {movie_id for uid, movie_id, _, _ in train if uid == user_id}
    for title in plan.exclude_titles:
        excluded.update(movie["id"] for movie in matching_movies(catalog, title))
    if not seen_ids <= available or seen_ids & excluded:
        raise ValueError("В выдаче есть недоступный, уже оценённый или исключённый фильм")

    requested = set(plan.genres)
    for movie in result["movies"]:
        genres = set(movie["genres"])
        if requested and plan.genre_mode == "and" and not requested <= genres:
            raise ValueError("В выдаче нарушено условие «и» для жанров")
        if requested and plan.genre_mode == "or" and not requested & genres:
            raise ValueError("В выдаче нарушено условие «или» для жанров")


def check_meaning(question, plan, result, history=(), preferences=None):
    evidence = {
        "question": question,
        "plan": plan.model_dump(),
        "result": result,
        "history": list(history),
        "preferences": preferences or {},
    }
    response = chat(
        MODELS["qwen"],
        [{"role": "system", "content": system_prompt("validator")},
         {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}],
        response_format=Verdict.model_json_schema(), think=False,
        options={"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 512},
    )
    try:
        return Verdict.model_validate_json(response["message"]["content"])
    except (KeyError, ValidationError) as error:
        raise ValueError("Валидатор вернул некорректное решение") from error
