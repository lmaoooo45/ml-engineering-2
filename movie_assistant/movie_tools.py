import re

from movie_assistant.catalog import load_catalog, verify_files
from movie_assistant.contracts import LookupPlan, RecommendPlan
from movie_assistant.recommendations import MovieRecommender, load_ratings, temporal_split

TOOL_FOR_ACTION = {"lookup": "search_movies", "recommend": "recommend_movies"}


def tool_definition(name, description, contract):
    parameters = contract.model_json_schema()
    del parameters["properties"]["action"]
    parameters["required"].remove("action")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOLS = [
    tool_definition("search_movies", "Для lookup ищет фильм по названию из title.", LookupPlan),
    tool_definition("recommend_movies", "Для recommend подбирает фильмы по ограничениям.", RecommendPlan),
]


def matching_movies(movies, title):
    query = title.strip().casefold()
    matches = []
    for movie in movies.values():
        full_title = movie["title"].casefold()
        short_title = re.sub(r"\s*\(\d{4}\)$", "", full_title)
        if query in {full_title, short_title}:
            matches.append(movie)
    return matches


def describe_movie(movie):
    year = movie["year"] if movie["year"] is not None else "не указан"
    genres = ", ".join(movie["genres"])
    return (
        f'{movie["title"]}. Год: {year}. '
        f'Жанры: {genres}.'
    )


def run_tool(root, name, plan, user_id):
    if TOOL_FOR_ACTION.get(plan.action) != name:
        raise ValueError("Инструмент не соответствует действию в плане")
    directory = root / "data/ml-100k"
    verify_files(directory)
    movies = load_catalog(directory)
    if name == "search_movies":
        matches = matching_movies(movies, plan.title)
        if not matches:
            return {"status": "clarify", "text": "Фильм не найден. Уточни название.", "movies": []}
        if len(matches) > 1:
            return {"status": "clarify", "text": "Нашлось несколько фильмов. Уточни год.", "movies": matches}
        movie = matches[0]
        return {"status": "ok", "text": describe_movie(movie), "movies": matches}

    excluded = set()
    for title in plan.exclude_titles:
        matches = matching_movies(movies, title)
        if len(matches) != 1:
            return {"status": "clarify", "text": f"Уточни название исключённого фильма: {title}", "movies": matches}
        excluded.add(matches[0]["id"])

    from movie_assistant.two_tower import TwoTower

    train, _ = temporal_split(load_ratings(directory))
    baseline = MovieRecommender(movies, train)
    model = TwoTower(baseline, train)
    model.load(root / "models/two-tower.pt")
    scores, fallback = model.scores(user_id)
    films = baseline.rank(
        user_id, scores, plan.count, genres=plan.genres,
        genre_mode=plan.genre_mode, exclude_ids=excluded,
    )
    return {
        "status": "ok",
        "text": f"Подходящих фильмов в выдаче: {len(films)} из {plan.count}.\n" +
                "\n".join(f"{index}. {describe_movie(movie)}" for index, movie in enumerate(films, 1)),
        "movies": films,
        "method": "popular" if fallback else "two-tower",
        "fallback": fallback,
    }
