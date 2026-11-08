from movie_assistant.catalog import load_catalog, verify_files
from movie_assistant.contracts import RecommendPlan
from movie_assistant.movie_tools import matching_movies


def save_preferences(memory, root, profile, genres, exclude_titles):
    directory = root / "data/ml-100k"
    verify_files(directory)
    movies = load_catalog(directory)
    fields = RecommendPlan(action="recommend", count=3, genre_mode="and",
                           genres=genres or [], exclude_titles=exclude_titles or [])
    allowed = {genre for movie in movies.values() for genre in movie["genres"]}
    if not set(fields.genres) <= allowed:
        raise ValueError("В списке есть неизвестный жанр")
    for title in fields.exclude_titles:
        if len(matching_movies(movies, title)) != 1:
            raise ValueError(f"Укажи однозначное название фильма: {title}")
    return memory.save_preferences(
        profile,
        fields.genres if genres is not None else None,
        fields.exclude_titles if exclude_titles is not None else None,
    )
