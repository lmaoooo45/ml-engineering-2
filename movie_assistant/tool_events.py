import math


def validate_tool_event(message):
    if (set(message) != {"type", "tool", "seconds", "status", "movie_count", "attempt"}
            or message["type"] != "tool_event"
            or message["tool"] not in ("search_movies", "recommend_movies")
            or message["status"] not in ("ok", "clarify", "unsupported", "error")
            or type(message["seconds"]) not in (int, float)
            or not math.isfinite(message["seconds"]) or not 0 <= message["seconds"] <= 240
            or type(message["movie_count"]) is not int or not 0 <= message["movie_count"] <= 10000
            or type(message["attempt"]) is not int or message["attempt"] not in (1, 2)):
        raise ValueError("Недопустимое событие инструмента")
    return {key: value for key, value in message.items() if key != "type"}
