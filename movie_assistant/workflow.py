import json
from time import monotonic
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from movie_assistant.contracts import Plan, RecommendPlan
from movie_assistant.bridge import emit_tool_event
from movie_assistant.memory import open_memory
from movie_assistant.movie_tools import TOOLS, TOOL_FOR_ACTION, run_tool
from movie_assistant.ollama_client import MODELS, chat
from movie_assistant.planner import build_plan, system_prompt
from movie_assistant.routing import choose_route
from movie_assistant.validation import check_catalog_result, check_meaning


class State(TypedDict, total=False):
    question: str
    user_id: int
    plan: dict | None
    tool: str | None
    result: dict
    attempts: int
    feedback: str | None
    validation: dict | None
    history: list[dict]
    preferences: dict
    routing: dict
    validator_rejections: int


def build_graph(root, model=MODELS["gemma"], checkpointer=None, routing=False):
    if model not in {MODELS["gemma"], MODELS["ministral"]}:
        raise ValueError("Основная модель должна отличаться от семейства валидатора Qwen")

    def start_turn(state):
        route = choose_route(state["question"], state.get("history", []), root) if routing else {
            "model": model, "reason": "fixed", "fallback": False,
        }
        return {"attempts": 0, "feedback": None, "plan": None, "routing": route,
                "result": {}, "tool": None, "validation": None, "validator_rejections": 0}

    def coordinate(state):
        attempts = state.get("attempts", 0) + 1
        route = dict(state["routing"])
        if attempts > 1 and route["model"] == MODELS["ministral"]:
            route.update(model=MODELS["gemma"], fallback=True)
        try:
            plan = build_plan(state["question"], route["model"], feedback=state.get("feedback"),
                              history=state.get("history", []), preferences=state.get("preferences", {}))
            preferences = state.get("preferences", {})
            if isinstance(plan.root, RecommendPlan):
                fields = plan.model_dump()
                if not fields["genres"]:
                    fields["genres"] = preferences.get("genres", [])
                    fields["genre_mode"] = "and"
                excluded = fields["exclude_titles"] + preferences.get("exclude_titles", [])
                fields["exclude_titles"] = list(dict.fromkeys(excluded))
                plan = Plan.model_validate(fields)
        except (ValueError, KeyError, RuntimeError) as error:
            return {"attempts": attempts, "tool": None, "validation": None,
                    "routing": route, "feedback": f"Некорректный план: {error}"[:500]}
        return {"plan": plan.model_dump(), "attempts": attempts, "tool": None, "feedback": None,
                "validation": None, "routing": route}

    def retry_or_reject(state):
        return "coordinator" if state["attempts"] < 2 else "rejected"

    def next_step(state):
        if state["feedback"]:
            return retry_or_reject(state)
        if state["plan"]["action"] in TOOL_FOR_ACTION:
            return "specialist"
        return "message"

    def message(state):
        plan = state["plan"]
        return {"result": {"status": plan["action"], "text": plan["message"], "movies": []}}

    def specialist(state):
        messages = [
            {"role": "system", "content": system_prompt("specialist") +
             "\nСделай ровно один вызов инструмента. Скопируй поля плана в аргументы, кроме action."},
            {"role": "user", "content": json.dumps(state["plan"], ensure_ascii=False)},
        ]
        try:
            response = chat(
                state["routing"]["model"], messages, tools=TOOLS, think=False,
                options={"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 256},
            )
        except RuntimeError as error:
            return {"feedback": str(error)}
        calls = response["message"].get("tool_calls", [])
        if len(calls) != 1:
            return {"feedback": "Специалист должен вызвать один инструмент"}
        function = calls[0]["function"]
        expected = TOOL_FOR_ACTION[state["plan"]["action"]]
        expected_args = {key: value for key, value in state["plan"].items() if key != "action"}
        actual_args = json.dumps(function.get("arguments"), sort_keys=True)
        if function["name"] != expected or actual_args != json.dumps(expected_args, sort_keys=True):
            return {"feedback": "Специалист прислал недопустимый вызов инструмента"}
        return {"tool": function["name"]}

    def execute(state):
        started = monotonic()
        result = {"status": "error", "movies": []}
        try:
            plan = Plan.model_validate(state["plan"]).root
            result = run_tool(root, state["tool"], plan, state["user_id"])
        except ValueError as error:
            return {"feedback": str(error)}
        finally:
            emit_tool_event(state["tool"], round(monotonic() - started, 6),
                            result["status"], len(result["movies"]), state["attempts"])
        return {"result": result}

    def validate(state):
        if state["feedback"]:
            return {}
        plan, result = Plan.model_validate(state["plan"]).root, state["result"]
        try:
            check_catalog_result(root / "data/ml-100k", plan, result, state["user_id"])
            verdict = check_meaning(state["question"], plan, result,
                                    history=state.get("history", []), preferences=state.get("preferences"))
        except ValueError as error:
            return {"feedback": str(error)}
        return {
            "feedback": None if verdict.accepted else verdict.reason,
            "validation": {"model": MODELS["qwen"], **verdict.model_dump()},
            "validator_rejections": state.get("validator_rejections", 0) + int(not verdict.accepted),
        }

    def finish_or_retry(state):
        return retry_or_reject(state) if state["feedback"] else "remember"

    def remember(state):
        result = state["result"]
        titles = ", ".join(movie["title"] for movie in result["movies"])
        answer = result["text"]
        if titles and titles != answer:
            answer += " Найденные фильмы: " + titles
        history = state.get("history", []) + [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer[:1500]},
        ]
        return {"history": history[-6:]}

    def rejected(state):
        return {"tool": None, "result": {
            "status": "error", "text": "Не удалось подтвердить ответ. " + state["feedback"], "movies": [],
        }}

    graph = StateGraph(State)
    graph.add_node("coordinator", coordinate)
    graph.add_node("specialist", specialist)
    graph.add_node("tool", execute)
    graph.add_node("message", message)
    graph.add_node("validate", validate)
    graph.add_node("rejected", rejected)
    graph.add_node("start_turn", start_turn)
    graph.add_node("remember", remember)
    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "coordinator")
    graph.add_conditional_edges("coordinator", next_step, ["specialist", "message", "coordinator", "rejected"])
    graph.add_conditional_edges("specialist", lambda state: "validate" if state["feedback"] else "tool",
                                ["validate", "tool"])
    graph.add_edge("tool", "validate")
    graph.add_edge("message", "validate")
    graph.add_conditional_edges("validate", finish_or_retry, ["coordinator", "rejected", "remember"])
    graph.add_edge("remember", END)
    graph.add_edge("rejected", END)
    return graph.compile(checkpointer=checkpointer)


def run_session(root, question, *, profile="default", session_id=None, user_id=None, routing=True):
    if not question.strip() or len(question) > 1000:
        raise ValueError("Вопрос должен содержать от 1 до 1000 символов")
    if user_id is not None and not 0 <= user_id < 2**63:
        raise ValueError("Недопустимый ID пользователя")
    with open_memory(root) as memory:
        session_id, user_id = memory.session(profile, session_id, user_id)
        graph = build_graph(root, checkpointer=memory.checkpointer, routing=routing)
        state = graph.invoke({
            "question": question,
            "user_id": user_id,
            "preferences": memory.preferences(profile),
        }, config={"configurable": {"thread_id": session_id}})
    return session_id, state
