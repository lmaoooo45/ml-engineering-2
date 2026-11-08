import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from statistics import median
from time import monotonic
from uuid import uuid4

from movie_assistant.catalog import load_catalog
from movie_assistant.gateway import isolated_request, sandbox_settings
from movie_assistant.recommendations import load_ratings, temporal_split
from movie_assistant.settings import PROJECT_ROOT, storage_root


def check_result(result, expected, catalog, seen):
    errors=[]
    if result.get("status") != expected["status"]:
        errors.append("status")
    movies=result.get("movies",[])
    ids=[movie["id"] for movie in movies]
    if len(ids)!=len(set(ids)):
        errors.append("duplicate")
    for movie in movies:
        reference=catalog.get(movie["id"])
        if reference is None or any(movie.get(key)!=reference[key] for key in ("title","year","genres")):
            errors.append("catalog")
    if "film_id" in expected and ids != [expected["film_id"]]:
        errors.append("film_id")
    if "year" in expected and (not movies or movies[0].get("year")!=expected["year"]):
        errors.append("year")
    for text in expected.get("text_contains",[]):
        if text not in result.get("text",""):
            errors.append("answer_text")
    if "count" in expected:
        if len(movies)!=expected["count"]:
            errors.append("count")
        if set(ids)&seen:
            errors.append("seen")
        if set(ids)&set(expected.get("excluded_ids",[])):
            errors.append("excluded")
        requested=set(expected.get("genres",[]))
        for movie in movies:
            genres=set(movie["genres"])
            if requested and ((expected["genre_mode"]=="and" and not requested<=genres)
                              or (expected["genre_mode"]=="or" and not requested&genres)):
                errors.append("genres")
        if "fallback" in expected and result.get("fallback")!=expected["fallback"]:
            errors.append("fallback")
    if expected["status"] in ("unsupported","clarify") and movies:
        errors.append("unexpected_movies")
    return sorted(set(errors))


def compact_summary(rows):
    times=[row["seconds"] for row in rows]
    return {"total":len(rows),"passed":sum(row.get("passed",False) for row in rows),
            "request_errors":sum("error" in row for row in rows),
            "median_seconds":median(times) if times else None,
            "p95_seconds_nearest_rank":sorted(times)[math.ceil(.95*len(times))-1] if times else None,
            "llm_calls":sum(row.get("result",{}).get("usage",{}).get("llm_calls",0) for row in rows),
            "validator_rejections":sum(row.get("result",{}).get("validator_rejections",0) for row in rows)}


def main():
    parser=argparse.ArgumentParser(description='Живые проверки системы, валидатора и стоимости маршрутизации.')
    parser.add_argument("--suite",required=True,choices=("dev","holdout","validator","routing"))
    parser.add_argument("--case",help="ID через запятую для проверки исправлений; это не новый независимый набор")
    args=parser.parse_args()
    root=storage_root()
    raw=(PROJECT_ROOT/"benchmarks/final_cases.json").read_bytes()
    cases=json.loads(raw)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path=root/"results"/f"final-{args.suite}-{stamp}.json"
    prefix="eval-"+uuid4().hex[:10]
    snapshot=sandbox_settings()["snapshot_sha256"]
    catalog=load_catalog(root/"data/ml-100k")
    train,_=temporal_split(load_ratings(root/"data/ml-100k"))
    sessions={}
    rows=[]
    report={"suite":args.suite,"created_utc":datetime.now(timezone.utc).isoformat(),
            "cases_sha256":hashlib.sha256(raw).hexdigest(),"snapshot_sha256":snapshot,
            "weights_sha256":hashlib.sha256((root/"models/two-tower.pt").read_bytes()).hexdigest(),
            "note":"Живые локальные учебные опыты. Разметка фиксирована до прогона; исходные ответы сохранены.",
            "selected_cases":args.case,"rows":rows}
    if args.suite in ("dev","holdout"):
        selected=[case for case in cases["system"] if case["split"]==args.suite]
    elif args.suite=="validator":
        selected=cases["validator"]
    else:
        selected=[]
        for index,case in enumerate(cases["routing_comparison"]):
            order=("auto","standard") if index%2==0 else ("standard","auto")
            selected.extend({**case,"route_mode":mode,"id":case["id"]+"-"+mode} for mode in order)
    if args.case:
        requested=set(args.case.split(","))
        selected=[case for case in selected if case["id"] in requested]
        if {case["id"] for case in selected} != requested:
            parser.error("Неизвестный ID сценария")
    for case in selected:
        started=monotonic()
        row={"case":case["id"],"passed":False}
        try:
            if args.suite=="validator":
                result=isolated_request("judge",{key:case[key] for key in ("question","plan","result","preferences","history") if key in case},source="synthetic")
                row["expected_accepted"]=case["accepted"]
                row["predicted_accepted"]=result.get("verdict",{}).get("accepted")
                row["passed"]=row["predicted_accepted"]==case["accepted"]
            else:
                profile=prefix+"-"+case.get("profile_group",case.get("session_group",case["id"]))
                if "preferences" in case:
                    isolated_request("profile",{"profile":profile,"genres":case["preferences"]["genres"],
                                     "exclude_titles":case["preferences"]["exclude_titles"],"forget":False},source="synthetic")
                group=case.get("session_group")
                result=isolated_request("chat",{"question":case["question"],"profile":profile,
                    "session_id":sessions.get(group) if group else None,"user_id":case.get("user_id")},
                    source="synthetic",route_mode=case.get("route_mode","auto"))
                if group and "session_id" in result:
                    sessions[group]=result["session_id"]
                if args.suite=="routing":
                    row["route_mode"]=case["route_mode"]
                    if "комедии" in case["question"]:
                        expected={"status":"ok","count":2,"genres":["Comedy"],"genre_mode":"and"}
                    elif "играет" in case["question"]:
                        expected={"status":"unsupported"}
                    else:
                        
                        title=case["question"].replace("Какого года ","").replace("Назови год выхода ","").strip("?")
                        from movie_assistant.movie_tools import matching_movies
                        movie=matching_movies(catalog,title)[0]
                        expected={"status":"ok","film_id":movie["id"],"year":movie["year"]}
                else:
                    expected=case["expected"]
                seen={mid for uid,mid,_,_ in train if uid==case.get("user_id",0)}
                row["errors"]=check_result(result,expected,catalog,seen)
                row["passed"]=not row["errors"]
            row["result"]=result
        except Exception as error:
            row["error"]={"type":type(error).__name__,"message":str(error)[:1000]}
        row["seconds"]=monotonic()-started
        rows.append(row)
        report["summary"]=compact_summary(rows)
        if args.suite=="validator":
            report["summary"].update({
                "missed_errors":sum(r.get("expected_accepted") is False and r.get("predicted_accepted") is True for r in rows),
                "false_rejections":sum(r.get("expected_accepted") is True and r.get("predicted_accepted") is False for r in rows),
                "missing_verdicts":sum(r.get("predicted_accepted") is None for r in rows)})
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"suite":args.suite,"case":case["id"],"passed":row["passed"],
                          "seconds":round(row["seconds"],2),"errors":row.get("errors",row.get("error"))},ensure_ascii=False),flush=True)
    print("Сохранено:",path,flush=True)
    print(json.dumps(report["summary"],ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
