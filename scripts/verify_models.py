import json
from movie_assistant.ollama_client import MODELS, request_json
from movie_assistant.routing import DECISION_MODEL
from movie_assistant.settings import storage_root


def main():
    tags={model["name"]:model for model in request_json("/api/tags")["models"]}
    rows=[]
    for name in list(MODELS.values())+[DECISION_MODEL]:
        metadata=request_json("/api/show",{"model":name})
        rows.append({"name":name,"digest":tags[name]["digest"],"size_bytes":tags[name]["size"],
                     "details":metadata["details"],
                     "model_info":{key:value for key,value in metadata.get("model_info",{}).items()
                                   if key in ("general.architecture","general.parameter_count") or key.endswith(".context_length")}})
    families={row["name"]:row["details"]["family"] for row in rows}
    if families[MODELS["qwen"]] in {families[MODELS["gemma"]],families[MODELS["ministral"]]}:
        raise ValueError("Валидатор должен принадлежать другому семейству")
    report={"ollama":request_json("/api/version")["version"],"models":rows,"different_validator_family":True}
    output=storage_root()/"results/model-manifest.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Семейство валидатора отличается от обоих основных:",families)


if __name__=="__main__":
    main()
