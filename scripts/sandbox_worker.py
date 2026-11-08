import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys


def probe():
    checks = {"no_windows_drive": not Path("/mnt/c").exists(),
              "no_host_home": not (Path(os.environ["MOVIE_SANDBOX_HOST_HOME"])/'.ssh').exists(),
              "no_host_canary": not Path(os.environ["MOVIE_SANDBOX_CANARY"]).exists(),
              "no_host_mount_via_proc": not Path("/proc/1/root/mnt/c").exists(),
              "no_interop_env": "WSL_INTEROP" not in os.environ, "non_root_uid": os.getuid() != 0}
    caps = next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("CapEff:"))
    checks["no_effective_capabilities"] = int(caps, 16) == 0
    status = Path("/proc/self/status").read_text()
    checks["no_new_privileges"] = "NoNewPrivs:\t1" in status
    nested = subprocess.run(["/usr/bin/unshare", "--user", "--map-root-user", "/bin/true"], capture_output=True)
    checks["nested_user_namespace_denied"] = nested.returncode != 0
    for name, path in (("code_readonly", "/app/SOUL.txt"), ("data_readonly", "/state/data/ml-100k/u.data"),
                       ("model_readonly", "/state/models/two-tower.pt")):
        try:
            with open(path, "r+"):
                pass
            checks[name] = False
        except PermissionError:
            checks[name] = True
        except OSError as error:
            checks[name] = error.errno == 30
    try:
        with socket.create_connection(("192.0.2.1", 443), timeout=1):
            pass
        checks["network_denied"] = False
    except OSError:
        checks["network_denied"] = True
    marker = Path("/state/probe-write.txt")
    marker.write_text("isolated state is writable\n")
    checks["state_writable"] = marker.read_text().startswith("isolated")
    from movie_assistant.bridge import rpc
    try:
        rpc("/api/pull", {"model": "anything"})
        checks["unapproved_bridge_call_denied"] = False
    except RuntimeError:
        checks["unapproved_bridge_call_denied"] = True
    return {"checks": checks, "passed": all(checks.values()), "uid": os.getuid(), "caps": caps,
            "limits": {"address_space_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                       "cpu_seconds": resource.getrlimit(resource.RLIMIT_CPU)[0]}}


def main():
    task = json.loads(sys.stdin.readline())
    if task["mode"] == "probe":
        result = probe()
    elif task["mode"] == "profile":
        from movie_assistant.memory import open_memory
        with open_memory(Path("/state")) as memory:
            body = task["body"]
            if body.get("forget"):
                result = {"removed_sessions": memory.forget(body["profile"])}
            elif body.get("genres") is not None or body.get("exclude_titles") is not None:
                from movie_assistant.profile import save_preferences
                result = save_preferences(memory, Path("/state"), body["profile"],
                                          body.get("genres"), body.get("exclude_titles"))
            else:
                result = memory.preferences(body["profile"])
            if not body.get("forget"):
                result["updated_at"] = memory.preference_updated_at(body["profile"])
    elif task["mode"] == "judge":
        from movie_assistant.validation import check_meaning
        from movie_assistant.contracts import Plan
        body = task["body"]
        verdict = check_meaning(body["question"], Plan.model_validate(body["plan"]).root,
                                body["result"], history=body.get("history", []),
                                preferences=body.get("preferences"))
        result = {"status": "ok", "verdict": verdict.model_dump()}
    elif task["mode"] == "chat":
        from movie_assistant.workflow import run_session
        session_id, state = run_session(Path("/state"), **task["body"], routing=task.get("route_mode", "auto") == "auto")
        result = {"session_id": session_id, "plan": state["plan"], "tool": state.get("tool"),
                  "attempts": state["attempts"], "validation": state.get("validation"),
                  "routing": state["routing"], "validator_rejections": state.get("validator_rejections", 0), **state["result"]}
    else:
        raise ValueError("Unknown worker mode")
    if task["mode"] in ("chat", "judge"):
        from movie_assistant.resources import worker_resources
        result["worker_resources"] = worker_resources()
    print(json.dumps({"type": "result", "result": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"type": "error", "error_type": type(error).__name__,
                          "message": str(error)[:1000]}), flush=True)
        raise SystemExit(1)
