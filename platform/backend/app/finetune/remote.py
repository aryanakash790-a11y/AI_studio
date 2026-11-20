"""DSLab runtime access for fine-tuning jobs.

Job control (stage the notebook, start, status, cancel, list outputs) goes through the DSLab
MCP server (app/mcp_servers/dslab_runtime.py) that the Fine-Tuning Assistant has bound in Agent
Garden, via the platform's MCP client — the same server and client agents use. Bulk file transfer (datasets in, model files out) uses
the same authenticated lab proxy directly, because multi-MB payloads do not belong in MCP
JSON-RPC messages. All functions here are blocking; callers run them in a worker thread.
"""
from __future__ import annotations

import base64
import json
import os

from .. import mcp_client, mcp_store
from ..modules.data_science_lab import config as dsl_cfg
from ..modules.data_science_lab import store as dsl_store
from ..modules.data_science_lab.jupyter_remote import LabClient, LabError, LabNotOpen
from . import config


class RemoteError(RuntimeError):
    pass


# ------------------------------------------------------------------ runtime access
# A runtime is reached ONLY through a DSLab MCP server that the calling agent has bound in
# Agent Garden (and that is enabled on the MCP Servers screen). Nothing is registered or picked
# implicitly: the Fine-Tuning Assistant asks the user which of its bound runtimes to use.
DSLAB_SERVER_PREFIX = "mcp_dslab_"


def is_dslab_server(sid: str) -> bool:
    return bool(sid) and str(sid).startswith(DSLAB_SERVER_PREFIX)


def bound_dslab_servers(agent_tools: list[dict]) -> list[str]:
    """Server ids of the DSLab runtime MCP servers bound to an agent, in binding order."""
    out: list[str] = []
    for t in agent_tools or []:
        if t.get("type") != "mcp":
            continue
        sid = t.get("ref") or t.get("server_id") or ""
        if is_dslab_server(sid) and sid not in out:
            out.append(sid)
    return out


def runtime_id_of(server: dict) -> str:
    env = server.get("env") or {}
    rid = env.get("DSLAB_RUNTIME_ID") or ""
    if not rid and is_dslab_server(server.get("id", "")):
        rid = server["id"][len(DSLAB_SERVER_PREFIX):]
    return rid


def describe_server(sid: str) -> dict:
    """Runtime card for a bound DSLab MCP server (no secrets). `usable` False + `problem`
    when the server or runtime cannot be used right now."""
    server = mcp_store.get_server(sid)
    if not server:
        return {"server_id": sid, "usable": False,
                "problem": "MCP server is no longer registered (MCP Servers screen)"}
    rid = runtime_id_of(server)
    rt = dsl_store.get_runtime(rid) if rid else None
    card = {"server_id": sid, "server_name": server.get("name", sid), "runtime_id": rid,
            "name": (rt or {}).get("name") or server.get("name", sid),
            "gpu_type": (rt or {}).get("gpu_type", ""), "status": (rt or {}).get("status", ""),
            "provider": (rt or {}).get("provider", ""), "enabled": bool(server.get("enabled", True)),
            "usable": True, "problem": ""}
    if not dsl_cfg.is_enabled():
        card.update(usable=False, problem="Data Science Lab is disabled")
    elif not server.get("enabled", True):
        card.update(usable=False, problem=f"MCP server '{server.get('name')}' is disabled")
    elif not rt:
        card.update(usable=False, problem=f"DSLab runtime '{rid}' no longer exists")
    return card


def server_for(sid: str) -> tuple[dict, dict]:
    """(server, runtime) for a bound DSLab MCP server id. Raises RemoteError with an
    actionable message when it cannot be used."""
    card = describe_server(sid)
    if not card["usable"]:
        raise RemoteError(f"runtime unavailable: {card['problem']}")
    server = mcp_store.get_server(sid)
    rt = dsl_store.get_runtime(card["runtime_id"])
    return server, rt


def ensure_lab_open(rt: dict) -> str:
    """Open the runtime's lab (SSH tunnel + Jupyter) from the backend if it is not open.

    Lab tunnels live in the backend process, so after every backend restart or redeploy all labs
    are closed. The DSLab MCP server can re-open a lab itself, but only inside one MCP tool call,
    and an SSH connect + Jupyter start can take longer than that call is allowed. Doing it here,
    first, avoids the timeout and gives the exact reason when it fails. Returns "" when the lab
    is open, else the error."""
    from ..modules.data_science_lab import session as dsl_session
    try:
        if dsl_session.get_tunnel_port(rt["id"]) is not None:
            return ""
        r = dsl_session.open_lab(rt["id"])
    except Exception as e:
        return f"opening the DSLab lab for '{rt.get('name', rt['id'])}' failed: {type(e).__name__}: {e}"
    if r.get("ok"):
        return ""
    steps = "; ".join(f"{s.get('step')}: {s.get('detail') or ('ok' if s.get('ok') else 'failed')}"
                      for s in (r.get("steps") or []) if isinstance(s, dict))
    return (f"the DSLab lab for '{rt.get('name', rt['id'])}' could not be opened: "
            f"{r.get('detail') or 'unknown error'}" + (f" ({steps})" if steps else ""))


def is_error_text(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("ERROR:") or t.startswith('{"error"')


def runtime_mlflow_uri(rt: dict) -> str:
    return (config.get().get("remote_mlflow_uri") or rt.get("mlflow_uri")
            or config.backend_mlflow_uri() or "")


# ------------------------------------------------------------------ MCP job control
def call(server: dict, tool: str, args: dict) -> dict:
    text = mcp_client.call_tool(server, tool, args) or ""
    t = text.strip()
    if t.startswith("ERROR:"):
        raise RemoteError(t[6:].strip())
    try:
        out = json.loads(t)
    except Exception:
        raise RemoteError(f"{tool}: unexpected reply: {t[:500]}")
    if isinstance(out, dict) and out.get("error") and not out.get("ok", False) and "state" not in out:
        raise RemoteError(f"{tool}: {out['error']}")
    return out


def stage_text(server: dict, job_id: str, name: str, content: str) -> None:
    call(server, "stage_job_file", {"job_id": job_id, "name": name, "content": content,
                                    "encoding": "text"})


def start(server: dict, job_id: str, env: dict, timeout_hours: float) -> dict:
    c = config.get()
    res = call(server, "start_notebook_job", {
        "job_id": job_id, "notebook": "notebook.ipynb", "env": env,
        "timeout_hours": float(timeout_hours),
        "log_outputs_to_mlflow": bool(c["log_outputs_to_mlflow"]),
        "mlflow_max_gb": float(c["mlflow_max_gb"])})
    if not res.get("ok"):
        raise RemoteError(f"job did not start: {res.get('error') or res}")
    return res


def status(server: dict, job_id: str) -> dict:
    return call(server, "notebook_job_status", {"job_id": job_id})


def cancel(server: dict, job_id: str) -> dict:
    return call(server, "cancel_notebook_job", {"job_id": job_id})


def outputs(server: dict, job_id: str) -> dict:
    return call(server, "list_job_outputs", {"job_id": job_id})


# ------------------------------------------------------------------ bulk transfer (lab proxy)
def _lab_url(server: dict, rt: dict) -> str:
    env = server.get("env") or {}
    if env.get("DSLAB_LAB_URL"):
        return env["DSLAB_LAB_URL"]
    api = (env.get("DSLAB_API_BASE") or os.getenv("DSLAB_API_BASE")
           or f"http://127.0.0.1:{os.getenv('BACKEND_PORT', '8099')}").rstrip("/")
    return f"{api}/api/v1/dsl/runtimes/{rt['id']}/lab/"


def lab(server: dict, rt: dict, fn):
    """Run fn(LabClient); if the lab is not open, open it once and retry."""
    client = LabClient(_lab_url(server, rt))
    try:
        return fn(client)
    except LabNotOpen:
        from ..modules.data_science_lab import session as dsl_session
        r = dsl_session.open_lab(rt["id"])
        if not r.get("ok"):
            raise RemoteError(f"the DSLab lab is not open and could not be opened: {r.get('detail')}")
        return fn(client)
    except LabError as e:
        raise RemoteError(str(e))


def jobs_dir(server: dict) -> str:
    return ((server.get("env") or {}).get("DSLAB_JOBS_DIR") or "studio-jobs").strip("/")


def upload_bytes(server: dict, rt: dict, job_id: str, rel: str, data: bytes) -> str:
    target = f"{jobs_dir(server)}/{job_id}/{rel}"

    def go(c: LabClient):
        c.mkdirs(target.rsplit("/", 1)[0])
        c.put_base64(target, base64.b64encode(data).decode())
        return target
    return lab(server, rt, go)


def download(server: dict, rt: dict, job_id: str, rel: str) -> bytes:
    target = f"{jobs_dir(server)}/{job_id}/{rel}"
    return lab(server, rt, lambda c: c.read_bytes(target))
