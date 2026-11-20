"""Fine-tuning router — /api/v1/finetune/*.

Conversations run through the normal run API (POST /runs with the Fine-Tuning Assistant
workflow). This router adds what the UI needs around them: settings, the notebook catalog,
runtimes, the per-conversation session (runtime, working notebook + diff, plan), the Launch
action, and jobs (status, cancel).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/finetune", tags=["finetuning"])


@router.get("/config")
def get_config():
    from ..finetune import config
    return {**config.get(), "backend_mlflow_uri": config.backend_mlflow_uri(),
            "mlflow_ui_url_effective": config.mlflow_ui_url()}


@router.put("/config")
def put_config(body: dict):
    from ..finetune import config
    from ..governance import audit
    cfg = config.update(body or {})
    audit("admin", "update", "finetune_settings", detail=",".join((body or {}).keys()))
    return cfg


@router.get("/catalog")
def get_catalog():
    """Every template notebook in the fine-tuning folder, as the assistant sees it."""
    from ..finetune import catalog, config
    items = catalog.list_notebooks()
    return {"folder": config.get()["notebooks_dir"], "count": len(items), "notebooks": items}


@router.get("/runtimes")
def runtimes():
    """DSLab runtimes (secrets removed): which have agent access (a DSLab MCP server), and
    which agents have that server bound."""
    from .. import mcp_store
    from ..finetune import config
    from ..finetune import remote
    from ..modules.data_science_lab import config as dsl_cfg
    from ..modules.data_science_lab import store as dsl_store
    from ..store import agents as agents_repo
    agents = agents_repo.list()
    out = []
    for r in dsl_store.list_runtimes():
        sid = f"{remote.DSLAB_SERVER_PREFIX}{r['id']}"
        srv = mcp_store.get_server(sid)
        out.append({"id": r["id"], "name": r.get("name"), "gpu_type": r.get("gpu_type", ""),
                    "status": r.get("status"), "mlflow_uri": r.get("mlflow_uri", ""),
                    "server_id": sid if srv else "", "agent_access": bool(srv),
                    "server_enabled": bool(srv and srv.get("enabled", True)),
                    "bound_agents": [a["agent_name"] for a in agents
                                     if sid in remote.bound_dslab_servers(a.get("tools") or [])]})
    return {"dsl_enabled": dsl_cfg.is_enabled(),
            "configured_default": config.get().get("default_runtime_id", ""), "runtimes": out}


# ------------------------------------------------------------------ sessions
def _session_view(ses: dict) -> dict:
    from ..finetune import jobs, remote
    from ..finetune import notebook_ops as nbo
    view = dict(ses)
    view["jobs"] = [jobs.brief(d) for d in (jobs.get(j) for j in ses.get("jobs") or []) if d]
    view["jobs"].reverse()
    if ses.get("runtime"):
        view["runtime_card"] = remote.describe_server(ses["runtime"]["server_id"])
    view["drafts"] = nbo.list_drafts(ses["session_id"])
    return view


@router.get("/sessions/by-conversation/{conversation_id}")
def session_by_conversation(conversation_id: str):
    from ..finetune import session as S
    ses = S.by_conversation(conversation_id)
    if not ses:
        raise HTTPException(404, "no fine-tuning session for this conversation yet")
    return _session_view(ses)


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    from ..finetune import session as S
    ses = S.get(session_id)
    if not ses:
        raise HTTPException(404, "session not found")
    return _session_view(ses)


@router.get("/sessions/{session_id}/notebook")
def session_notebook(session_id: str, path: str = "", version: int = 0):
    """Cells of the working notebook (or `path` / an earlier `version`) and its diff vs base."""
    from ..finetune import notebook_ops as nbo
    from ..finetune import session as S
    ses = S.get(session_id)
    if not ses:
        raise HTTPException(404, "session not found")
    cur = ses.get("notebook") or {}
    p = path or cur.get("path") or ""
    if not p:
        raise HTTPException(404, "no working notebook yet")
    try:
        p = nbo.resolve(p, session_id)
        target = nbo.version_path(session_id, p, version) if version else p
        nb = nbo.load(target)
        base = cur.get("base") if cur.get("path") == p else p
        diff = nbo.diff(nbo.load(base), nb, base.rsplit("/", 1)[-1], p.rsplit("/", 1)[-1]) \
            if base and base != target else ""
        return {"path": p, "version": version or (cur.get("version") if cur.get("path") == p else None),
                "base": base, "diff": diff, "validation": nbo.validate(nb),
                **nbo.summarize(nb, max_chars=20000)}
    except nbo.NotebookError as e:
        raise HTTPException(400, str(e))


@router.post("/sessions/{session_id}/plans/{plan_id}/launch")
async def launch(session_id: str, plan_id: str):
    """The Launch button: the user's explicit confirmation of the proposed plan."""
    from ..finetune import jobs
    from ..finetune.tools import launch_plan
    from ..governance import audit
    try:
        doc = await asyncio.to_thread(launch_plan, session_id, plan_id,
                                      launched_by="user (Launch button)")
    except ValueError as e:
        raise HTTPException(409, str(e))
    audit("user", "launch", "finetune_job", detail=f"{doc['job_id']} plan={plan_id}")
    return jobs.brief(doc)


# ------------------------------------------------------------------ jobs
@router.get("/jobs")
def list_jobs(session_id: str = "", limit: int = 50):
    from ..finetune import jobs
    return [jobs.brief(d) for d in jobs.list_jobs(session_id, limit)]


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    from ..finetune import jobs
    d = jobs.get(job_id)
    if not d:
        raise HTTPException(404, "job not found")
    return {"job_id": job_id, "status": d.get("status"), "session_id": d.get("session_id"),
            "conversation_id": d.get("conversation_id"), "runtime": d.get("runtime"),
            "selection": {k: v for k, v in (d.get("selection") or {}).items()
                          if k != "declared_parameters"},
            "plan_id": d.get("plan_id"), "launched_by": d.get("launched_by"),
            "created_at": d.get("created_at"), "ended_at": d.get("ended_at"),
            "progress": d.get("progress") or {}}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    from ..finetune import jobs
    from ..governance import audit
    try:
        res = jobs.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    audit("user", "cancel", "finetune_job", detail=job_id)
    return res


# ------------------------------------------------------------------ legacy (build 37 runs)
@router.get("/runs/{run_id}/progress")
def run_progress(run_id: str):
    from ..store import runs as runs_repo
    r = runs_repo.get(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return {"run_id": run_id, "status": r.get("status"), "error": r.get("error", ""),
            "progress": r.get("progress") or {}}
