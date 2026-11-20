"""Fine-tuning session: the structured state of one fine-tuning conversation.

Chat starts a fresh run for every message, so the Fine-Tuning Assistant is re-grounded from this
record on every turn (runtime chosen, working notebook + version, pending plan, open question,
jobs). Keyed by the Chat conversation id; a run without a conversation (API, schedule) gets a
session of its own.

    finetune_sessions/<session_id>
      runtime   {server_id, runtime_id, name, gpu_type}       chosen by the user
      notebook  {path, base, name, version, hash}             current working notebook
      plan      {plan_id, created_run_id, status, ...}        last proposed run
      question  {text, options, run_id}                       open question to the user
      jobs      [job_id, ...]
      events    [{at, run_id, event}]                         short audit trail (last 60)
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from ..store import make_repo

repo = make_repo("finetune_sessions", "session_id")
_LOCK = threading.RLock()
MAX_EVENTS = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get(session_id: str) -> dict | None:
    return repo.get(session_id) if session_id else None


def by_conversation(conversation_id: str) -> dict | None:
    if not conversation_id:
        return None
    rows = repo.list(lambda d: d.get("conversation_id") == conversation_id)
    rows.sort(key=lambda d: d.get("created_at", ""))
    return rows[-1] if rows else None


def get_or_create(*, conversation_id: str = "", run_id: str = "", agent: str = "",
                  workflow_id: str = "") -> dict:
    with _LOCK:
        doc = by_conversation(conversation_id) if conversation_id else None
        if not doc and not conversation_id and run_id:
            rows = repo.list(lambda d: d.get("origin_run_id") == run_id)
            doc = rows[0] if rows else None
        if doc:
            return doc
        doc = {"session_id": f"ftses_{uuid.uuid4().hex[:12]}", "conversation_id": conversation_id,
               "origin_run_id": run_id, "agent": agent, "workflow_id": workflow_id,
               "runtime": None, "notebook": None, "plan": None, "question": None, "jobs": [],
               "events": [], "created_at": _now(), "updated_at": _now()}
        repo.upsert(doc)
        return doc


def update(session_id: str, run_id: str = "", event: str = "", **fields) -> dict:
    """Read-modify-write under a lock (tools run in worker threads)."""
    with _LOCK:
        doc = repo.get(session_id)
        if not doc:
            raise KeyError(f"fine-tuning session {session_id} not found")
        doc.update(fields)
        if event:
            ev = doc.setdefault("events", [])
            ev.append({"at": _now(), "run_id": run_id, "event": event[:300]})
            del ev[:-MAX_EVENTS]
        doc["updated_at"] = _now()
        repo.upsert(doc)
        return doc


def add_job(session_id: str, job_id: str, run_id: str = "") -> dict:
    with _LOCK:
        doc = repo.get(session_id) or {}
        jobs = list(doc.get("jobs") or [])
        if job_id not in jobs:
            jobs.append(job_id)
        return update(session_id, run_id=run_id, event=f"job {job_id} launched", jobs=jobs)
