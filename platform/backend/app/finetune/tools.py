"""The Fine-Tuning toolset (agent tool binding {"type": "finetune"}).

Bound to an agent in Agent Garden like File access. It gives a conversational agent everything it
needs to run fine-tuning interactively, while the platform keeps control of what matters:

  * runtimes   only DSLab MCP servers bound to THIS agent and enabled are offered; the user
               chooses one (ft_ask_user), and only that runtime's MCP tools are exposed after.
               Raw job-control MCP tools are hidden: jobs start only through a confirmed plan.
  * notebooks  read / create / edit (structured cell ops, copy-on-write drafts + versions) /
               validate / diff / save as template.
  * plan       ft_propose_run validates everything and shows the user a plan with a Launch
               button. ft_submit_job refuses to launch in the same turn the plan was proposed, so
               a launch always follows an explicit user action.
  * jobs       status / cancel / list; the job itself runs in the job service (jobs.py).

Every tool reads the calling run/agent/conversation from tool_exec.tool_context().
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from .. import filestore
from . import catalog, config, jobs, notebook_ops as nbo, remote, session as S

# DSLab MCP tools that are NOT exposed to a fine-tuning agent: job control goes through the
# confirmed-plan path, and run_notebook would execute a whole notebook unconfirmed.
HIDDEN_RUNTIME_TOOLS = {"stage_job_file", "start_notebook_job", "notebook_job_status",
                        "cancel_notebook_job", "list_job_outputs", "run_notebook"}
RUNTIME_TOOLS = ["run_python", "run_shell", "runtime_info", "list_files", "read_file",
                 "write_file", "add_cell", "restart_kernel"]

_RUN_UI: dict[str, dict] = {}          # run_id -> {"jobs": [...]} touched during the step


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_MAX_RESULT = 24000


def _fit(obj: dict) -> str:
    """JSON result for the model; oversized results stay valid JSON (truncated text)."""
    s = json.dumps(obj, default=str)
    if len(s) <= _MAX_RESULT:
        return s
    return json.dumps({"ok": obj.get("ok", True), "truncated": True,
                       "note": "result too long - narrow it (e.g. read fewer cells with start/end)",
                       "text": s[:_MAX_RESULT - 400]})


def _ok(**kw) -> str:
    return _fit({"ok": True, **kw})


def _err(msg: str, **kw) -> str:
    return _fit({"ok": False, "error": msg, **kw})


# ------------------------------------------------------------------ context
def _ctx() -> dict:
    from ..tool_exec import tool_context
    return tool_context()


def _session(create: bool = True) -> dict | None:
    c = _ctx()
    if not create:
        if c.get("conversation_id"):
            return S.by_conversation(c["conversation_id"])
        return None
    return S.get_or_create(conversation_id=c.get("conversation_id", ""), run_id=c.get("run_id", ""),
                           agent=c.get("agent", ""), workflow_id=c.get("workflow_id", ""))


def _touch(job_id: str) -> None:
    rid = _ctx().get("run_id")
    if rid:
        ui = _RUN_UI.setdefault(rid, {"jobs": []})
        if job_id not in ui["jobs"]:
            ui["jobs"].append(job_id)


def _bound(agent_tools: list[dict] | None = None) -> list[str]:
    return remote.bound_dslab_servers(agent_tools if agent_tools is not None
                                      else _ctx().get("agent_tools") or [])


def _selected_server(agent_tools: list[dict]) -> str:
    """The session's runtime, if it is still bound to the agent."""
    c = _ctx()
    if not (c.get("conversation_id") or c.get("run_id")):
        return ""
    ses = S.by_conversation(c["conversation_id"]) if c.get("conversation_id") else None
    sid = ((ses or {}).get("runtime") or {}).get("server_id", "")
    return sid if sid in _bound(agent_tools) else ""


# ------------------------------------------------------------------ tool definitions
def _fn(name: str, desc: str, props: dict | None = None, required: list | None = None) -> dict:
    sch: dict = {"type": "object", "properties": props or {}}
    if required:
        sch["required"] = required
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": sch}}


_OPS_SCHEMA = {
    "type": "array", "description": (
        "Edit operations applied in order. Each item has 'op' plus fields: "
        "replace_cell {index, source, cell_type?}; insert_cell {index (-1 = append), source, "
        "cell_type='code', tags?}; delete_cell {index}; replace_text {find, replace, index?}; "
        "set_parameters {values: {name: value}}; set_tags {index, tags}. Indices refer to the "
        "notebook as it is when that operation runs."),
    "items": {"type": "object"}}

TOOL_DEFS = [
    _fn("ft_session_status", "Current fine-tuning session: chosen runtime, working notebook and "
        "version, pending plan, recent jobs. Call when unsure of the state."),
    _fn("ft_ask_user", "Ask the user a question and END YOUR TURN. Options are shown as clickable "
        "choices (e.g. runtimes, notebooks). Use for every decision that is the user's to make.",
        {"question": {"type": "string"},
         "options": {"type": "array", "items": {"type": "string"},
                     "description": "2-8 short choices; omit for a free-text answer"}},
        ["question"]),
    _fn("ft_list_runtimes", "DSLab runtimes this agent may use (its bound, enabled DSLab MCP "
        "servers), with GPU and status, and which one is selected."),
    _fn("ft_select_runtime", "Record the runtime the USER chose (by name or id). Only after the "
        "user picked it. Its code tools (run_python, runtime_info, ...) are available from the "
        "next turn.", {"runtime": {"type": "string"}}, ["runtime"]),
    _fn("ft_list_notebooks", "Template notebooks (File Storage › finetuning/notebooks) with "
        "description and parameters, plus this conversation's drafts."),
    _fn("ft_read_notebook", "Read a notebook's cells with their indices, tags and parameters.",
        {"path": {"type": "string", "description": "File Storage path, template name or draft name"},
         "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path"]),
    _fn("ft_create_notebook", "Create a new notebook draft from cells (the first markdown cell "
        "should describe it; put tunable values in one code cell tagged 'parameters').",
        {"name": {"type": "string"},
         "cells": {"type": "array", "items": {"type": "object"},
                   "description": "[{cell_type: code|markdown, source, tags?}]"}},
        ["name", "cells"]),
    _fn("ft_edit_notebook", "Edit a notebook with structured cell operations. Templates are never "
        "changed: the first edit creates a draft; each edit saves a new version. Returns the diff "
        "and validation result.", {"path": {"type": "string"}, "ops": _OPS_SCHEMA},
        ["path", "ops"]),
    _fn("ft_validate_notebook", "Check a notebook: structure, Python syntax of every code cell, "
        "parameters cell.", {"path": {"type": "string"}}, ["path"]),
    _fn("ft_diff_notebook", "Cell-level diff of a draft against its base template, or against an "
        "earlier version.", {"path": {"type": "string"},
                             "against_version": {"type": "integer",
                                                 "description": "omit to compare with the base"}},
        ["path"]),
    _fn("ft_save_as_template", "Save a draft into the template folder for reuse. Only when the "
        "user asks.", {"path": {"type": "string"}, "name": {"type": "string"},
                       "overwrite": {"type": "boolean"}}, ["path", "name"]),
    _fn("ft_list_datasets", "Datasets in File Storage (finetuning/datasets and shared/)."),
    _fn("ft_propose_run", "Validate and propose a run: selected runtime + notebook + parameters "
        "(+ dataset). Shows the user a plan card with a Launch button. You cannot launch it in the "
        "same turn; the user must confirm.",
        {"notebook": {"type": "string"},
         "parameters": {"type": "object", "description": "values for the notebook's parameters"},
         "dataset": {"type": "string", "description": "File Storage path or Hugging Face dataset id"},
         "summary": {"type": "string", "description": "one-line description of what this run does"}},
        ["notebook", "summary"]),
    _fn("ft_submit_job", "Launch a proposed plan. ONLY after the user explicitly confirmed it in "
        "a later message (e.g. 'yes, launch').", {"plan_id": {"type": "string"}}, ["plan_id"]),
    _fn("ft_job_status", "Live status of a job: stage, % done, cell, loss/eval loss/epoch, "
        "errors, MLflow link. Defaults to the latest job.", {"job_id": {"type": "string"}}),
    _fn("ft_cancel_job", "Stop a running job (outputs collected so far are kept). Only when the "
        "user asks.", {"job_id": {"type": "string"}}, ["job_id"]),
    _fn("ft_list_jobs", "Jobs launched in this conversation, newest first."),
]
TOOL_NAMES = [d["function"]["name"] for d in TOOL_DEFS]


def extend(defs: list, registry: dict, agent_tools: list[dict]) -> tuple[list, dict]:
    """Add the ft_* tools and scope DSLab runtime MCP tools to the session's runtime."""
    bound = set(_bound(agent_tools))
    selected = _selected_server(agent_tools)
    keep_defs = []
    for d in defs:
        name = d["function"]["name"]
        entry = registry.get(name)
        if entry and entry[0] == "mcp" and entry[1].get("server_id") in bound:
            if entry[1]["server_id"] != selected or entry[1]["tool"] in HIDDEN_RUNTIME_TOOLS:
                registry.pop(name, None)
                continue
        keep_defs.append(d)
    for d in TOOL_DEFS:
        keep_defs.append(d)
        registry[d["function"]["name"]] = ("finetune", {}, None)
    return keep_defs, registry


def scope_bindings(agent_tools: list[dict]) -> tuple[list[dict], dict]:
    """For ADK native MCP toolsets: drop unselected runtimes, filter the selected one."""
    bound = set(_bound(agent_tools))
    selected = _selected_server(agent_tools)
    out = [t for t in agent_tools or []
           if not (t.get("type") == "mcp" and (t.get("ref") or t.get("server_id")) in bound
                   and (t.get("ref") or t.get("server_id")) != selected)]
    return out, ({selected: list(RUNTIME_TOOLS)} if selected else {})


# ------------------------------------------------------------------ prompt + UI
def prompt_block(agent_tools: list[dict]) -> str:
    """Session state injected into every turn so the agent never loses track."""
    ses = _session()
    bound = [remote.describe_server(s) for s in _bound(agent_tools)]
    rt = ses.get("runtime") or {}
    nb = ses.get("notebook") or {}
    plan = ses.get("plan") or {}
    recent = [jobs.brief(d) for d in (jobs.get(j) for j in (ses.get("jobs") or [])[-3:]) if d]
    state = {
        "runtimes_available_to_you": [{"name": b["name"], "gpu": b["gpu_type"], "usable": b["usable"],
                                       **({"problem": b["problem"]} if b["problem"] else {})}
                                      for b in bound] or "NONE - no DSLab runtime MCP server is bound to you",
        "selected_runtime": rt.get("name") or None,
        "working_notebook": ({"path": nb.get("path"), "version": nb.get("version"),
                              "based_on": nb.get("base")} if nb else None),
        "pending_plan": ({"plan_id": plan.get("plan_id"), "status": plan.get("status"),
                          "notebook": (plan.get("notebook") or {}).get("name"),
                          "version": (plan.get("notebook") or {}).get("version"),
                          "parameters": (plan.get("selection") or {}).get("parameters"),
                          "proposed_this_turn": plan.get("created_run_id") == _ctx().get("run_id")}
                         if plan and plan.get("status") == "proposed" else None),
        "recent_jobs": [{k: v for k, v in r.items() if v not in (None, "", [])} for r in recent],
        "open_question": (ses.get("question") or {}).get("text"),
    }
    return ("\n=== FINE-TUNING SESSION STATE (authoritative; refreshed every turn) ===\n"
            + json.dumps(state, default=str, indent=1)[:5000]
            + "\n=== END SESSION STATE ===\n")


def after_step(run, agent_tools: list[dict]) -> None:
    """Attach the session's UI state to the run: Chat / Fine-Tuning Studio render it."""
    ses = _session(create=False) or (S.get_or_create(run_id=run.run_id) if not
                                     (run.input or {}).get("conversation_id") else None)
    if not ses:
        return
    ses = S.get(ses["session_id"]) or ses
    ui = _RUN_UI.pop(run.run_id, {"jobs": []})
    q = ses.get("question") or {}
    plan = ses.get("plan") or {}
    job_ids = list(ui.get("jobs") or [])
    for j in reversed(ses.get("jobs") or []):
        d = jobs.get(j)
        if d and d.get("status") in jobs.ACTIVE and j not in job_ids:
            job_ids.append(j)
    run.progress = {
        "kind": "ft_session", "session_id": ses["session_id"],
        "conversation_id": ses.get("conversation_id", ""),
        "question": q if q.get("run_id") == run.run_id else None,
        "plan": plan if plan.get("created_run_id") == run.run_id and plan.get("status") == "proposed" else None,
        "jobs": job_ids[:4],
        "runtime": ses.get("runtime"), "notebook": ses.get("notebook"),
    }


# ------------------------------------------------------------------ execution
def execute(name: str, args: dict) -> str:
    fn = _HANDLERS.get(name)
    if not fn:
        return _err(f"unknown fine-tuning tool {name}")
    try:
        return fn(args or {})
    except nbo.NotebookError as e:
        return _err(str(e))
    except remote.RemoteError as e:
        return _err(str(e))
    except filestore.StorageError as e:
        return _err(f"File Storage: {e}")
    except Exception as e:
        return _err(f"{name} failed: {type(e).__name__}: {e}")


def t_status(a):
    ses = _session()
    return _ok(session_id=ses["session_id"], runtime=ses.get("runtime"), notebook=ses.get("notebook"),
               plan=ses.get("plan"), jobs=[jobs.brief(d) for d in
                                           (jobs.get(j) for j in (ses.get("jobs") or [])[-5:]) if d])


def t_ask(a):
    ses = _session()
    q = str(a.get("question") or "").strip()
    if not q:
        return _err("question is required")
    opts = [str(o)[:120] for o in (a.get("options") or []) if str(o).strip()][:8]
    S.update(ses["session_id"], run_id=_ctx().get("run_id", ""), event=f"asked: {q[:120]}",
             question={"text": q[:600], "options": opts, "run_id": _ctx().get("run_id", ""),
                       "asked_at": _now()})
    return _ok(shown_to_user=True, note="The question is shown with clickable options. Write ONE "
               "short sentence and end your turn now; the user's choice arrives as the next message.")


def t_runtimes(a):
    ses = _session()
    cards = [remote.describe_server(s) for s in _bound()]
    sel = (ses.get("runtime") or {}).get("server_id")
    for c in cards:
        c["selected"] = c["server_id"] == sel
    if not cards:
        return _err("No DSLab runtime is bound to this agent. Ask an admin to: (1) Data Science Lab › "
                    "runtime › Enable agent access, then (2) Agent Garden › edit this agent › Tools › "
                    "MCP servers › tick 'DSLab · <runtime>'.", runtimes=[])
    default = config.get().get("default_runtime_id", "")
    return _ok(runtimes=cards, suggested=next((c["name"] for c in cards if c["runtime_id"] == default), None))


def t_select_runtime(a):
    ses = _session()
    want = str(a.get("runtime") or "").strip().lower()
    for sid in _bound():
        c = remote.describe_server(sid)
        if want in (sid.lower(), c["runtime_id"].lower(), c["name"].lower(), c["server_name"].lower()):
            if not c["usable"]:
                return _err(f"runtime '{c['name']}' cannot be used: {c['problem']}")
            rt = {k: c[k] for k in ("server_id", "runtime_id", "name", "gpu_type")}
            S.update(ses["session_id"], run_id=_ctx().get("run_id", ""), event=f"runtime {c['name']}",
                     runtime=rt, question=None)
            return _ok(selected=rt, note="Runtime recorded. Its code tools are available from the next turn.")
    return _err(f"'{a.get('runtime')}' is not one of your runtimes",
                runtimes=[remote.describe_server(s)["name"] for s in _bound()])


def t_list_notebooks(a):
    ses = _session()
    tmpl = []
    for e in catalog.list_notebooks():
        tmpl.append({"path": e["path"], "name": e["name"], "title": e.get("title"),
                     "description": (e.get("description") or "")[:300], "valid": e["valid"],
                     "metadata": e.get("metadata"), "warnings": e.get("warnings"),
                     "parameters": [{"name": p["name"], "default": p["default"]}
                                    for p in e.get("parameters") or []][:30]})
    return _ok(templates=tmpl, drafts=nbo.list_drafts(ses["session_id"]),
               working_notebook=ses.get("notebook"))


def t_read(a):
    ses = _session()
    path = nbo.resolve(a.get("path", ""), ses["session_id"])
    nb = nbo.load(path)
    per_cell = max(800, min(nbo.MAX_CELL_CHARS, 18000 // max(1, len(nb.get("cells", [])))))
    return _ok(path=path, is_draft=nbo.is_draft(path, ses["session_id"]),
               **nbo.summarize(nb, a.get("start") or 0, a.get("end"), max_chars=per_cell))


def _remember_nb(ses: dict, rec: dict, event: str) -> None:
    S.update(ses["session_id"], run_id=_ctx().get("run_id", ""), event=event, notebook=rec)


def t_create(a):
    ses = _session()
    cells = []
    for c in a.get("cells") or []:
        if not isinstance(c, dict):
            continue
        cells.append(nbo.new_cell(str(c.get("source", "")), c.get("cell_type") or "code", c.get("tags")))
    if not cells:
        return _err("cells must be a non-empty list of {cell_type, source, tags?}")
    nb = nbo.new_notebook(cells)
    v = nbo.validate(nb)
    rec = nbo.save_draft(ses["session_id"], nb, name=a.get("name") or "notebook")
    _remember_nb(ses, rec, f"created {rec['name']} v{rec['version']}")
    return _ok(notebook=rec, validation=v)


def t_edit(a):
    ses = _session()
    sid = ses["session_id"]
    src_path = nbo.resolve(a.get("path", ""), sid)
    nb = nbo.load(src_path)
    new, notes = nbo.apply_ops(nb, a.get("ops"))
    cur = ses.get("notebook") or {}
    if nbo.is_draft(src_path, sid):
        base = cur.get("base") if cur.get("path") == src_path else src_path
    else:
        base = src_path                       # first edit of a template: copy-on-write
    rec = nbo.save_draft(sid, new, name=src_path, base=base)
    v = nbo.validate(new)
    _remember_nb(ses, rec, f"edited {rec['name']} -> v{rec['version']}: " + "; ".join(notes)[:150])
    return _ok(notebook=rec, changes=notes, validation=v,
               diff=nbo.diff(nb, new, src_path.rsplit("/", 1)[-1], f"{rec['name']} v{rec['version']}"),
               note=None if nbo.is_draft(src_path, sid) else
               "The template was not changed; your edits are in a draft for this conversation.")


def t_validate(a):
    ses = _session()
    path = nbo.resolve(a.get("path", ""), ses["session_id"])
    return _ok(path=path, **nbo.validate(nbo.load(path)))


def t_diff(a):
    ses = _session()
    sid = ses["session_id"]
    path = nbo.resolve(a.get("path", ""), sid)
    cur = nbo.load(path)
    if a.get("against_version"):
        other_path = nbo.version_path(sid, path, int(a["against_version"]))
    else:
        nbrec = ses.get("notebook") or {}
        other_path = nbrec.get("base") if nbrec.get("path") == path else path
    other = nbo.load(other_path)
    d = nbo.diff(other, cur, other_path.rsplit("/", 1)[-1], path.rsplit("/", 1)[-1])
    return _ok(path=path, against=other_path, diff=d or "(no differences)")


def t_save_template(a):
    ses = _session()
    path = nbo.resolve(a.get("path", ""), ses["session_id"])
    target = nbo.save_as_template(path, a.get("name") or "", bool(a.get("overwrite")))
    S.update(ses["session_id"], run_id=_ctx().get("run_id", ""), event=f"saved template {target}")
    return _ok(template=target)


def t_datasets(a):
    from .planner import dataset_candidates
    return _ok(datasets=dataset_candidates())


def t_propose(a):
    from .planner import finalize
    ses = _session()
    sid = ses["session_id"]
    rt = ses.get("runtime") or {}
    if not rt.get("server_id"):
        return _err("No runtime selected yet. Call ft_list_runtimes and ask the user to choose one.")
    if rt["server_id"] not in _bound():
        return _err(f"runtime '{rt.get('name')}' is no longer bound to this agent - ask the user to "
                    f"choose again (ft_list_runtimes)")
    card = remote.describe_server(rt["server_id"])
    if not card["usable"]:
        return _err(f"runtime '{rt.get('name')}' cannot be used: {card['problem']}")
    path = nbo.resolve(a.get("notebook", ""), sid)
    nb = nbo.load(path)
    val = nbo.validate(nb)
    if not val["ok"]:
        return _err("the notebook does not validate - fix it with ft_edit_notebook first",
                    validation=val)
    st = filestore.stat(path)
    entry = catalog._entry(st)
    if not entry.get("valid"):
        return _err(f"notebook cannot be read: {entry.get('warnings')}")
    params = a.get("parameters") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params) if params.strip() else {}
        except Exception:
            return _err("parameters must be a JSON object")
    req = {"text": "", "notebook": path, "dataset": str(a.get("dataset") or "").strip(),
           "parameters": params if isinstance(params, dict) else {}, "runtime_id": rt["runtime_id"]}
    summary = str(a.get("summary") or "").strip()[:500]
    sel = finalize(entry, "assistant", req, None, 1.0, summary, [])
    nbrec = ses.get("notebook") or {}
    version = nbrec.get("version") if nbrec.get("path") == path else None
    sel["version"] = version
    plan = {"plan_id": f"ftplan_{uuid.uuid4().hex[:10]}", "created_run_id": _ctx().get("run_id", ""),
            "created_at": _now(), "status": "proposed", "summary": summary,
            "runtime": {k: rt.get(k) for k in ("server_id", "runtime_id", "name", "gpu_type")},
            "notebook": {"path": path, "name": path.rsplit("/", 1)[-1], "version": version,
                         "hash": nbo.nb_hash(nb), "is_draft": nbo.is_draft(path, sid)},
            "selection": sel, "validation": val,
            "warnings": sel.get("warnings") or []}
    S.update(sid, run_id=plan["created_run_id"], event=f"proposed {plan['plan_id']}", plan=plan,
             question=None)
    return _ok(plan_id=plan["plan_id"], runtime=plan["runtime"]["name"], notebook=plan["notebook"],
               parameters=sel["parameters"], dataset=sel.get("dataset"), warnings=plan["warnings"],
               note="Plan shown to the user with a Launch button. Summarise it in 2-4 lines and ask "
                    "them to click Launch or reply 'launch'. Do NOT call ft_submit_job this turn.")


def launch_plan(session_id: str, plan_id: str, *, run_id: str = "", launched_by: str = "user") -> dict:
    """Shared by ft_submit_job and the Launch button. Raises ValueError when refused."""
    ses = S.get(session_id)
    if not ses:
        raise ValueError("session not found")
    plan = ses.get("plan") or {}
    if plan.get("plan_id") != plan_id:
        raise ValueError(f"plan {plan_id} is not the current plan "
                         f"(current: {plan.get('plan_id') or 'none'}) - propose it again")
    if plan.get("status") == "launched" and plan.get("job_id"):
        return jobs.get(plan["job_id"]) or {"job_id": plan["job_id"]}      # idempotent
    if plan.get("status") != "proposed":
        raise ValueError(f"plan {plan_id} is {plan.get('status')}")
    if run_id and plan.get("created_run_id") == run_id:
        raise ValueError("the user has not confirmed this plan yet - it was proposed in this same "
                         "turn. Ask them to click Launch or reply to confirm.")
    path = plan["selection"]["path"]
    try:
        current = nbo.nb_hash(nbo.load(path))
    except nbo.NotebookError as e:
        raise ValueError(f"the notebook can no longer be read: {e}")
    if current != plan["notebook"]["hash"]:
        S.update(session_id, run_id=run_id, event=f"plan {plan_id} stale",
                 plan={**plan, "status": "stale"})
        raise ValueError("the notebook changed after the plan was proposed - propose the run again")
    card = remote.describe_server(plan["runtime"]["server_id"])
    if not card["usable"]:
        raise ValueError(f"runtime '{plan['runtime'].get('name')}' cannot be used: {card['problem']}")
    doc = jobs.submit(session=ses, plan=plan, agent=ses.get("agent", ""),
                      workflow_id=ses.get("workflow_id", ""), launched_by=launched_by)
    S.update(session_id, run_id=run_id, event=f"launched {plan_id} as {doc['job_id']}",
             plan={**plan, "status": "launched", "job_id": doc["job_id"], "launched_at": _now()},
             question=None)
    S.add_job(session_id, doc["job_id"], run_id)
    return doc


def t_submit(a):
    ses = _session()
    try:
        doc = launch_plan(ses["session_id"], str(a.get("plan_id") or ""),
                          run_id=_ctx().get("run_id", ""), launched_by="assistant (user confirmed in chat)")
    except ValueError as e:
        return _err(str(e))
    _touch(doc["job_id"])
    return _ok(job=jobs.brief(doc), note="Job queued; progress is shown to the user live.")


def _job_for(a) -> dict | None:
    ses = _session()
    jid = str(a.get("job_id") or "").strip()
    if not jid:
        ids = ses.get("jobs") or []
        jid = ids[-1] if ids else ""
    if not jid:
        return None
    d = jobs.get(jid)
    return d if d and d.get("session_id") == ses["session_id"] else None


def t_job_status(a):
    d = _job_for(a)
    if not d:
        return _err("no such job in this conversation")
    _touch(d["job_id"])
    b = jobs.brief(d)
    p = d.get("progress") or {}
    b["log_tail"] = (p.get("log_tail") or [])[-8:]
    b["loss_history"] = ((p.get("train") or {}).get("loss_history") or [])[-20:]
    if d.get("status") not in jobs.ACTIVE:
        b["summary"] = p.get("summary", "")[:3000]
    return _ok(job=b)


def t_cancel(a):
    d = _job_for(a)
    if not d:
        return _err("no such job in this conversation")
    _touch(d["job_id"])
    return _ok(**jobs.cancel(d["job_id"]))


def t_list_jobs(a):
    ses = _session()
    return _ok(jobs=[jobs.brief(d) for d in jobs.list_jobs(ses["session_id"], limit=20)])


_HANDLERS = {
    "ft_session_status": t_status, "ft_ask_user": t_ask, "ft_list_runtimes": t_runtimes,
    "ft_select_runtime": t_select_runtime, "ft_list_notebooks": t_list_notebooks,
    "ft_read_notebook": t_read, "ft_create_notebook": t_create, "ft_edit_notebook": t_edit,
    "ft_validate_notebook": t_validate, "ft_diff_notebook": t_diff,
    "ft_save_as_template": t_save_template, "ft_list_datasets": t_datasets,
    "ft_propose_run": t_propose, "ft_submit_job": t_submit, "ft_job_status": t_job_status,
    "ft_cancel_job": t_cancel, "ft_list_jobs": t_list_jobs,
}
