"""Tests for the Fine-Tuning Assistant (v2): notebook operations, conversation history, tool
scoping to the agent's bound DSLab MCP servers, the plan -> confirm -> launch gate, and a full
job through the job service against a simulated runtime that really executes the notebook with
the platform's remote runner.

Run from platform/backend:   STORE_BACKEND=json python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("STORE_BACKEND", "json")
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app import filestore, mcp_client, mcp_store                       # noqa: E402
from app.finetune import jobs, notebook_ops as nbo, remote, seed, session as S, tools  # noqa: E402
from app.runtime.conversation import format_user_input, wants_full_history  # noqa: E402
from app.tool_exec import build_tool_defs, execute_tool, set_tool_context  # noqa: E402

SAMPLES = BACKEND / "app" / "finetune" / "samples"
RUNNER = BACKEND / "app" / "finetune" / "remote_runner.py"


def _nb(name):
    return json.loads((SAMPLES / name).read_text())


# ------------------------------------------------------------------ notebook operations
def test_apply_ops_edits_cells_and_parameters_without_touching_the_input():
    nb = _nb("text_classification_finetune.ipynb")
    before = json.dumps(nb)
    pidx = next(i for i, c in enumerate(nb["cells"]) if "parameters" in c["metadata"].get("tags", []))
    new, notes = nbo.apply_ops(nb, [
        {"op": "set_parameters", "values": {"num_train_epochs": 5, "brand_new": "x"}},
        {"op": "insert_cell", "index": -1, "source": "print('done')"},
        {"op": "replace_cell", "index": 0, "source": "# Title\nchanged", "cell_type": "markdown"},
    ])
    assert json.dumps(nb) == before                                   # copy, not in place
    psrc = "".join(new["cells"][pidx]["source"])
    assert "num_train_epochs = 5" in psrc and "brand_new = 'x'" in psrc
    assert new["cells"][-1]["source"] == "print('done')"
    assert new["cells"][0]["source"].startswith("# Title")
    assert any("num_train_epochs" in n for n in notes)
    assert nbo.validate(new)["ok"]
    d = nbo.diff(nb, new)
    assert "+num_train_epochs = 5" in d.replace(" ", "").replace("+num_train_epochs=5", "+num_train_epochs = 5")


def test_replace_text_and_bad_ops_are_reported():
    nb = nbo.new_notebook([nbo.new_cell("x = 1\ny = 2")])
    new, _ = nbo.apply_ops(nb, [{"op": "replace_text", "find": "y = 2", "replace": "y = 3"}])
    assert new["cells"][0]["source"] == "x = 1\ny = 3"
    with pytest.raises(nbo.NotebookError):
        nbo.apply_ops(nb, [{"op": "replace_text", "find": "nope", "replace": "z"}])
    with pytest.raises(nbo.NotebookError):
        nbo.apply_ops(nb, [{"op": "delete_cell", "index": 7}])
    with pytest.raises(nbo.NotebookError):
        nbo.apply_ops(nb, [{"op": "explode"}])


def test_validate_catches_syntax_errors_but_accepts_magics():
    ok = nbo.new_notebook([nbo.new_cell("%pip install peft\n!nvidia-smi\nimport os\nos?"),
                           nbo.new_cell("%%bash\necho hi")])
    assert nbo.validate(ok)["ok"]
    bad = nbo.new_notebook([nbo.new_cell("def f(:\n  pass")])
    v = nbo.validate(bad)
    assert not v["ok"] and v["errors"][0]["cell"] == 0


def test_samples_validate():
    for f in SAMPLES.glob("*.ipynb"):
        assert nbo.validate(_nb(f.name))["ok"], f.name


# ------------------------------------------------------------------ conversation history
def test_full_history_formatting():
    txt = format_user_input({"message": "use VM2", "history": [
        {"role": "user", "text": "fine-tune bert"}, {"role": "assistant", "text": "Which runtime?"}]})
    assert "User: fine-tune bert" in txt and "Assistant: Which runtime?" in txt
    assert txt.rstrip().endswith("use VM2")
    assert wants_full_history({"config": {"chat_history": "full"}})
    assert not wants_full_history({"config": {}})


# ------------------------------------------------------------------ fixtures: bound runtimes
SID_A, SID_B = "mcp_dslab_testA", "mcp_dslab_testB"
CARDS = {SID_A: {"name": "VM-A", "gpu_type": "T4"}, SID_B: {"name": "VM-B", "gpu_type": "A100"}}
MCP_TOOLS = [{"name": n, "description": n, "schema": {"type": "object", "properties": {}}}
             for n in ("run_python", "runtime_info", "start_notebook_job", "run_notebook")]


@pytest.fixture()
def bound(monkeypatch):
    seed._seed_files()
    servers = {sid: {"id": sid, "name": f"DSLab · {c['name']}", "enabled": True,
                     "env": {"DSLAB_RUNTIME_ID": sid[len("mcp_dslab_"):]}} for sid, c in CARDS.items()}
    real_get = mcp_store.get_server
    monkeypatch.setattr(mcp_store, "get_server", lambda sid: servers.get(sid) or real_get(sid))
    monkeypatch.setattr(mcp_client, "list_tools", lambda server: MCP_TOOLS)

    def describe(sid):
        c = CARDS.get(sid)
        if not c:
            return {"server_id": sid, "usable": False, "problem": "unknown"}
        return {"server_id": sid, "server_name": f"DSLab · {c['name']}", "runtime_id": sid[10:],
                "name": c["name"], "gpu_type": c["gpu_type"], "status": "ready", "provider": "ssh",
                "enabled": True, "usable": True, "problem": ""}
    monkeypatch.setattr(remote, "describe_server", describe)
    agent_tools = [{"type": "finetune"}, {"type": "mcp", "ref": SID_A}, {"type": "mcp", "ref": SID_B}]
    conv = f"conv_{uuid.uuid4().hex[:8]}"
    return agent_tools, conv


def _turn(agent_tools, conv):
    rid = f"run_{uuid.uuid4().hex[:8]}"
    set_tool_context(run_id=rid, workflow_id="wf_test", agent="Fine-Tuning Assistant",
                     node_id="n1", conversation_id=conv, agent_tools=agent_tools)
    return rid


def _call(name, **args):
    out = json.loads(tools.execute(name, args))
    return out


def test_runtime_tools_are_scoped_to_the_selected_bound_runtime(bound):
    agent_tools, conv = bound
    _turn(agent_tools, conv)
    defs, reg = build_tool_defs(agent_tools)
    names = [d["function"]["name"] for d in defs]
    assert "ft_propose_run" in names
    assert not any(n.startswith("mcp_") for n in names)              # nothing until the user picks

    rts = _call("ft_list_runtimes")
    assert {r["name"] for r in rts["runtimes"]} == {"VM-A", "VM-B"}
    assert _call("ft_select_runtime", runtime="nope")["ok"] is False
    assert _call("ft_select_runtime", runtime="vm-b")["ok"]

    _turn(agent_tools, conv)                                          # next turn
    defs, reg = build_tool_defs(agent_tools)
    mcp = {reg[n][1]["tool"]: reg[n][1]["server_id"] for n in reg if reg[n][0] == "mcp"}
    assert set(mcp) == {"run_python", "runtime_info"}                 # job-control tools hidden
    assert set(mcp.values()) == {SID_B}                               # only the chosen runtime
    out, filters = tools.scope_bindings(agent_tools)
    assert [t.get("ref") for t in out if t.get("type") == "mcp"] == [SID_B]
    assert "start_notebook_job" not in filters[SID_B]


def test_unbound_agent_gets_admin_instructions(bound):
    _, conv = bound
    _turn([{"type": "finetune"}], conv)
    r = _call("ft_list_runtimes")
    assert r["ok"] is False and "Enable agent access" in r["error"]


def test_ask_user_question_reaches_the_run_ui(bound):
    agent_tools, conv = bound
    rid = _turn(agent_tools, conv)
    _call("ft_ask_user", question="Which runtime?", options=["VM-A · T4", "VM-B · A100"])

    class R:
        run_id = rid
        input = {"conversation_id": conv}
        progress: dict = {}
    r = R()
    tools.after_step(r, agent_tools)
    assert r.progress["kind"] == "ft_session"
    assert r.progress["question"]["options"] == ["VM-A · T4", "VM-B · A100"]


def test_edit_is_copy_on_write_and_plan_needs_a_new_turn(bound):
    agent_tools, conv = bound
    _turn(agent_tools, conv)
    _call("ft_select_runtime", runtime="VM-A")
    nbs = _call("ft_list_notebooks")
    tpl = next(t for t in nbs["templates"] if t["name"] == "smoke_test_cpu.ipynb")
    original = filestore.read_bytes(tpl["path"])

    e = _call("ft_edit_notebook", path="smoke_test_cpu", ops=[
        {"op": "set_parameters", "values": {"epochs": 2}}])
    assert e["ok"] and e["notebook"]["version"] == 1 and "/workspace/" in e["notebook"]["path"]
    assert filestore.read_bytes(tpl["path"]) == original              # template untouched
    e2 = _call("ft_edit_notebook", path=e["notebook"]["path"], ops=[
        {"op": "insert_cell", "index": -1, "source": "print('extra')"}])
    assert e2["notebook"]["version"] == 2 and e2["notebook"]["base"] == tpl["path"]
    assert "print('extra')" in _call("ft_diff_notebook", path=e2["notebook"]["path"])["diff"]

    p = _call("ft_propose_run", notebook=e2["notebook"]["path"], parameters={"sleep_per_epoch": 0},
              summary="smoke test")
    assert p["ok"] and p["parameters"]["sleep_per_epoch"] == 0.0
    refused = _call("ft_submit_job", plan_id=p["plan_id"])            # same turn: refused
    assert refused["ok"] is False and "not confirmed" in refused["error"]

    _turn(agent_tools, conv)                                          # user edits again ...
    _call("ft_edit_notebook", path=e2["notebook"]["path"], ops=[
        {"op": "replace_text", "find": "print('extra')", "replace": "print('changed')"}])
    _turn(agent_tools, conv)                                          # ... then says "launch"
    stale = _call("ft_submit_job", plan_id=p["plan_id"])
    assert stale["ok"] is False and "changed" in stale["error"]


# ------------------------------------------------------------------ job service end to end
class FakeRuntime:
    """Simulated DSLab runtime: a local job directory per job; the notebook is executed by the
    platform's real remote runner, exactly as start_notebook_job does on a DSLab machine."""

    def __init__(self, root: Path):
        self.root = root
        self.procs: dict[str, subprocess.Popen] = {}

    def jd(self, rid):
        d = self.root / rid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def install(self, mp):
        mp.setattr(remote, "server_for", lambda sid: ({"id": sid, "env": {}},
                                                      {"id": "testA", "name": "VM-A", "gpu_type": "T4"}))
        mp.setattr(remote, "runtime_mlflow_uri", lambda rt: "")
        mp.setattr(remote, "stage_text", lambda srv, rid, name, content: (self.jd(rid) / name).write_text(content))
        mp.setattr(remote, "start", self.start)
        mp.setattr(remote, "status", lambda srv, rid: json.loads((self.jd(rid) / "status.json").read_text())
                   if (self.jd(rid) / "status.json").exists() else {"state": "starting"})
        mp.setattr(remote, "cancel", self.cancel)
        mp.setattr(remote, "outputs", self.outputs)
        mp.setattr(remote, "download", lambda srv, rt, rid, rel: (self.jd(rid) / rel).read_bytes())

    def start(self, srv, rid, env, timeout_hours):
        e = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}
        e.update(env)
        p = subprocess.Popen([sys.executable, str(RUNNER), "--timeout-hours", "0.05"], cwd=self.jd(rid),
                             env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        self.procs[rid] = p
        return {"ok": True, "pid": p.pid, "job_dir": str(self.jd(rid)), "python": sys.executable}

    def cancel(self, srv, rid):
        p = self.procs.get(rid)
        if p and p.poll() is None:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(30)
        return {"ok": True}

    def outputs(self, srv, rid):
        d = self.jd(rid)
        man = json.loads((d / "outputs_manifest.json").read_text()) if (d / "outputs_manifest.json").exists() else {}
        return {"manifest": man, "job_files": [{"name": f.name, "type": "file", "size": f.stat().st_size}
                                               for f in d.iterdir() if f.is_file()]}


async def _wait(job_id, timeout=150):
    end = time.time() + timeout
    while time.time() < end:
        d = jobs.get(job_id)
        if d and d.get("status") not in jobs.ACTIVE:
            return d
        await asyncio.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish: {jobs.get(job_id)}")


def test_confirmed_plan_runs_a_real_notebook_job(bound, tmp_path, monkeypatch):
    agent_tools, conv = bound
    FakeRuntime(tmp_path).install(monkeypatch)

    async def scenario():
        jobs.bind_loop(asyncio.get_running_loop())

        def turn1():
            _turn(agent_tools, conv)
            _call("ft_select_runtime", runtime="VM-A")
            e = _call("ft_edit_notebook", path="smoke_test_cpu", ops=[
                {"op": "set_parameters", "values": {"epochs": 2, "sleep_per_epoch": 0.0}}])
            return _call("ft_propose_run", notebook=e["notebook"]["path"], summary="2-epoch smoke test")
        plan = await asyncio.to_thread(turn1)
        assert plan["ok"], plan

        def turn2():                                                   # user: "yes, launch"
            _turn(agent_tools, conv)
            return _call("ft_submit_job", plan_id=plan["plan_id"])
        sub = await asyncio.to_thread(turn2)
        assert sub["ok"], sub
        again = await asyncio.to_thread(turn2)                         # idempotent
        assert again["job"]["job_id"] == sub["job"]["job_id"]
        return await _wait(sub["job"]["job_id"])

    doc = asyncio.run(scenario())
    p = doc["progress"]
    assert doc["status"] == "completed", p.get("error")
    assert [s["status"] for s in p["stages"]] == ["done", "done", "done", "done", "done", "done", "skipped", "done"] \
        or all(s["status"] in ("done", "skipped") for s in p["stages"])
    assert p["percent"] == 100.0 and p["custom"]["step"] == 2
    assert p["outputs"]["copied"] == 2                                 # model.json + metrics.json
    assert "Fine-tuning job" in p["summary"]
    ses = S.by_conversation(conv)
    assert doc["job_id"] in ses["jobs"] and ses["plan"]["status"] == "launched"
    run_dir = p["output_dir"]
    assert filestore.stat(f"{run_dir}/model/MANIFEST.json")
    executed = json.loads(filestore.read_text(f"{run_dir}/notebook/parameterized.ipynb"))
    assert any("epochs = 2" in "".join(c["source"]) for c in executed["cells"])


def test_cancel_stops_a_running_job(bound, tmp_path, monkeypatch):
    agent_tools, conv = bound
    FakeRuntime(tmp_path).install(monkeypatch)

    async def scenario():
        jobs.bind_loop(asyncio.get_running_loop())

        def turn1():
            _turn(agent_tools, conv)
            _call("ft_select_runtime", runtime="VM-A")
            e = _call("ft_edit_notebook", path="smoke_test_cpu", ops=[
                {"op": "set_parameters", "values": {"epochs": 50, "sleep_per_epoch": 1.0}}])
            p = _call("ft_propose_run", notebook=e["notebook"]["path"], summary="long one")
            _turn(agent_tools, conv)
            return _call("ft_submit_job", plan_id=p["plan_id"])
        sub = await asyncio.to_thread(turn1)
        jid = sub["job"]["job_id"]
        for _ in range(60):
            if (jobs.get(jid).get("progress") or {}).get("stage") == "execute":
                break
            await asyncio.sleep(0.5)
        r = await asyncio.to_thread(lambda: (_turn(agent_tools, conv), _call("ft_cancel_job", job_id=jid))[1])
        assert r["ok"]
        return await _wait(jid)

    doc = asyncio.run(scenario())
    assert doc["status"] == "cancelled"
    assert doc["progress"]["job"]["final_state"] == "cancelled"


def test_execute_tool_routes_finetune_kind(bound):
    agent_tools, conv = bound
    _turn(agent_tools, conv)
    defs, reg = build_tool_defs(agent_tools)
    out = json.loads(execute_tool(reg, "ft_session_status", {}))
    assert out["ok"] and out["session_id"].startswith("ftses_")


# ------------------------------------------------------------------ retiring build 37
def test_legacy_pipeline_is_retired_and_assistant_seeded():
    from app.schemas import AgentCard, WorkflowGraph
    from app.store import agents as A, workflows as W
    from app.runtime import handlers
    A.upsert(AgentCard(agent_name="Fine-Tuning Planner", handler="finetune_planner").model_dump(by_alias=True))
    W.upsert(WorkflowGraph(name=seed.LEGACY_WORKFLOW, config={"template": True}).model_dump(by_alias=True))
    seed.seed()
    assert all(a["status"] == "deprecated" for a in A.list(lambda x: x["agent_name"] == "Fine-Tuning Planner"))
    legacy = W.list(lambda x: x["name"] == seed.LEGACY_WORKFLOW)
    assert legacy and all(w["config"]["deprecated"] and not w["config"]["template"] for w in legacy)
    asst = A.list(lambda x: x["agent_name"] == seed.ASSISTANT)[0]
    assert any(t["type"] == "finetune" for t in asst["tools"]) and asst["max_tool_rounds"] == 12
    wf = W.list(lambda x: x["name"] == seed.WORKFLOW_NAME)[0]
    assert len(wf["nodes"]) == 1 and wf["config"]["chat_history"] == "full" and wf["config"]["template"]
    res = asyncio.run(handlers._registry()["finetune_planner"](None))
    assert res.status == "failed" and "Fine-Tuning Assistant" in res.error
