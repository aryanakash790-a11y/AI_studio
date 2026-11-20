"""Idempotent seeding for fine-tuning (called from app.seed on every start):

  * File Storage folders: finetuning/notebooks, datasets, runs, workspace
  * the sample notebooks + README, copied ONCE (deleting or editing them is respected)
  * the "Fine-Tuning Assistant" agent (Fine-Tuning toolset + any DSLab runtime MCP servers that
    are registered at first seed — admins change the bindings in Agent Garden afterwards)
  * the one-node "Fine-Tuning Assistant" workflow (Template, Chat, Domain AI › Fine-Tuning Studio)
  * retires the build-37 pipeline: its 3 handler agents are deprecated and its workflow hidden

Existing records are never overwritten, so user edits survive restarts.
"""
from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger("tcs.finetune.seed")
SAMPLES = Path(__file__).resolve().parent / "samples"
SEED_VERSION = 1

ASSISTANT = "Fine-Tuning Assistant"
WORKFLOW_NAME = "Fine-Tuning Assistant"
LEGACY_WORKFLOW = "Fine-Tuning (DSLab)"
LEGACY_AGENTS = ("Fine-Tuning Planner", "Fine-Tuning Executor", "Fine-Tuning Reporter")

ASSISTANT_PROMPT = """You are the Fine-Tuning Assistant of TCS Domain AI Studio. You help the user \
fine-tune models by preparing, editing and running Jupyter notebooks on Data Science Lab (DSLab) \
GPU runtimes, in a conversation. You act only through your tools; never claim you did something \
a tool did not confirm, and never invent metrics, files or runtimes.

The FINE-TUNING SESSION STATE block in every message is authoritative: it tells you the runtimes \
you may use, the selected runtime, the working notebook (and its version), a pending plan and \
recent jobs. Continue the conversation from it; do not start over.

How to work:
1. Understand the goal (task, base model, data, method such as LoRA/QLoRA/full, constraints). If \
something essential is missing, ask - one question at a time.
2. Runtime: if none is selected, call ft_list_runtimes, then ft_ask_user with purpose='runtime' \
(the platform shows the runtimes as clickable options and records the user's click itself; the \
session state then shows selected_runtime). If the user answers in words instead, call \
ft_select_runtime with what they said. Never pick a runtime yourself, even if only one exists. \
Right after a runtime is selected, check it with ft_runtime_info (ft_select_runtime already \
does). If no runtime is available or a runtime tool returns an error, quote the error exactly \
and say how to fix it (e.g. open the lab in Data Science Lab, check SSH, enable the MCP \
server) - never describe it vaguely.
3. Notebook: call ft_list_notebooks. If several could fit or the user did not say, ask with \
ft_ask_user. Read it with ft_read_notebook before changing or running it. You may also create a \
new notebook (ft_create_notebook): first markdown cell = title and description, one code cell \
tagged 'parameters' with literal defaults, write outputs to output_dir.
4. Changes: apply the user's instructions with ft_edit_notebook (structured cell operations; \
templates are never modified, edits go to a versioned draft). Prefer set_parameters for values \
that are parameters. After editing, report the changes briefly (cell numbers, what changed) \
and the validation result. Fix validation errors before proposing a run. With ft_runtime_info, \
ft_run_python, ft_run_shell and ft_runtime_files (they act on the selected runtime) you may \
check the environment or test a small snippet - never run the full training that way.
5. Run: when runtime and notebook are ready, call ft_propose_run with the parameters and a \
one-line summary. Then summarise the plan (runtime, notebook + version, key parameters, dataset) \
and ask the user to click Launch or reply 'launch'. Do NOT call ft_submit_job in the same turn. \
Call ft_submit_job only when the user's latest message clearly confirms.
6. During and after a job: answer questions with ft_job_status (stage, %, cell, loss, eval \
loss, epoch, errors). Stop a job with ft_cancel_job only when the user asks. For a rerun with \
changes: edit, then propose again. After a job ends, report the outcome, key metrics with values, \
where outputs are (File Storage folder, MLflow link) and 1-3 next steps.

Style: short, concrete answers in markdown. Show code only when the user asks or when it is the \
change you made. Use ft_ask_user for choices the user must make, then end your turn."""

# Earlier seeded prompts: an assistant whose prompt is still exactly one of these is upgraded on
# start-up (a prompt edited in Agent Garden is never touched).
SEEDED_PROMPT_SHA256 = {
    "0f8f74edbf00692e85880be5e5af051a606f4d37bbee85c41ce4d8c3018d05aa",   # build 38
}

DESCRIPTION = ("Conversational fine-tuning: choose a DSLab runtime and a notebook, have the agent "
               "create or modify the notebook from your instructions, review the plan, launch, "
               "and follow the job live — ask questions or stop and rerun at any time.")


def _seed_files() -> None:
    from .. import filestore
    from ..governance import get_settings, update_settings
    from . import config
    c = config.get()
    for k in ("notebooks_dir", "datasets_dir", "runs_dir", "workspace_dir"):
        filestore.mkdir(c[k])
    state = get_settings().get("finetune_seed") or {}
    if int(state.get("samples", 0)) >= SEED_VERSION:
        return
    for f in sorted(SAMPLES.iterdir()):
        if f.suffix not in (".ipynb", ".md"):
            continue
        rel = f"{c['notebooks_dir']}/{f.name}"
        if not filestore.stat(rel):
            filestore.write_bytes(rel, f.read_bytes(), actor="system", enforce_limits=False)
    update_settings({"finetune_seed": {**state, "samples": SEED_VERSION}})


def _dslab_bindings() -> list[dict]:
    """Bindings for DSLab runtime MCP servers that exist and are enabled right now."""
    try:
        from .. import mcp_store
        from . import remote
        return [{"type": "mcp", "ref": s["id"], "name": s.get("name", s["id"])}
                for s in mcp_store.list_servers()
                if remote.is_dslab_server(s.get("id", "")) and s.get("enabled", True)]
    except Exception:
        return []


def _seed_assistant_and_workflow() -> None:
    from ..modules.data_science_lab.agent_access import _pick_endpoint
    from ..schemas import (AgentCard, AgentCategory, IOContract, ModelConfig, Prompts,
                           WorkflowGraph, WorkflowNode)
    from ..store import agents as A
    from ..store import workflows as W

    import hashlib
    for a in A.list(lambda x: x.get("agent_name") == ASSISTANT):
        cur = ((a.get("prompts") or {}).get("system_prompt") or "")
        if cur != ASSISTANT_PROMPT and hashlib.sha256(cur.encode()).hexdigest() in SEEDED_PROMPT_SHA256:
            a.setdefault("prompts", {})["system_prompt"] = ASSISTANT_PROMPT
            A.upsert(a)
    if not any(a.get("agent_name") == ASSISTANT for a in A.list()):
        card = AgentCard(
            agent_name=ASSISTANT, role="Fine-tuning engineer", category=AgentCategory.use_case,
            description=("Plans and runs fine-tuning with you: picks the DSLab runtime and notebook "
                         "with you, creates or edits notebooks from your instructions, launches "
                         "confirmed runs and reports live progress."),
            tags=["finetuning", "dslab", "mlflow", "notebooks", "interactive"],
            model_config=ModelConfig(endpoint_id=_pick_endpoint("") or None, temperature=0.2),
            prompts=Prompts(system_prompt=ASSISTANT_PROMPT),
            io_contract=IOContract(output_key="ft_reply"),
            tools=[{"type": "finetune", "name": "Fine-Tuning toolset"}] + _dslab_bindings(),
            max_tool_rounds=12,
            owner={"user_id": "system", "team": "finetuning"},
            metrics={"runs": 0, "success_rate": None, "avg_cost_usd": None})
        A.upsert(card.model_dump(by_alias=True))

    if not any(w.get("name") == WORKFLOW_NAME for w in W.list()):
        wf = WorkflowGraph(
            name=WORKFLOW_NAME,
            goal_prompt=("Help the user fine-tune a model interactively: agree the DSLab runtime and "
                         "notebook, create or modify the notebook as instructed, propose the run for "
                         "confirmation, and follow the launched job."),
            nodes=[WorkflowNode(node_id="n1", agent_ref=f"agent_cards/{ASSISTANT}",
                                output_key="ft_reply",
                                input_bindings={"message": "workflow.input.message"})],
            edges=[],
            config={"template": True, "category": "finetuning", "interactive": True,
                    "chat_history": "full", "screen": "ftstudio",
                    "run_timeout_seconds": 600, "max_iterations": 3,
                    "description": DESCRIPTION,
                    "input_hint": ("e.g. Fine-tune distilbert on finetuning/datasets/tickets.csv for "
                                   "intent classification, use bf16 and add early stopping")})
        W.upsert(wf.model_dump(by_alias=True))


def _retire_legacy() -> None:
    """Hide the build-37 handler pipeline. Records are kept so old runs still display."""
    from ..store import agents as A
    from ..store import workflows as W
    for a in A.list(lambda x: x.get("agent_name") in LEGACY_AGENTS and x.get("handler")):
        if a.get("status") != "deprecated":
            a["status"] = "deprecated"
            a["description"] = ("Retired: replaced by the Fine-Tuning Assistant. "
                                + (a.get("description") or ""))[:1000]
            A.upsert(a)
    for w in W.list(lambda x: x.get("name") == LEGACY_WORKFLOW):
        cfg = w.get("config") or {}
        if not cfg.get("deprecated"):
            cfg.update({"deprecated": True, "template": False,
                        "description": "Retired - use the Fine-Tuning Assistant."})
            w["config"] = cfg
            W.upsert(w)


def seed() -> None:
    try:
        _seed_files()
    except Exception as e:
        _log.warning("fine-tuning file seed skipped: %s", e)
    _seed_assistant_and_workflow()
    try:
        _retire_legacy()
    except Exception as e:
        _log.warning("retiring the legacy fine-tuning workflow skipped: %s", e)
