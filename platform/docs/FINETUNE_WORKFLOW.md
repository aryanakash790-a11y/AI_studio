# Fine-Tuning Assistant (v1.0 build 38)

Fine-tuning is a **conversation** with the *Fine-Tuning Assistant*: an ordinary Agent Card with
the **Fine-Tuning toolset**, running in a one-node workflow. It replaces the build-37
*Fine-Tuning (DSLab)* pipeline (three handler-backed agents with a fixed planner → executor →
reporter order), which could only pick an existing notebook, ran it on the runtime set in File
Storage settings, and could not change notebooks or talk to the user.

The same workflow is available in three places:

| Where | How |
|---|---|
| **Domain AI › Fine-Tuning Studio** | conversation on the left; workspace on the right (runtime, notebook with diff and versions, plan with Launch, jobs with live progress) |
| **Chat** | pick *Fine-Tuning Assistant* in the workflow list; choices, plan card and progress appear in the replies |
| **Templates** | the *Fine-Tuning Assistant* card opens Fine-Tuning Studio |

A conversation started in one place is listed in the other.

## 1. Architecture: agents decide, tools act, the platform tracks

```
Chat / Fine-Tuning Studio ──POST /runs (one run per message, full history + conversation_id)──▶
  n1  Fine-Tuning Assistant      ordinary agent: prompt, Model Endpoint, tools from Agent Garden
        tools: Fine-Tuning toolset (binding {"type":"finetune"})
               DSLab · <runtime> MCP servers ticked in Agent Garden (scoped, see §3)
        state: finetune_sessions/<id>  (runtime, working notebook + version, plan, question, jobs)
                └─ injected into the prompt every turn ("FINE-TUNING SESSION STATE")
  ── user confirms the plan (Launch button / explicit reply) ──▶
  Job service (finetune/jobs.py, no model calls)
        stage → launch → monitor → collect → MLflow → summary, through the agent's bound MCP server
        finetune_jobs/<job_id>.progress  ──▶ progress bar in Chat and Fine-Tuning Studio
```

* **Interactive.** Every message continues the conversation. The workflow sets
  `config.chat_history = "full"`, so Chat sends both sides of the conversation, and the session
  state re-grounds the agent each turn.
* **Configuration, not code.** The model, prompt and tools come from the Agent Card. The agent
  reaches a runtime **only** through a DSLab MCP server that is bound to it in Agent Garden and
  enabled on the MCP Servers screen. Nothing is registered or picked automatically.
* **The model never drives the long job.** A job is a separate record watched by the job service,
  so a 6-hour training costs no tokens. The chat turn that launched it ends right away, and you
  can keep talking.

## 2. The Fine-Tuning toolset

| Tool | Purpose |
|---|---|
| `ft_session_status` | current runtime, notebook + version, plan, recent jobs |
| `ft_ask_user` | ask a question with clickable options and end the turn |
| `ft_list_runtimes`, `ft_select_runtime` | the runtimes bound to this agent; record the user's choice (tolerant: name, id, the clicked option, "yes" when only one exists) and check the runtime responds |
| `ft_runtime_info`, `ft_run_python`, `ft_run_shell`, `ft_runtime_files` | act on the **selected** runtime through its bound DSLab MCP server; available in the same turn the runtime is chosen |
| `ft_list_notebooks`, `ft_read_notebook` | templates (`finetuning/notebooks/`) and this conversation's drafts; cells with indices |
| `ft_create_notebook` | new notebook from cells (markdown description + `parameters` cell) |
| `ft_edit_notebook` | structured cell operations: `replace_cell`, `insert_cell`, `delete_cell`, `replace_text`, `set_parameters`, `set_tags` |
| `ft_validate_notebook`, `ft_diff_notebook` | structure + Python syntax per code cell (magics allowed); diff vs base or an earlier version |
| `ft_save_as_template` | copy a draft into the template folder (only when asked) |
| `ft_list_datasets` | datasets in File Storage |
| `ft_propose_run` | validate runtime + notebook + parameters + dataset → plan with a Launch button |
| `ft_submit_job` | launch a plan **only in a later turn** (the user confirmed) |
| `ft_job_status`, `ft_cancel_job`, `ft_list_jobs` | live job status, stop, list |

**Copy-on-write notebooks.** Templates are never modified. The first edit copies the notebook to
`finetuning/workspace/<session>/<name>.ipynb`; every edit also writes
`.versions/<name>.v<N>.ipynb`. The workspace panel shows the diff against the template and lets you
open any version.

## 3. Runtime scoping and launch safety

* `ft_list_runtimes` offers only the `DSLab · <runtime>` MCP servers bound to the agent. The agent
  must ask the user to choose one, even when only one exists.
* The runtime question is asked with `ft_ask_user(purpose="runtime")`: the platform fills the
  options with its own labels (`vm1 · T4`) and records the user's click itself at the start of the
  next turn. Selection does not depend on how the model reads "Yes".
* The bound DSLab MCP servers' raw tools are never exposed. The agent reaches the selected runtime
  through `ft_runtime_*`, which call that runtime's bound, enabled MCP server. This works in the
  same turn the runtime is chosen, and job control (`start_notebook_job`, `run_notebook` …) stays
  behind the confirmed plan. The ADK runtime's native MCP toolsets follow the same rule.
* **Labs after a restart:** lab tunnels live in the backend process, so after every restart or
  redeploy all labs are closed. Before any runtime call (selection check, `ft_runtime_*`, job
  staging) the backend opens the lab itself (`remote.ensure_lab_open`). It does not rely on the MCP
  server's auto-open, which runs inside one MCP call and can time out. If opening fails, the exact
  reason (SSH, Jupyter start …) is returned to the agent and shown in the Studio workspace
  (*Runtime did not respond*).
* `ft_submit_job` is refused in the turn the plan was proposed, and when the notebook changed after
  the proposal (hash check). The plan id makes launching idempotent. The **Launch** button calls
  `POST /finetune/sessions/{id}/plans/{plan}/launch` with the same checks.

## 4. Progress

Stages: **Configure → Validate notebook → Stage on DSLab → Launch job → Execute notebook (cell x/y,
step/total, loss, eval loss, epoch, ETA, log tail) → Collect outputs → Log to MLflow → Summary**.
Progress is persisted on the job after every poll. Chat and Fine-Tuning Studio poll
`GET /finetune/jobs/{id}`, so progress survives reloads and backend restarts. Stop is available on
every running job.

## 5. Setup

1. **Data Science Lab:** enable it, register a runtime, and click **Enable agent access** on it.
2. **Agent Garden › Fine-Tuning Assistant › Tools:** tick the `DSLab · <runtime>` MCP servers the
   assistant may use. The *Fine-Tuning toolset* switch and *Max tool rounds* (12) are on the same
   step. Runtimes registered before the first start are bound automatically by the seed.
3. **Model Endpoint:** a chat endpoint with tool calling for the assistant.
4. **MLflow reachable from the runtime:** set it in the runtime settings or in *File Storage ›
   Fine-tuning settings*, plus `MLFLOW_UI_URL` for browser links. Otherwise everything as in
   build 37.
5. **Runtime Python:** the notebook's libraries (torch, transformers, peft…). The runner needs only
   the standard library plus `nbclient`.

## 6. Where everything goes

| Place | Content |
|---|---|
| `finetuning/workspace/<session>/` | drafts and `.versions/` of every edited/created notebook |
| `finetuning/runs/<job_id>/` | `config/`, `notebook/` (snapshot, parameterised, executed), `logs/`, `metrics/`, `model/` + `MANIFEST.json`, `summary.md` |
| DSLab runtime `studio-jobs/<job_id>/` | full job directory |
| MLflow experiment `finetuning` | one run per job (unchanged from build 37) |
| collections | `finetune_sessions`, `finetune_jobs` |

## 7. Settings (`/api/v1/finetune/config`)

As in build 37, plus `workspace_dir` (`finetuning/workspace`) and `max_concurrent_jobs` (2; further
jobs show as *queued*). `default_runtime_id` is now only a suggestion the assistant may mention; it
is never used without asking.

## 8. API

`GET /finetune/sessions/by-conversation/{cid}`, `GET /finetune/sessions/{id}`,
`GET /finetune/sessions/{id}/notebook?path=&version=`, `POST /finetune/sessions/{id}/plans/{plan}/launch`,
`GET /finetune/jobs?session_id=`, `GET /finetune/jobs/{id}`, `POST /finetune/jobs/{id}/cancel`,
plus `config`, `catalog`, `runtimes` (now includes which agents each runtime's server is bound to).

## 9. What changed from build 37

| Area | Change |
|---|---|
| `finetune/steps.py`, `finetune/llm.py` | removed; the job engine moved to `finetune/jobs.py` (logic reused) |
| `finetune/planner.py` | keyword/LLM notebook selection removed; validation (`finalize`) kept |
| `finetune/remote.py` | `resolve_runtime` / `ensure_server` (implicit runtime choice and MCP auto-registration) removed; `server_for` uses the agent's bound server |
| new | `finetune/tools.py`, `session.py`, `notebook_ops.py`, `jobs.py`; `runtime/conversation.py` |
| `tool_exec.py` | tool execution context (run, agent, conversation); `finetune` binding type; toolset prompt/after-step hooks; MCP scoping helper |
| `runtime/live.py`, `adk.py` | set the tool context, per-agent `max_tool_rounds`, full history for `chat_history: "full"` workflows, toolset hooks. Ordinary workflows unchanged |
| `schemas.py` | `ToolType.finetune`, `AgentCard.max_tool_rounds` |
| `runtime/handlers.py` | the 3 fine-tuning handlers retired (their agents are deprecated and the old workflow hidden; old runs still display) |
| UI | `FineTuneSession.tsx` (choices, plan + Launch, job progress), `domainai/FineTuningStudio.tsx`, Chat (embeddable, full history, choices), Templates (opens the Studio), agent wizard (toolset switch, max tool rounds), API client |
| Chat fix | settled conversations are saved from the current messages; before, a re-render could save an empty conversation |

## 10. Verification

* `STORE_BACKEND=json python -m pytest tests -q` → 33 tests (build 38.1 added: runtime tools in
  the selection turn, tolerant matching, platform-recorded runtime clicks, exact error reporting,
  prompt upgrade). They cover notebook ops, validation,
  history formatting, runtime scoping (unbound / selected / hidden tools, ADK filters), the
  same-turn and stale-notebook launch refusals, a full job against a simulated runtime that really
  executes the notebook with the platform runner, cancellation, and retiring build 37.
* End to end through the real API and UI (Playwright) with a scripted OpenAI-compatible model:
  runtime question with choices → choice → notebook edited into a draft (template untouched, diff
  shown) → plan → Launch → live progress → "how is it going?" answered from `ft_job_status` → job
  completed with outputs; the conversation reopened in Chat; Templates card opens the Studio. An
  existing multi-agent workflow (*Research Brief*) still runs with an unchanged prompt.
* Not verified here: a real GPU runtime, a real LLM's behaviour with the prompt, and the ADK
  runtime (google-adk is not installed in the test environment). The DSLab MCP job tools are
  unchanged from build 37.
