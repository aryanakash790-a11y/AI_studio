"""Fine-tuning on DSLab runtimes — the Fine-Tuning Assistant (v2).

A conversational agent (an ordinary Agent Card) drives fine-tuning through the Fine-Tuning
toolset; long jobs run in a job service that needs no model calls.

  tools.py          the Fine-Tuning toolset (binding type "finetune"): runtimes, notebooks,
                    plan, jobs, ask-user; scopes the bound DSLab MCP tools
  session.py        per-conversation state (runtime, working notebook, plan, question, jobs)
  notebook_ops.py   read / create / edit (cell ops) / validate / diff notebooks; drafts+versions
  jobs.py           job service: stage -> launch -> monitor -> collect -> MLflow (+ resume)
  remote.py         runtime access through the agent's bound DSLab MCP server
  catalog.py        template notebooks in File Storage (finetuning/notebooks/)
  notebook.py       papermill-style parameter injection
  planner.py        validation of a proposed run (parameters, dataset)
  mlflow_log.py     MLflow run per job (MlflowClient only - never fluent)
  storage.py        per-job output folder (finetuning/runs/<job_id>/)
  progress.py       structured, persisted job progress
  config.py         settings (Governance settings doc, section "finetune")
  resume.py         re-attach jobs after a backend restart
  seed.py           folders, sample notebooks, the assistant agent and its workflow
  remote_runner.py  the job runner that executes ON the DSLab runtime
"""

# set by the app's shutdown hook: a cancelled monitor must then leave the remote job running
SHUTTING_DOWN = {"value": False}
