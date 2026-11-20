"""Run-selection validation used by ft_propose_run.

The Fine-Tuning Assistant chooses the notebook and parameters in conversation with the user; this
module VALIDATES that choice before anything can be launched: parameters must be ones the
notebook declares (values coerced to the declared type, `name=value` pairs honoured), and the
dataset must exist in File Storage (anything else, e.g. a Hugging Face dataset id, is passed
through to the notebook's dataset parameter with a warning).
(Build 37's keyword/LLM notebook *selection* was retired with the handler pipeline.)
"""
from __future__ import annotations

import re

from .. import filestore
from . import config
from .notebook import coerce

DATASET_PARAM_NAMES = ("dataset_path", "data_path", "train_file", "train_path", "train_data",
                       "dataset_file", "data_file", "train_dataset", "dataset")


def dataset_candidates(limit: int = 80) -> list[dict]:
    c = config.get()
    skip = (c["notebooks_dir"] + "/", c["runs_dir"] + "/")
    out: list[dict] = []
    for prefix in (c["datasets_dir"], "shared"):
        try:
            for f in filestore.walk(prefix, limit=limit):
                if not f["path"].startswith(skip) and not f["name"].endswith(".ipynb"):
                    out.append({"path": f["path"], "size": f["size"]})
        except filestore.StorageError:
            pass
    return out[:limit]


def kv_pairs(text: str, names: list[str]) -> dict:
    """key=value / key: value pairs in free text, for declared parameter names only."""
    out = {}
    for n in names:
        m = re.search(rf"(?<![\w.]){re.escape(n)}\s*[=:]\s*(\"[^\"]*\"|'[^']*'|[^\s,;]+)", text)
        if m:
            v = m.group(1).strip().strip("\"'")
            out[n] = v
    return out


def _dataset_entry(path: str) -> dict | None:
    if not path:
        return None
    try:
        st = filestore.stat(path.strip("/"))
    except filestore.StorageError:
        return None
    return {"path": st["path"], "size": st["size"]} if st and st["type"] == "file" else None


def finalize(entry: dict, method: str, req: dict, raw: dict | None, confidence: float,
             reason: str, alternatives: list) -> dict:
    """Validated selection. Unknown parameters are dropped with a warning."""
    declared = {p["name"]: p for p in entry["parameters"]}
    warnings: list[str] = []
    params: dict = {}
    proposed = dict((raw or {}).get("parameters") or {})
    proposed.update(kv_pairs(req["text"], list(declared)))
    proposed.update(req["parameters"])                      # explicit always wins
    for k, v in proposed.items():
        if k in declared:
            params[k] = coerce(v, declared[k]["default"])
        else:
            warnings.append(f"ignored parameter `{k}` - `{entry['name']}` does not declare it")
    if not declared and proposed:
        warnings.append("this notebook has no 'parameters' cell, so the values are injected as "
                        "new variables at the top of the notebook")
        params = {k: v for k, v in proposed.items() if str(k).isidentifier()}

    ds_path = req["dataset"] or ((raw or {}).get("dataset") or "")
    dataset = _dataset_entry(ds_path) if ds_path else None
    ds_param = (raw or {}).get("dataset_parameter") or ""
    if ds_path and not dataset:
        # not in File Storage: pass it through untouched (e.g. a Hugging Face dataset id or a
        # path that already exists on the runtime)
        guess = ds_param if ds_param in declared else next((n for n in DATASET_PARAM_NAMES if n in declared), "")
        if guess and guess not in params:
            params[guess] = ds_path
            warnings.append(f"dataset `{ds_path}` is not in File Storage - passed as-is to `{guess}`")
        elif not guess:
            warnings.append(f"dataset `{ds_path}` is not in File Storage and the notebook has no "
                            f"dataset parameter - ignored")
    if dataset and ds_param not in declared:
        ds_param = next((n for n in DATASET_PARAM_NAMES if n in declared), "")
    warnings += [str(w) for w in ((raw or {}).get("warnings") or []) if w][:8]
    return {"notebook": entry["name"], "path": entry["path"], "title": entry["title"],
            "method": method, "confidence": confidence, "reason": reason[:1500],
            "parameters": params, "dataset": dataset,
            "dataset_parameter": ds_param if dataset else "",
            "warnings": warnings, "alternatives": alternatives,
            "declared_parameters": entry["parameters"]}
