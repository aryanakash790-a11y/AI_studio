"""Notebook authoring for the Fine-Tuning Assistant: read, create, edit, validate, diff.

Copy-on-write: templates in `finetuning/notebooks/` (and any other File Storage notebook) are
never modified by the assistant. The first edit copies the notebook into the conversation's
draft folder; every edit then writes the current draft and a numbered version snapshot:

    finetuning/workspace/<session_id>/<stem>.ipynb                  current draft
    finetuning/workspace/<session_id>/.versions/<stem>.v<N>.ipynb   every saved version

Edits are structured cell operations (not raw JSON), applied in order:
    replace_cell   {index, source, cell_type?}
    insert_cell    {index, source, cell_type="code", tags?}   (index = position; -1 appends)
    delete_cell    {index}
    replace_text   {find, replace, index?}                    (exact text; all cells if no index)
    set_parameters {values: {name: value}}                    (edits the `parameters` cell)
    set_tags       {index, tags}
"""
from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import re

from .. import filestore
from . import config
from .notebook import _src, _tags, parameters_cell_index, read_parameters

MAX_CELL_CHARS = 6000
CELL_TYPES = ("code", "markdown", "raw")


class NotebookError(ValueError):
    pass


# ------------------------------------------------------------------ paths
def drafts_dir(session_id: str) -> str:
    return f"{config.get()['workspace_dir'].strip('/')}/{session_id}"


def _norm(path: str) -> str:
    p = (path or "").strip().strip("/")
    if not p:
        raise NotebookError("a notebook path is required")
    if not p.endswith(".ipynb"):
        raise NotebookError(f"'{path}' is not a notebook (.ipynb)")
    return p


def resolve(path: str, session_id: str = "") -> str:
    """Accept a full File Storage path, a catalog name, or a bare draft file name."""
    p = (path or "").strip().strip("/")
    if p and not p.endswith(".ipynb"):
        p += ".ipynb"
    candidates = [p]
    if session_id:
        candidates.append(f"{drafts_dir(session_id)}/{p}")
    candidates.append(f"{config.get()['notebooks_dir'].strip('/')}/{p}")
    for c in candidates:
        try:
            st = filestore.stat(c)
        except filestore.StorageError:
            st = None
        if st and st["type"] == "file":
            return st["path"]
    raise NotebookError(f"notebook '{path}' not found in File Storage")


def is_draft(path: str, session_id: str) -> bool:
    return bool(session_id) and path.startswith(drafts_dir(session_id) + "/")


def load(path: str) -> dict:
    try:
        nb = json.loads(filestore.read_bytes(_norm(path)).decode("utf-8"))
    except filestore.StorageError as e:
        raise NotebookError(str(e))
    except json.JSONDecodeError as e:
        raise NotebookError(f"'{path}' is not valid notebook JSON: {e}")
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise NotebookError(f"'{path}' has no cell list")
    return nb


def nb_hash(nb: dict) -> str:
    core = [{"t": c.get("cell_type"), "s": _src(c), "tags": _tags(c)} for c in nb.get("cells", [])]
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ building blocks
def new_cell(source: str, cell_type: str = "code", tags: list | None = None) -> dict:
    cell_type = cell_type if cell_type in CELL_TYPES else "code"
    c: dict = {"cell_type": cell_type, "metadata": {"tags": list(tags)} if tags else {},
               "source": source or ""}
    if cell_type == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def new_notebook(cells: list[dict] | None = None) -> dict:
    return {"cells": cells or [], "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}}}


def _clear_outputs(nb: dict) -> dict:
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"], c["execution_count"] = [], None
    return nb


def summarize(nb: dict, start: int = 0, end: int | None = None, max_chars: int = MAX_CELL_CHARS) -> dict:
    cells = nb.get("cells", [])
    end = len(cells) if end is None else max(0, min(len(cells), int(end)))
    start = max(0, int(start or 0))
    out = []
    for i in range(start, end):
        c = cells[i]
        s = _src(c)
        out.append({"index": i, "type": c.get("cell_type"), "tags": _tags(c),
                    "source": s if len(s) <= max_chars else s[:max_chars] + "\n…(truncated)"})
    pidx = parameters_cell_index(nb)
    return {"total_cells": len(cells), "showing": [start, end], "parameters_cell": pidx,
            "parameters": read_parameters(nb), "cells": out}


# ------------------------------------------------------------------ edit operations
def _idx(nb: dict, i, allow_end: bool = False) -> int:
    n = len(nb["cells"])
    try:
        i = int(i)
    except (TypeError, ValueError):
        raise NotebookError(f"cell index must be an integer, got {i!r}")
    if i < 0:
        i = n + 1 + i if allow_end else n + i
    hi = n if allow_end else n - 1
    if i < 0 or i > hi:
        raise NotebookError(f"cell index {i} out of range (notebook has {n} cells)")
    return i


def _set_param_lines(src: str, values: dict) -> tuple[str, list[str]]:
    lines = src.splitlines()
    done, notes = set(), []
    for k, v in values.items():
        if not str(k).isidentifier():
            notes.append(f"skipped invalid parameter name `{k}`")
            continue
        pat = re.compile(rf"^(\s*){re.escape(k)}\s*(:[^=]+)?=\s*(.*?)(\s+#.*)?$")
        for n, line in enumerate(lines):
            m = pat.match(line)
            if m:
                lines[n] = f"{m.group(1)}{k}{m.group(2) or ''} = {v!r}{m.group(4) or ''}"
                done.add(k)
                notes.append(f"parameter `{k}` = {v!r}")
                break
        if k not in done:
            lines.append(f"{k} = {v!r}")
            notes.append(f"added parameter `{k}` = {v!r}")
    return "\n".join(lines) + ("\n" if src.endswith("\n") or not src else ""), notes


def apply_ops(nb: dict, ops: list[dict]) -> tuple[dict, list[str]]:
    """Return (new notebook, change notes). Raises NotebookError on a bad operation; the
    original notebook is never modified."""
    if not isinstance(ops, list) or not ops:
        raise NotebookError("ops must be a non-empty list of edit operations")
    nb = copy.deepcopy(nb)
    notes: list[str] = []
    for k, op in enumerate(ops):
        if not isinstance(op, dict):
            raise NotebookError(f"operation {k} is not an object")
        kind = str(op.get("op") or op.get("type") or "").lower()
        if kind == "replace_cell":
            i = _idx(nb, op.get("index"))
            old = nb["cells"][i]
            ct = op.get("cell_type") or old.get("cell_type", "code")
            cell = new_cell(str(op.get("source", "")), ct, _tags(old))
            if old.get("id"):
                cell["id"] = old["id"]
            nb["cells"][i] = cell
            notes.append(f"cell {i}: replaced ({ct})")
        elif kind == "insert_cell":
            i = _idx(nb, op.get("index", -1), allow_end=True)
            ct = op.get("cell_type") or "code"
            nb["cells"].insert(i, new_cell(str(op.get("source", "")), ct, op.get("tags")))
            notes.append(f"cell {i}: inserted ({ct})")
        elif kind == "delete_cell":
            i = _idx(nb, op.get("index"))
            if "parameters" in _tags(nb["cells"][i]):
                notes.append(f"cell {i}: deleted the parameters cell")
            del nb["cells"][i]
            notes.append(f"cell {i}: deleted")
        elif kind == "replace_text":
            find, repl = str(op.get("find") or ""), str(op.get("replace") or "")
            if not find:
                raise NotebookError(f"operation {k}: replace_text needs 'find'")
            idxs = [_idx(nb, op["index"])] if op.get("index") is not None else range(len(nb["cells"]))
            hits = 0
            for i in idxs:
                s = _src(nb["cells"][i])
                if find in s:
                    nb["cells"][i]["source"] = s.replace(find, repl)
                    hits += s.count(find)
                    notes.append(f"cell {i}: replaced text")
            if not hits:
                raise NotebookError(f"operation {k}: text {find[:80]!r} not found")
        elif kind == "set_parameters":
            values = op.get("values") or {}
            if not isinstance(values, dict) or not values:
                raise NotebookError(f"operation {k}: set_parameters needs 'values' {{name: value}}")
            pidx = parameters_cell_index(nb)
            if pidx is None:
                cell = new_cell("", "code", ["parameters"])
                nb["cells"].insert(0, cell)
                pidx = 0
                notes.append("cell 0: created a `parameters` cell")
            src, pn = _set_param_lines(_src(nb["cells"][pidx]), values)
            nb["cells"][pidx]["source"] = src
            notes += [f"cell {pidx}: {n}" for n in pn]
        elif kind == "set_tags":
            i = _idx(nb, op.get("index"))
            nb["cells"][i].setdefault("metadata", {})["tags"] = [str(t) for t in (op.get("tags") or [])]
            notes.append(f"cell {i}: tags = {nb['cells'][i]['metadata']['tags']}")
        else:
            raise NotebookError(f"operation {k}: unknown op '{kind}' (use replace_cell, insert_cell, "
                                f"delete_cell, replace_text, set_parameters or set_tags)")
    return _clear_outputs(nb), notes


# ------------------------------------------------------------------ validation + diff
_MAGIC = re.compile(r"^\s*([%!?]|.*\?\s*$)")


def _python_of(src: str) -> str:
    """Neutralise IPython syntax (magics, shell escapes, help) so ast can check the rest."""
    out = []
    for line in src.splitlines():
        if _MAGIC.match(line):
            indent = len(line) - len(line.lstrip())
            out.append(" " * indent + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def validate(nb: dict) -> dict:
    errors, warnings = [], []
    if nb.get("nbformat") != 4:
        warnings.append("nbformat is not 4 - the runner expects a v4 notebook")
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append({"cell": None, "error": "the notebook has no cells"})
        return {"ok": False, "errors": errors, "warnings": warnings}
    for i, c in enumerate(cells):
        ct = c.get("cell_type")
        if ct not in CELL_TYPES:
            errors.append({"cell": i, "error": f"unknown cell_type {ct!r}"})
            continue
        src = c.get("source", "")
        if not isinstance(src, (str, list)):
            errors.append({"cell": i, "error": "source must be text"})
            continue
        if ct == "code":
            text = _src(c)
            if not text.strip():
                continue
            if text.lstrip().startswith("%%"):
                continue                   # cell magic: body is not Python
            try:
                ast.parse(_python_of(text))
            except SyntaxError as e:
                errors.append({"cell": i, "error": f"SyntaxError line {e.lineno}: {e.msg}",
                               "line": (e.text or "").rstrip()[:200]})
    if parameters_cell_index(nb) is None:
        warnings.append("no cell tagged 'parameters' - run parameters will be injected as the first cell")
    elif not read_parameters(nb):
        warnings.append("the 'parameters' cell has no literal assignments")
    if not any(c.get("cell_type") == "code" and _src(c).strip() for c in cells):
        errors.append({"cell": None, "error": "the notebook has no code to run"})
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _cell_text(nb: dict) -> list[str]:
    lines = []
    for i, c in enumerate(nb.get("cells", [])):
        tags = _tags(c)
        lines.append(f"# ── cell {i} [{c.get('cell_type')}]" + (f" tags={tags}" if tags else ""))
        lines.extend(_src(c).splitlines())
    return lines


def diff(a: dict, b: dict, a_name: str = "before", b_name: str = "after", max_lines: int = 300) -> str:
    d = list(difflib.unified_diff(_cell_text(a), _cell_text(b), a_name, b_name, lineterm="", n=2))
    if len(d) > max_lines:
        d = d[:max_lines] + [f"… ({len(d) - max_lines} more diff lines)"]
    return "\n".join(d)


# ------------------------------------------------------------------ drafts + versions
def _stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".ipynb")


def _next_version(session_id: str, stem: str) -> int:
    vdir = f"{drafts_dir(session_id)}/.versions"
    try:
        files = filestore.walk(vdir)
    except filestore.StorageError:
        files = []
    nums = [int(m.group(1)) for f in files
            if (m := re.match(rf"^{re.escape(stem)}\.v(\d+)\.ipynb$", f["name"]))]
    return max(nums, default=0) + 1


def save_draft(session_id: str, nb: dict, *, name: str, base: str = "") -> dict:
    """Write the current draft and a version snapshot. Returns the notebook record."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", _stem(name)).strip("._") or "notebook"
    path = f"{drafts_dir(session_id)}/{stem}.ipynb"
    v = _next_version(session_id, stem)
    data = json.dumps(nb, indent=1)
    filestore.write_text(path, data, actor="finetune-assistant", enforce_limits=False)
    filestore.write_text(f"{drafts_dir(session_id)}/.versions/{stem}.v{v}.ipynb", data,
                         actor="finetune-assistant", enforce_limits=False)
    return {"path": path, "name": f"{stem}.ipynb", "base": base or path, "version": v,
            "hash": nb_hash(nb)}


def version_path(session_id: str, path: str, version: int) -> str:
    return f"{drafts_dir(session_id)}/.versions/{_stem(path)}.v{int(version)}.ipynb"


def list_drafts(session_id: str) -> list[dict]:
    try:
        files = filestore.walk(drafts_dir(session_id))
    except filestore.StorageError:
        return []
    return [{"path": f["path"], "name": f["name"], "size": f["size"], "modified": f["modified"]}
            for f in files if f["name"].endswith(".ipynb") and "/.versions/" not in f["path"]]


def save_as_template(path: str, name: str, overwrite: bool = False) -> str:
    nb = load(path)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", _stem(name or path)).strip("._") or "notebook"
    target = f"{config.get()['notebooks_dir'].strip('/')}/{stem}.ipynb"
    if filestore.stat(target) and not overwrite:
        raise NotebookError(f"template '{stem}.ipynb' already exists - choose another name or "
                            f"confirm overwrite with the user")
    filestore.write_text(target, json.dumps(_clear_outputs(copy.deepcopy(nb)), indent=1),
                         actor="finetune-assistant", enforce_limits=False)
    return target
