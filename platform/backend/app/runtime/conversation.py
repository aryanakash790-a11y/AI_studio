"""Conversation-aware user input for interactive workflows.

Chat starts one run per message. Ordinary workflows receive the input dict exactly as before.
A workflow that sets `config.chat_history = "full"` (e.g. the Fine-Tuning Assistant) receives
the earlier turns of the conversation (user AND assistant, role-tagged, sent by Chat) followed
by the latest message, so a conversational agent can continue where it left off.
"""
from __future__ import annotations

MAX_TURNS = 16
MAX_CHARS_PER_TURN = 2000

# platform fields that are never part of the user's request text
INTERNAL_KEYS = {"message", "input", "history", "ontology_hops", "tenant", "conversation_id"}


def wants_full_history(workflow) -> bool:
    cfg = (getattr(workflow, "config", None) or {}) if not isinstance(workflow, dict) \
        else (workflow.get("config") or {})
    return str(cfg.get("chat_history") or "").lower() == "full"


def _turns(history) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for h in history or []:
        if isinstance(h, dict):
            role = str(h.get("role") or "user").lower()
            text = str(h.get("text") or h.get("content") or "")
        else:                       # legacy Chat history: a list of earlier assistant replies
            role, text = "assistant", str(h)
        text = text.strip()
        if not text:
            continue
        if len(text) > MAX_CHARS_PER_TURN:
            text = text[:MAX_CHARS_PER_TURN] + " …(truncated)"
        out.append(("User" if role == "user" else "Assistant", text))
    return out[-MAX_TURNS:]


def format_user_input(run_input) -> str:
    """Conversation transcript + latest message, as plain text for the prompt."""
    inp = run_input if isinstance(run_input, dict) else {"input": run_input}
    message = str(inp.get("message") or inp.get("input") or "").strip()
    turns = _turns(inp.get("history"))
    parts = []
    if turns:
        parts.append("Conversation so far:\n" + "\n\n".join(f"{r}: {t}" for r, t in turns))
    parts.append(f"Latest user message:\n{message or '(empty)'}")
    return "\n\n".join(parts)
