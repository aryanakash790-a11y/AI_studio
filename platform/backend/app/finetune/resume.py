"""Re-attach fine-tuning jobs after a backend restart.

A job runs detached on its DSLab runtime, so a restart does not stop it. On startup every job
that was queued or running is handed back to the job service, which goes straight back to
monitoring (jobs.resume). Runs of the retired build-37 pipeline are closed honestly.
"""
from __future__ import annotations


async def resume_detached(delay: float = 3.0) -> int:
    from . import jobs
    return await jobs.resume(delay)
