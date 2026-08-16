"""Background job registry for web-triggered fine-tuning.

The old /training/fine-tune ran the whole training inside the request handler --
an async def calling a blocking train loop, so one training request froze the
entire event loop, parse requests and health checks included (FINDINGS M6). A
training run takes minutes to hours; it cannot live inside an HTTP request.

This is the smallest structure that fixes that: one worker thread per job, an
in-memory registry guarded by a lock, and a JSON snapshot per state change so a
finished job survives a server restart. It is not a queue -- concurrent jobs are
refused rather than queued, because two simultaneous CPU trainings on one
machine just starve each other.
"""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "queued"  # queued | running | completed | failed
    phase: str = "queued"
    created_at: str = ""
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    output_dir: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "phase": self.phase,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
            "output_dir": self.output_dir,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.snapshot() if job else None

    def running_job(self, kind: str) -> str | None:
        with self._lock:
            for job in self._jobs.values():
                if job.kind == kind and job.status in {"queued", "running"}:
                    return job.job_id
        return None

    def submit(
        self,
        kind: str,
        output_dir: Path,
        work: Callable[[Callable[[str], None]], dict[str, Any]],
    ) -> str:
        """Run ``work`` on a thread. ``work`` receives a phase-report callback
        and returns the JSON-safe result dict."""
        job = Job(
            job_id=str(uuid4()),
            kind=kind,
            created_at=datetime.now().isoformat(timespec="seconds"),
            output_dir=str(output_dir),
        )
        with self._lock:
            self._jobs[job.job_id] = job

        def set_phase(phase: str) -> None:
            with self._lock:
                job.phase = phase
            self._persist(job, output_dir)

        def run() -> None:
            with self._lock:
                job.status = "running"
                job.phase = "starting"
            self._persist(job, output_dir)
            try:
                result = work(set_phase)
                with self._lock:
                    job.status = "completed"
                    job.phase = "done"
                    job.result = result
            except Exception as exc:  # noqa: BLE001 -- the job IS the error boundary
                with self._lock:
                    job.status = "failed"
                    job.phase = "failed"
                    # str(exc) alone loses where it happened; keep a short trace tail.
                    job.error = f"{exc}\n{traceback.format_exc(limit=3)}"
            finally:
                with self._lock:
                    job.finished_at = datetime.now().isoformat(timespec="seconds")
                self._persist(job, output_dir)

        threading.Thread(target=run, name=f"job-{job.job_id[:8]}", daemon=True).start()
        return job.job_id

    def _persist(self, job: Job, output_dir: Path) -> None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "job.json").write_text(
                json.dumps(job.snapshot(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            # Persistence is best-effort; the in-memory registry stays the truth.
            pass


REGISTRY = JobRegistry()
