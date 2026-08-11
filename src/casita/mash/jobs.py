"""Per-reviewer FIFO queue for mash preference jobs (memo + rank).

One worker thread drains jobs serially per reviewer so memo/rank writes never race.
Phases let the UI advance after memo_ready while rank continues.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PrefJob:
    reviewer: str
    left_key: str
    right_key: str
    winner: str | None
    skipped: bool
    reason: str | None
    left_meta: dict | None = None
    right_meta: dict | None = None


@dataclass
class PrefStatus:
    status: str = "idle"  # idle | running | error
    phase: str = "idle"  # idle | memo | rank | error
    memo_ready: bool = False
    rank_ready: bool = True
    last_error: str | None = None
    updated_at: float = field(default_factory=time.time)
    pending: int = 0


ProgressFn = Callable[[str, dict[str, Any]], None]
# progress(reviewer, {"phase", "memo_ready", "rank_ready", "last_error"?})


def memo_gate_satisfied(st: dict[str, Any]) -> bool:
    """Play may reload once memo is ready (rank may still be running)."""
    if st.get("memo_ready"):
        return True
    phase = st.get("phase") or "idle"
    if phase in ("rank", "idle", "error"):
        return True
    status = st.get("status") or "idle"
    return status in ("idle", "error")


class PrefJobQueue:
    def __init__(self, worker_fn: Callable[[PrefJob, ProgressFn], None]):
        self._worker_fn = worker_fn
        self._queues: dict[str, deque[PrefJob]] = defaultdict(deque)
        self._status: dict[str, PrefStatus] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._run, name="mash-pref-queue", daemon=True)
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread.start()

    def enqueue(self, job: PrefJob) -> None:
        self.start()
        with self._cv:
            self._queues[job.reviewer].append(job)
            st = self._status.setdefault(job.reviewer, PrefStatus())
            st.pending = len(self._queues[job.reviewer]) + (1 if st.status == "running" else 0)
            st.memo_ready = False
            st.rank_ready = False
            st.updated_at = time.time()
            self._cv.notify()

    def status(self, reviewer: str) -> dict[str, Any]:
        with self._lock:
            st = self._status.get(reviewer) or PrefStatus()
            pending = len(self._queues.get(reviewer, ()))
            if st.status == "running":
                pending += 1
            status = st.status
            if status == "idle" and pending > 0:
                status = "running"
            phase = st.phase
            if status == "running" and phase == "idle":
                phase = "memo"
            return {
                "status": status,
                "phase": phase,
                "memo_ready": bool(st.memo_ready),
                "rank_ready": bool(st.rank_ready),
                "last_error": st.last_error,
                "updated_at": st.updated_at,
                "pending": pending,
            }

    def _set_progress(self, reviewer: str, patch: dict[str, Any]) -> None:
        with self._lock:
            st = self._status.setdefault(reviewer, PrefStatus())
            if "phase" in patch:
                st.phase = patch["phase"]
            if "memo_ready" in patch:
                st.memo_ready = bool(patch["memo_ready"])
            if "rank_ready" in patch:
                st.rank_ready = bool(patch["rank_ready"])
            if "last_error" in patch:
                st.last_error = patch["last_error"]
            st.updated_at = time.time()

    def _run(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                continue
            reviewer = job.reviewer
            with self._lock:
                st = self._status.setdefault(reviewer, PrefStatus())
                st.status = "running"
                st.phase = "memo"
                st.memo_ready = False
                st.rank_ready = False
                st.last_error = None
                st.updated_at = time.time()
                st.pending = len(self._queues[reviewer]) + 1

            def progress(_rev: str, patch: dict[str, Any], *, _r=reviewer) -> None:
                self._set_progress(_r, patch)

            try:
                self._worker_fn(job, progress)
                with self._lock:
                    st = self._status.setdefault(reviewer, PrefStatus())
                    st.status = "idle"
                    st.phase = "idle"
                    st.memo_ready = True
                    st.rank_ready = True
                    st.last_error = None
                    st.updated_at = time.time()
                    st.pending = len(self._queues[reviewer])
            except Exception as e:
                err = str(e)[:240]
                print(f"  mash pref job err [{reviewer}]: {err}", flush=True)
                with self._lock:
                    st = self._status.setdefault(reviewer, PrefStatus())
                    st.status = "error"
                    st.phase = "error"
                    st.last_error = err
                    st.updated_at = time.time()
                    st.pending = len(self._queues[reviewer])
                    # Unblock play if memo never landed.
                    if not st.memo_ready:
                        st.memo_ready = True
                    if st.pending == 0:
                        st.status = "idle"
                        st.phase = "idle"

    def _next_job(self) -> PrefJob | None:
        with self._cv:
            while True:
                for reviewer, q in list(self._queues.items()):
                    if q:
                        return q.popleft()
                self._cv.wait(timeout=1.0)
