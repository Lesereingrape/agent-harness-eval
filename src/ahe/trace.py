"""Core trace data model and instrumentation recorder.

A `Trace` is the atomic unit of evaluation: one rollout of one agent on one
task, recorded step by step. Traces serialize to JSONL so results from
different harnesses can be pooled and compared offline.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

StepKind = Literal["llm", "tool", "result", "error", "note"]


@dataclass
class Step:
    kind: StepKind
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    t: float = 0.0
    duration: float = 0.0

    def is_error(self) -> bool:
        return self.kind == "error" or self.payload.get("ok") is False


@dataclass
class Trace:
    task_id: str
    agent_id: str = "unknown"
    harness_id: str = "unknown"
    steps: list[Step] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def wall_time(self) -> float:
        return self.finished_at - self.started_at if self.finished_at else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(
            s.payload.get("tokens_in", 0) + s.payload.get("tokens_out", 0)
            for s in self.steps
            if s.kind == "llm"
        )

    def steps_of(self, kind: StepKind) -> list[Step]:
        return [s for s in self.steps if s.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        steps = [Step(**s) for s in d.pop("steps", [])]
        return cls(steps=steps, **d)


class TraceRecorder:
    """Collects a `Trace` while an agent runs.

    Usage::

        rec = TraceRecorder("task-1", agent_id="my-agent")
        with rec.llm("plan", tokens_in=120, tokens_out=45):
            ...
        rec.tool("search", {"query": "..."}, ok=True, output="...")
        trace = rec.finish(success=True, answer="42")
    """

    def __init__(self, task_id: str, agent_id: str = "unknown", harness_id: str = "unknown"):
        self.trace = Trace(
            task_id=task_id, agent_id=agent_id, harness_id=harness_id, started_at=time.time()
        )

    def _push(self, step: Step) -> Step:
        self.trace.steps.append(step)
        return step

    def llm(self, name: str, *, tokens_in: int = 0, tokens_out: int = 0, **payload: Any):
        return _Span(self, Step(kind="llm", name=name, payload={
            "tokens_in": tokens_in, "tokens_out": tokens_out, **payload}))

    def tool(self, name: str, args: dict[str, Any] | None = None, *, ok: bool = True,
             output: Any = None, error: str | None = None, **payload: Any) -> Step:
        body: dict[str, Any] = {"args": args or {}, "ok": ok, **payload}
        if output is not None:
            body["output"] = output
        if error is not None:
            body["error"] = error
        return self._push(Step(kind="tool", name=name, payload=body, t=time.time()))

    def result(self, name: str, value: Any) -> Step:
        return self._push(Step(kind="result", name=name, payload={"value": value}))

    def error(self, name: str, message: str) -> Step:
        return self._push(Step(kind="error", name=name, payload={"error": message}))

    def note(self, name: str, **payload: Any) -> Step:
        return self._push(Step(kind="note", name=name, payload=payload))

    def finish(self, *, success: bool | None = None, **meta: Any) -> Trace:
        self.trace.finished_at = time.time()
        if success is not None:
            self.trace.meta["success"] = success
        self.trace.meta.update(meta)
        return self.trace


class _Span:
    """Context manager timing a single step."""

    def __init__(self, recorder: TraceRecorder, step: Step):
        self._recorder = recorder
        self._step = step

    def __enter__(self) -> Step:
        self._step.t = time.time()
        self._recorder._push(self._step)
        return self._step

    def __exit__(self, exc_type, exc, tb) -> None:
        self._step.duration = time.time() - self._step.t
        if exc_type is not None:
            self._step.payload["ok"] = False
            self._step.payload["error"] = str(exc)


def write_traces(path: str | Path, traces: Iterator[Trace] | list[Trace]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t.to_dict(), ensure_ascii=False) + "\n" for t in traces)


def read_traces(path: str | Path) -> list[Trace]:
    traces = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(Trace.from_dict(json.loads(line)))
    return traces
