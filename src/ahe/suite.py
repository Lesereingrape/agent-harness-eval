"""Task suites and the rollout runner.

A suite is a JSON file: {{"name": ..., "tasks": [{{"id", "prompt", "expected",
"scorer", "criteria"}}]}}. The runner drives an agent callable over every task,
capturing a `Trace` per rollout and scoring it with the task's scorer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .judge import Judge, JudgeVerdict
from .metrics import BehaviorMetrics, compute_metrics
from .scorers import scorer_for
from .trace import Trace, TraceRecorder


@dataclass
class Task:
    id: str
    prompt: str
    expected: str | dict | list = ""
    scorer: str = "exact"
    scorer_args: dict = field(default_factory=dict)
    criteria: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        allowed = set(Task.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class Suite:
    name: str
    tasks: list[Task]

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(name=d.get("name", Path(path).stem),
                   tasks=[Task.from_dict(t) for t in d["tasks"]])


# An agent takes a Task and a fresh recorder, and returns nothing meaningful;
# it drives the recorder and calls rec.finish(...) internally.
AgentFn = Callable[[Task, TraceRecorder], None]


@dataclass
class Rollout:
    task: Task
    trace: Trace
    reward: float
    metrics: BehaviorMetrics
    verdict: JudgeVerdict | None = None


class Runner:
    def __init__(self, suite: Suite, agent: AgentFn, *,
                 agent_id: str = "agent", harness_id: str = "harness",
                 judge: Judge | None = None, retries: int = 0):
        self.suite = suite
        self.agent = agent
        self.agent_id = agent_id
        self.harness_id = harness_id
        self.judge = judge
        self.retries = retries

    def run(self) -> list[Rollout]:
        return [self._one(t) for t in self.suite.tasks]

    def _one(self, task: Task) -> Rollout:
        rec = TraceRecorder(task.id, agent_id=self.agent_id, harness_id=self.harness_id)
        last_exc: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                self.agent(task, rec)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
        trace = rec.trace
        if last_exc is not None:
            rec.error("agent_exception", str(last_exc))
            trace = rec.finish(success=False)
        if "success" not in trace.meta:
            trace.meta["success"] = None
        reward = scorer_for(task.__dict__)(task.__dict__, trace)
        verdict = self.judge.grade(task.criteria, trace) if (self.judge and task.criteria) else None
        return Rollout(task=task, trace=trace, reward=reward,
                       metrics=compute_metrics(trace), verdict=verdict)
