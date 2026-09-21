"""Builtin demo agents used by `ahe demo` and the test-suite.

They simulate three failure modes seen in real harnesses — wasteful looping,
unrecovered tool errors, and clean execution — so the metrics and stats
layers can be exercised (and shown off) without any API key.
"""

from __future__ import annotations

from ..suite import Task
from ..trace import TraceRecorder


def oracle(task: Task, rec: TraceRecorder):
    """Does the job in one planning step plus one tool call."""
    with rec.llm("plan", tokens_in=80, tokens_out=25):
        pass
    rec.tool("lookup", {"q": task.id}, ok=True, output=task.expected)
    rec.finish(success=True, answer=str(task.expected))


def looper(task: Task, rec: TraceRecorder):
    """Repeats the same failing tool call before eventually giving up right."""
    with rec.llm("plan", tokens_in=90, tokens_out=20):
        pass
    for _ in range(3):
        rec.tool("search", {"q": "same"}, ok=False, error="timeout")
    rec.tool("search", {"q": "same"}, ok=True, output=task.expected)
    rec.finish(success=True, answer=str(task.expected))


def flaky(task: Task, rec: TraceRecorder):
    """Hits an unrecovered error and returns the wrong answer."""
    with rec.llm("plan", tokens_in=70, tokens_out=15):
        pass
    rec.tool("calc", {"expr": task.id}, ok=False, error="domain error")
    rec.finish(success=False, answer="42")
