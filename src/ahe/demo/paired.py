"""A paired mock-harness comparison that the README can quote honestly.

`ahe compare` needs two trace files, and the docs used to show output that nothing in
the repo could regenerate. This module supplies the missing half: a 120-task mock suite
and two harnesses that differ in exactly one mechanism - harness B retries a failed
lookup once, harness A does not. Their per-task outcomes come from a checksum of the
task id (`zlib.crc32`, not Python's salted `hash`), so the comparison is stable across
interpreters and runs, and `ahe demo --paired` + `ahe compare` print the real thing.

The success rates are properties of this mock, not a finding about any real agent; what
transfers is the reading rule in the README - a delta only counts if the paired CI and
McNemar's exact test both clear zero.
"""

from __future__ import annotations

import zlib

from ..suite import Suite, Task
from ..trace import TraceRecorder

N_TASKS = 120
#: below this, the lookup is hard for both harnesses
HARD_BELOW = 30
#: between HARD_BELOW and this, only the harness that retries gets through
FLAKY_BELOW = 38


def _load(task_id: str) -> int:
    """Deterministic 0-99 'difficulty roll' for a task id, stable across processes."""
    return zlib.crc32(task_id.encode("utf-8")) % 100


def paired_suite(n: int = N_TASKS) -> Suite:
    return Suite(
        name="paired-mock",
        tasks=[Task(id=f"q{i:03d}", prompt=f"look up {i}", expected=str(i % 7))
               for i in range(n)],
    )


def harness_a(task: Task, rec: TraceRecorder) -> None:
    """Plan, one lookup, done - a failed lookup is a failed task."""
    with rec.llm("plan", tokens_in=80, tokens_out=25):
        pass
    ok = _load(task.id) >= FLAKY_BELOW
    rec.tool("lookup", {"q": task.id}, ok=ok,
             **({"output": str(task.expected)} if ok else {"error": "timeout"}))
    rec.finish(success=ok, answer=str(task.expected))


def harness_b(task: Task, rec: TraceRecorder) -> None:
    """Identical plan, plus one rephrased retry when the first lookup fails."""
    load = _load(task.id)
    with rec.llm("plan", tokens_in=80, tokens_out=25):
        pass
    first_ok = load >= FLAKY_BELOW
    rec.tool("lookup", {"q": task.id}, ok=first_ok,
             **({"output": str(task.expected)} if first_ok else {"error": "timeout"}))
    if first_ok:
        rec.finish(success=True, answer=str(task.expected))
        return
    retry_ok = load >= HARD_BELOW
    rec.tool("lookup", {"q": f"{task.id} rephrased"}, ok=retry_ok,
             **({"output": str(task.expected)} if retry_ok else {"error": "timeout"}))
    rec.finish(success=retry_ok, answer=str(task.expected))
