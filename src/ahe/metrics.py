"""Process-level quality metrics derived from traces.

Outcome scores tell you *whether* an agent solved a task; these metrics tell
you *how* it behaved: did it loop, did it recover from tool errors, was it
wasteful. This is what you compare when two harnesses are both at ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .trace import Trace


@dataclass(frozen=True)
class BehaviorMetrics:
    n_steps: int
    n_llm_calls: int
    n_tool_calls: int
    n_errors: int
    error_rate: float
    redundant_call_rate: float
    recovery_rate: float
    wall_time: float
    total_tokens: int


def compute_metrics(trace: Trace) -> BehaviorMetrics:
    tools = [s for s in trace.steps if s.kind == "tool"]
    errors = [s for s in trace.steps if s.is_error()]
    failed_tools = [s for s in tools if s.payload.get("ok") is False]

    redundant = sum(
        1
        for a, b in zip(tools, tools[1:])
        if a.name == b.name and a.payload.get("args") == b.payload.get("args")
    )

    # Recovery: a failed tool call immediately followed by a successful one.
    recoverable = 0
    recovered = 0
    for prev, nxt in zip(tools, tools[1:]):
        if prev.payload.get("ok") is False:
            recoverable += 1
            if nxt.payload.get("ok") is not False:
                recovered += 1

    n_tool_calls = len(tools)
    return BehaviorMetrics(
        n_steps=len(trace.steps),
        n_llm_calls=len(trace.steps_of("llm")),
        n_tool_calls=n_tool_calls,
        n_errors=len(errors),
        error_rate=(len(failed_tools) / n_tool_calls) if n_tool_calls else 0.0,
        redundant_call_rate=(redundant / max(n_tool_calls - 1, 1)) if n_tool_calls > 1 else 0.0,
        recovery_rate=(recovered / recoverable) if recoverable else 1.0,
        wall_time=trace.wall_time,
        total_tokens=trace.total_tokens,
    )


def summarize(traces: list[Trace]) -> dict[str, float]:
    """Aggregate behavior metrics over a rollout set."""
    if not traces:
        return {}
    ms = [compute_metrics(t) for t in traces]
    keys = [f for f in BehaviorMetrics.__dataclass_fields__ if f != "n_steps"]
    out = {k: sum(getattr(m, k) for m in ms) / len(ms) for k in keys}
    out["success_rate"] = sum(
        1 for t in traces if t.meta.get("success") is True
    ) / len(traces)
    return out
