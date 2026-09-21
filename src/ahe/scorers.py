"""Outcome scorers: map a (task, trace) pair to a scalar reward in [0, 1].

Scorers are plain callables so users can supply their own; the built-ins
cover the common contract-verification cases (exact match, regex, JSON
subset match) without pulling in heavy dependencies.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from .trace import Trace


def final_answer(trace: Trace) -> str:
    value = trace.meta.get("answer")
    return str(value) if value is not None else ""


Scorer = Callable[[dict, Trace], float]


def exact_match(normalize: bool = True) -> Scorer:
    def score(task: dict, trace: Trace) -> float:
        got, want = final_answer(trace), str(task.get("expected", ""))
        if normalize:
            got, want = " ".join(got.split()).lower(), " ".join(want.split()).lower()
        return 1.0 if got == want else 0.0

    return score


def regex_match(pattern: str) -> Scorer:
    rx = re.compile(pattern)

    def score(task: dict, trace: Trace) -> float:
        return 1.0 if rx.search(final_answer(trace)) else 0.0

    return score


def json_subset_match() -> Scorer:
    """1.0 iff the answer parses as JSON and contains all expected fields/values."""

    def score(task: dict, trace: Trace) -> float:
        try:
            got = json.loads(final_answer(trace))
        except (json.JSONDecodeError, ValueError):
            return 0.0
        want = task.get("expected")
        if isinstance(want, str):
            try:
                want = json.loads(want)
            except json.JSONDecodeError:
                return 0.0
        return 1.0 if _subset(want, got) else 0.0

    return score


def _subset(want, got) -> bool:
    if isinstance(want, dict):
        return isinstance(got, dict) and all(
            k in got and _subset(v, got[k]) for k, v in want.items()
        )
    if isinstance(want, list):
        return isinstance(got, list) and len(want) == len(got) and all(
            _subset(w, g) for w, g in zip(want, got, strict=False)
        )
    return want == got


def scorer_for(task: dict) -> Scorer:
    kind = task.get("scorer", "exact")
    if kind == "exact":
        return exact_match()
    if kind == "json_subset":
        return json_subset_match()
    if kind == "regex":
        return regex_match(task["scorer_args"]["pattern"])
    raise ValueError(f"unknown scorer: {kind!r}")
