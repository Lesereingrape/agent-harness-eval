"""LLM-as-judge scoring with a deterministic fallback.

Judges are pluggable: any object with `grade(criteria, answer, transcript)`
returning a `JudgeVerdict`. `HeuristicJudge` is dependency-free and fully
deterministic, which keeps CI green and tests honest; `OpenAIJudge` talks to
any OpenAI-compatible endpoint when you want model-based rubric grading.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .trace import Trace


@dataclass(frozen=True)
class JudgeVerdict:
    score: float  # in [0, 1]
    label: str  # e.g. "pass" / "fail" / "partial"
    reason: str


class Judge(Protocol):
    def grade(self, criteria: str, trace: Trace) -> JudgeVerdict: ...


class HeuristicJudge:
    """Deterministic surrogate for offline evaluation and tests.

    Grades success-of-trace plus a penalty for errors in the transcript —
    useful as a baseline and to keep the pipeline testable without network.
    """

    def grade(self, criteria: str, trace: Trace) -> JudgeVerdict:
        success = trace.meta.get("success")
        n_err = sum(1 for s in trace.steps if s.is_error())
        score = (1.0 if success else 0.0) - min(0.5, 0.1 * n_err)
        score = max(0.0, min(1.0, score))
        label = "pass" if score >= 0.8 else ("partial" if score > 0 else "fail")
        return JudgeVerdict(score, label, f"heuristic: success={success}, errors={n_err}")


_JUDGE_PROMPT = """You are an expert grader for AI agent rollouts.

Criteria:
{criteria}

Transcript ( condensed steps ):
{transcript}

Reply with ONLY a JSON object: {{"score": <float 0..1>, "label": "pass|partial|fail", "reason": "<one sentence>"}}"""


class OpenAIJudge:
    """Rubric judge against any OpenAI-compatible /chat/completions endpoint.

    Requires the `http` extra (`pip install ahe[http]`) and an API key in the
    environment; never import it at module top-level elsewhere.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_base: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0):
        import os

        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("OpenAIJudge needs `pip install ahe[http]`") from e
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"missing API key in ${api_key_env}")
        self._http = httpx.Client(
            base_url=api_base, timeout=60, headers={"Authorization": f"Bearer {key}"}
        )
        self.model = model
        self.temperature = temperature

    def grade(self, criteria: str, trace: Trace) -> JudgeVerdict:
        prompt = _JUDGE_PROMPT.format(criteria=criteria, transcript=_condense(trace))
        resp = self._http.post(
            "/chat/completions",
            json={"model": self.model, "temperature": self.temperature,
                  "messages": [{"role": "user", "content": prompt}]},
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_verdict(content)


def _condense(trace: Trace, max_steps: int = 40) -> str:
    lines = []
    for s in trace.steps[:max_steps]:
        ok = "" if s.payload.get("ok", True) else " [FAILED]"
        lines.append(f"- {s.kind}:{s.name}{ok}")
    lines.append(f"- outcome: success={trace.meta.get('success')} answer={trace.meta.get('answer')!r}")
    return "\n".join(lines)


def _parse_verdict(text: str) -> JudgeVerdict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return JudgeVerdict(0.0, "fail", f"judge returned unparsable output: {text[:120]!r}")
    d = json.loads(m.group(0))
    return JudgeVerdict(float(d.get("score", 0.0)), str(d.get("label", "fail")),
                        str(d.get("reason", "")))
