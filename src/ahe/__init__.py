"""ahe — Agent Harness Eval.

Instrument agent rollouts as traces, score outcomes and behavior, and
compare harnesses with paired statistics.
"""

from .judge import HeuristicJudge, JudgeVerdict
from .metrics import BehaviorMetrics, compute_metrics, summarize
from .scorers import exact_match, json_subset_match, regex_match
from .suite import Rollout, Runner, Suite, Task
from .trace import Step, Trace, TraceRecorder, read_traces, write_traces

__version__ = "0.1.0"

__all__ = [
    "BehaviorMetrics", "HeuristicJudge", "JudgeVerdict", "Rollout", "Runner",
    "Step", "Suite", "Task", "Trace", "TraceRecorder", "compute_metrics",
    "exact_match", "json_subset_match", "read_traces", "regex_match",
    "summarize", "write_traces",
]
