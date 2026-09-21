from ahe.metrics import compute_metrics
from ahe.trace import TraceRecorder


def _looping_trace():
    rec = TraceRecorder("t")
    rec.tool("search", {"q": "a"}, ok=False, error="timeout")
    rec.tool("search", {"q": "a"}, ok=False, error="timeout")
    rec.tool("search", {"q": "a"}, ok=True, output="hit")
    rec.tool("calc", {"x": 1}, ok=True)
    return rec.finish(success=True)


def _clean_trace():
    rec = TraceRecorder("t")
    rec.tool("search", {"q": "a"}, ok=True, output="hit")
    rec.tool("calc", {"x": 1}, ok=True)
    return rec.finish(success=True)


def test_looping_trace_flags_redundancy_and_recovery():
    m = compute_metrics(_looping_trace())
    assert m.n_tool_calls == 4
    assert m.error_rate > 0
    assert m.redundant_call_rate > 0
    # two failures, both followed by a later call; only the second failure is
    # immediately followed by a success -> recovery_rate = 0.5
    assert m.recovery_rate == 0.5


def test_clean_trace_is_healthy():
    m = compute_metrics(_clean_trace())
    assert m.error_rate == 0.0
    assert m.redundant_call_rate == 0.0
    assert m.recovery_rate == 1.0
