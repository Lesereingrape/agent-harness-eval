from ahe.scorers import exact_match, json_subset_match, regex_match
from ahe.trace import TraceRecorder


def _trace_with(answer):
    rec = TraceRecorder("t")
    return rec.finish(success=True, answer=answer)


def test_exact_match_normalizes_case_and_whitespace():
    s = exact_match()
    assert s({"expected": "  Paris "}, _trace_with("paris")) == 1.0
    assert s({"expected": "Paris"}, _trace_with("London")) == 0.0


def test_regex_match():
    s = regex_match(r"\d+kg")
    assert s({}, _trace_with("the result is 12 kg")) == 0.0  # no space allowed by pattern
    assert s({}, _trace_with("the result is 12kg")) == 1.0


def test_json_subset_match():
    s = json_subset_match()
    trace = _trace_with('{"a": 1, "b": {"c": 2, "d": 3}}')
    assert s({"expected": '{"b": {"c": 2}}'}, trace) == 1.0
    assert s({"expected": {"b": {"c": 99}}}, trace) == 0.0
    assert s({"expected": {"a": 1}}, _trace_with("not json")) == 0.0
