import json

from ahe.trace import TraceRecorder, read_traces, write_traces


def test_recorder_roundtrip(tmp_path):
    rec = TraceRecorder("t1", agent_id="a", harness_id="h")
    with rec.llm("plan", tokens_in=10, tokens_out=5):
        pass
    rec.tool("search", {"q": "x"}, ok=True, output="y")
    trace = rec.finish(success=True, answer="y")

    assert trace.total_tokens == 15
    assert len(trace.steps_of("tool")) == 1

    path = tmp_path / "traces.jsonl"
    write_traces(path, [trace])
    (loaded,) = read_traces(path)
    assert loaded.task_id == "t1"
    assert loaded.meta["success"] is True
    assert loaded.steps[1].payload["args"] == {"q": "x"}


def test_span_marks_exception_as_error():
    rec = TraceRecorder("t2")
    try:
        with rec.llm("plan"):
            raise ValueError("boom")
    except ValueError:
        pass
    step = rec.trace.steps[0]
    assert step.payload["ok"] is False
    assert "boom" in step.payload["error"]


def test_jsonl_is_utf8_safe(tmp_path):
    rec = TraceRecorder("任务-1")
    trace = rec.finish(success=False, answer="答案")
    path = tmp_path / "t.jsonl"
    write_traces(path, [trace])
    raw = path.read_text(encoding="utf-8")
    assert "任务-1" in raw  # ensure_ascii=False
    assert json.loads(raw)["meta"]["answer"] == "答案"
