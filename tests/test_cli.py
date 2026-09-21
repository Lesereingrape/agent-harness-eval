import json
from pathlib import Path

from ahe.cli import main
from ahe.demo import flaky, looper, oracle
from ahe.suite import Runner, Suite, Task
from ahe.trace import TraceRecorder, read_traces

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runner_scores_builtin_agents():
    suite = Suite.load(REPO_ROOT / "examples" / "mini_qa.json")
    for agent, fn, expect_reward in [(oracle, oracle, 1.0), (flaky, flaky, 0.0)]:
        rollouts = Runner(suite, fn, judge=None).run()
        assert all(r.reward == expect_reward for r in rollouts), agent
    (rollout,) = Runner(Suite("x", [Task(id="t", prompt="p", expected="ok")]), looper).run()
    assert rollout.reward == 1.0
    assert rollout.metrics.redundant_call_rate > 0


def test_agent_exception_becomes_failed_rollout():
    def crashy(task, rec: TraceRecorder):
        raise RuntimeError("agent exploded")

    suite = Suite("x", [Task(id="t1", prompt="p", expected="anything")])
    (r,) = Runner(suite, crashy).run()
    assert r.reward == 0.0
    assert r.trace.meta["success"] is False


def test_cli_demo_and_compare(tmp_path, capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "oracle" in out and "flaky" in out

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    suite = Suite.load(REPO_ROOT / "examples" / "mini_qa.json")
    from ahe.trace import write_traces
    write_traces(a, (r.trace for r in Runner(suite, oracle, harness_id="A").run()))
    write_traces(b, (r.trace for r in Runner(suite, flaky, harness_id="B").run()))

    assert main(["compare", str(a), str(b), "--resamples", "500"]) == 0
    out = capsys.readouterr().out
    assert "mcnemar" in out
    assert "bootstrap" in out
    assert len(read_traces(a)) == 5


def test_cli_report_per_harness(tmp_path, capsys):
    suite = Suite.load(REPO_ROOT / "examples" / "mini_qa.json")
    from ahe.trace import write_traces
    out_file = tmp_path / "mixed.jsonl"
    traces = [r.trace for r in Runner(suite, oracle, harness_id="A").run()]
    traces += [r.trace for r in Runner(suite, flaky, harness_id="B").run()]
    write_traces(out_file, traces)

    assert main(["report", str(out_file)]) == 0
    printed = capsys.readouterr().out
    doc = json.loads(printed)
    assert doc["per_harness"]["A"]["success_rate"] == 1.0
    assert doc["per_harness"]["B"]["success_rate"] == 0.0
    assert doc["n_traces"] == 10
