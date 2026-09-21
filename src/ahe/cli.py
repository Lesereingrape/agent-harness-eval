"""Command-line interface: run, compare, report, demo."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .judge import HeuristicJudge
from .stats import mcnemar_exact, paired_bootstrap
from .suite import AgentFn, Runner, Suite
from .trace import Trace, read_traces, write_traces


def _load_agent(spec: str) -> AgentFn:
    module, _, fn = spec.partition(":")
    obj = importlib.import_module(module)
    return getattr(obj, fn) if fn else obj.default


def cmd_run(args: argparse.Namespace) -> int:
    suite = Suite.load(args.suite)
    runner = Runner(suite, _load_agent(args.agent), agent_id=args.agent_id,
                    harness_id=args.harness_id,
                    judge=HeuristicJudge() if args.judge else None)
    rollouts = runner.run()
    write_traces(args.out, (r.trace for r in rollouts))
    _print_table([(r.task.id, r.reward, r.metrics, r.trace.meta.get("success")) for r in rollouts])
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .metrics import summarize

    traces = read_traces(args.traces)
    doc = {
        "per_harness": _per_harness(traces),
        "overall": summarize(traces),
        "n_traces": len(traces),
    }
    print(json.dumps(doc, indent=2, sort_keys=True))
    return 0


def _per_harness(traces: list[Trace]) -> dict:
    groups: dict[str, list[Trace]] = {}
    for t in traces:
        groups.setdefault(t.harness_id, []).append(t)
    from .metrics import summarize

    return {k: summarize(v) for k, v in groups.items()}


def cmd_compare(args: argparse.Namespace) -> int:
    """Paired comparison of two trace files on the same task ids."""
    a = {t.task_id: t for t in read_traces(args.a)}
    b = {t.task_id: t for t in read_traces(args.b)}
    common = sorted(set(a) & set(b))
    if not common:
        print("no shared task ids between the two trace files", file=sys.stderr)
        return 2
    sa = [1.0 if a[t].meta.get("success") else 0.0 for t in common]
    sb = [1.0 if b[t].meta.get("success") else 0.0 for t in common]
    res = paired_bootstrap(sa, sb, n_resamples=args.resamples, seed=args.seed)
    pa, pb = zip(*[(bool(a[t].meta.get("success")), bool(b[t].meta.get("success"))) for t in common], strict=False)
    print(f"tasks compared : {len(common)}")
    print(f"A success rate : {sum(sa)/len(sa):.3f}")
    print(f"B success rate : {sum(sb)/len(sb):.3f}")
    print(f"bootstrap      : {res.summary()}")
    print(f"mcnemar exact  : p={mcnemar_exact(list(pa), list(pb)):.4f}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    here = Path(__file__).resolve()
    suite = Suite.load(here.parents[2] / "examples" / "mini_qa.json")
    from .demo import flaky, looper, oracle

    for agent_id, agent in [("oracle", oracle), ("looper", looper), ("flaky", flaky)]:
        rollouts = Runner(suite, agent, agent_id=agent_id, harness_id="demo",
                          judge=HeuristicJudge()).run()
        print(f"\n== agent: {agent_id}")
        _print_table([(r.task.id, r.reward, r.metrics, r.trace.meta.get("success")) for r in rollouts])
    return 0


def _print_table(rows) -> None:
    print(f"{'task':<6}{'reward':>7}{'ok':>5}{'steps':>7}{'tools':>7}{'err':>5}{'redun':>7}{'recov':>7}{'tokens':>8}")
    for task_id, reward, m, ok in rows:
        print(f"{task_id:<6}{reward:>7.2f}{bool(ok)!s:>5}{m.n_steps:>7}{m.n_tool_calls:>7}"
              f"{m.n_errors:>5}{m.redundant_call_rate:>7.2f}{m.recovery_rate:>7.2f}{m.total_tokens:>8}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ahe", description="Agent Harness Eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an agent over a task suite")
    r.add_argument("--suite", required=True)
    r.add_argument("--agent", required=True, help="import spec, e.g. mypkg.myagent:run")
    r.add_argument("--agent-id", default="agent")
    r.add_argument("--harness-id", default="harness")
    r.add_argument("--judge", action="store_true", help="attach heuristic rubric judge")
    r.add_argument("--out", default="traces.jsonl")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="paired statistical comparison of two trace files")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--resamples", type=int, default=5000)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=cmd_compare)

    rp = sub.add_parser("report", help="aggregate metrics from a traces.jsonl")
    rp.add_argument("traces")
    rp.set_defaults(fn=cmd_report)

    d = sub.add_parser("demo", help="run builtin demo agents on examples/mini_qa.json")
    d.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
