# ahe — Agent Harness Eval

> Instrument, score, and **statistically compare** agent rollouts.

`ahe` is a dependency-free evaluation harness for LLM agents. It records every
rollout as a structured **trace**, computes outcome rewards *and* behavior
metrics (loops, tool-error recovery, token/latency cost), and tells you
whether harness A genuinely beats harness B — with paired bootstrap CIs and
McNemar's exact test, not a hand-wavy average over 12 tasks.

**Why this exists.** Most agent eval tools report pass@1 and stop there. Two
harnesses both at ceiling look identical, yet one may be looping twice as
much, silently swallowing tool errors, or burning 3× the tokens. `ahe` makes
the *process* measurable, and treats harness comparison as the paired
statistical experiment it actually is.

```
tasks.json ──► Runner ──► TraceRecorder ──► traces.jsonl ──► report / compare
                │             │                                │
         your agent fn   steps: llm/tool/error            success ± CI
         + scorers        reward + metrics                McNemar p-value
```

![ci](https://github.com/Lesereingrape/agent-harness-eval/actions/workflows/ci.yml/badge.svg)

## Quickstart

```bash
pip install ahe            # pure stdlib core, Python >= 3.10
ahe demo                   # zero-config tour with builtin mock agents
```

```
== agent: looper
task   reward   ok  steps  tools  err  redun  recov  tokens
t1       1.00 True      5      4    3   1.00   0.33     110   # right answer, wasteful loop
```

All three demo agents finish with a usable answer, but `report` and `compare`
expose the difference. That is the point.

## Instrument your agent

```python
from ahe import Suite, Runner, Task, TraceRecorder

def my_agent(task: Task, rec: TraceRecorder) -> None:
    with rec.llm("plan", tokens_in=120, tokens_out=48):
        plan = call_llm(task.prompt)
    for call in plan:
        rec.tool(call.name, call.args, ok=result.ok, output=result.text)
    rec.finish(success=verified, answer=final)

suite = Suite.load("examples/mini_qa.json")
rollouts = Runner(suite, my_agent, harness_id="v3").run()
```

## Score outcomes

Built-in scorers: `exact`, `regex`, `json_subset` (contract verification for
structured outputs). Any `callable(task, trace) -> float` also works. For
open-ended tasks, attach a judge:

```python
from ahe import HeuristicJudge              # deterministic, CI-safe
from ahe.judge import OpenAIJudge           # rubric grading via any
judge = OpenAIJudge(model="gpt-4.1-mini")   # OpenAI-compatible endpoint
```

## Compare two harnesses

```bash
ahe run --suite suite.json --agent pkg.claude_code:rollout --out a.jsonl
ahe run --suite suite.json --agent pkg.openclaw:rollout    --out b.jsonl
ahe compare a.jsonl b.jsonl
```

Here is that command run for real, on the mock pair the library ships for the purpose:
`ahe demo --paired` writes 120 paired traces of two harnesses that differ in exactly one
mechanism (B rephrases and retries a failed lookup once, A does not), and the block below
is what `ahe compare` printed afterwards. A test regenerates it, so it cannot rot into a
sample output that no code produces:

<!-- COMPARE:START -->
```
tasks compared : 120
A success rate : 0.750
B success rate : 0.808
bootstrap      : delta=+0.058 95% CI=[+0.017, +0.100] P(B>A)=0.999
mcnemar exact  : p=0.0156
```
<!-- COMPARE:END -->

A 5.8-point gain whose bootstrap interval clears zero and whose exact paired test lands
at p < 0.05 is a real gain; a 5.8-point gain with `95% CI=[-0.01, +0.12]` is a coin flip
you happened to win. Reproduce it:

```bash
ahe demo --paired --out-dir /tmp/paired
ahe compare /tmp/paired/a.jsonl /tmp/paired/b.jsonl
```

## Metrics computed per trace

| Metric | Meaning |
|---|---|
| `reward` | task-level outcome score in [0, 1] |
| `error_rate` | failed tool calls / all tool calls |
| `redundant_call_rate` | identical consecutive tool calls (loop detector) |
| `recovery_rate` | failed calls followed by a successful retry |
| `total_tokens`, `wall_time` | cost and latency |

## Design notes

- **Zero required dependencies.** The core is stdlib-only so it can run inside
  sandboxes, CI jobs, and eval workers where you cannot pip-install the world.
  `pip install ahe[http]` only when you want the hosted judge.
- **JSONL as the interchange format.** Traces from different harnesses,
  languages, and machines pool into one file; every analysis is offline.
- **Statistics over vibes.** Paired bootstrap resamples task-level deltas;
  McNemar's exact test is the right test for paired binary outcomes.

## Roadmap

- [ ] Adapter pack: trace importers for OpenClaw / Claude Code / LangGraph event logs
- [ ] Cost-weighted pass@k and trajectory-similarity clustering
- [ ] Multi-seed variance analysis (which tasks are seed-fragile)
- [ ] HTML report renderer with per-task drill-down

## Development

```bash
pip install -e .[dev]
pytest
ahe demo
```

## License

Apache-2.0
