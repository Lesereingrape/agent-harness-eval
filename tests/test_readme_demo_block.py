"""Guard: the README's `ahe demo` transcript must be the table the CLI prints.

`ahe demo` runs three builtin mock agents over `examples/mini_qa.json` so the
Quickstart can show instrumented output without an API key. This test runs the command
in-process, byte-compares its stdout against the fenced block, and then checks the
reading of that block given in the paragraph below it - which agents solve, which
columns separate the two that do, and by how much. A demo that changes shape has to be
re-quoted rather than left as somebody's memory of a run.
"""

from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

from ahe.cli import main

ROOT = Path(__file__).resolve().parents[1]
MARKED = re.compile(r"<!-- DEMO:START -->.*?<!-- DEMO:END -->", re.DOTALL)
BLOCK = re.compile(r"<!-- DEMO:START -->\n```\n(.*?)```\n<!-- DEMO:END -->", re.DOTALL)
ROW = re.compile(r"^t\d\s+([\d.]+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+"
                 r"([\d.]+)\s+(\d+)$")

#: Parsed column -> index, for the rows below (reward, ok, steps, tools, err,
#: redundant, recovery, tokens).
REWARD, OK, STEPS, TOOLS, ERR, REDUN, RECOV, TOKENS = range(8)

#: The README spells the task count out, so the test has to spell it back.
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _captured() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["demo"]) == 0
    return buf.getvalue()


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _agents(out: str) -> dict[str, list[tuple]]:
    """The capture split into {agent_id: rows}, columns as above."""
    agents: dict[str, list[tuple]] = {}
    current = None
    for line in out.splitlines():
        head = re.fullmatch(r"== agent: (\w+)", line)
        if head:
            current = head.group(1)
            agents[current] = []
        elif current and re.match(r"t\d\s", line):
            row = ROW.fullmatch(line)
            assert row, f"unparseable demo row: {line!r}"
            agents[current].append(row.groups())
    return agents


def _suite_size() -> int:
    path = ROOT / "examples" / "mini_qa.json"
    return len(json.loads(path.read_text(encoding="utf-8"))["tasks"])


def test_readme_demo_block_is_the_cli_output():
    match = BLOCK.search(_readme())
    assert match, "README is missing the DEMO markers around a fenced block"
    assert match.group(1) == _captured(), (
        "README demo transcript no longer matches what `ahe demo` prints")


def test_the_block_shows_all_three_agents_on_the_whole_suite():
    agents = _agents(_captured())
    assert list(agents) == ["oracle", "looper", "flaky"]
    assert all(len(rows) == _suite_size() for rows in agents.values()), (
        f"the demo quoted a different task count than examples/mini_qa.json holds "
        f"({_suite_size()})")


def test_the_prose_solves_what_it_says_it_solves():
    """Two agents at 1.00 on every task, the third at 0.00 on every task."""
    agents = _agents(_captured())
    perfect = sorted(a for a, rows in agents.items()
                     if all(r[REWARD] == "1.00" and r[OK] == "True" for r in rows))
    zeroed = sorted(a for a, rows in agents.items()
                    if all(r[REWARD] == "0.00" and r[OK] == "False" for r in rows))
    assert perfect == ["looper", "oracle"], perfect
    assert zeroed == ["flaky"], zeroed
    assert f"both print 1.00 on all {_WORDS.get(_suite_size(), str(_suite_size()))} " \
           "tasks" in _readme(), (
        "the paragraph below the block no longer states the reward tie the table shows")


def test_the_looper_oracle_gap_the_prose_quotes_is_the_measured_gap():
    """The reward column cannot separate them, so the prose argues from the others."""
    agents = _agents(_captured())
    oracle, looper = agents["oracle"][0], agents["looper"][0]
    assert oracle[REWARD] == looper[REWARD], "reward now separates the two agents"
    assert looper[ERR] == "3", "the prose says three failed tool calls"
    assert looper[TOOLS] == "4", "the prose says four tool calls"
    assert looper[STEPS] == "5", "the prose says five steps"
    assert looper[REDUN] == "1.00" and oracle[REDUN] == "0.00", (
        "the prose says the looper's redun column is 1.00 against the oracle's 0.00")
    assert looper[TOKENS] == "110" and oracle[TOKENS] == "105", (
        "the prose quotes 110 tokens against the oracle's 105")
    for claim in ("three failed", "4 tool calls", "5 steps", "110 tokens", "105"):
        assert claim in _readme(), f"the paragraph no longer states {claim!r}"
