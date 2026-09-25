"""Guard: the README's compare output must be something the shipped code prints.

`ahe demo --paired` generates two mock harnesses over the same 120 tasks and
`ahe compare` summarises them; the README quotes that summary. This test runs both
commands in-process and byte-compares the result against the fenced block, so a change
to the harnesses, the suite or the statistics updates the docs or fails loudly - it
cannot leave a sample output behind that no code produces.
"""

from __future__ import annotations

import re
from pathlib import Path

from ahe.cli import main
from ahe.demo.paired import N_TASKS

ROOT = Path(__file__).resolve().parents[1]
MARKED = re.compile(r"<!-- COMPARE:START -->.*?<!-- COMPARE:END -->", re.DOTALL)
BLOCK = re.compile(r"<!-- COMPARE:START -->\n```\n(.*?)```\n<!-- COMPARE:END -->", re.DOTALL)


def _measured(tmp_path: Path, capsys) -> str:
    """Run the two commands the README tells readers to run, and return the summary."""
    assert main(["demo", "--paired", "--out-dir", str(tmp_path)]) == 0
    capsys.readouterr()  # drop the "wrote ..." line; only compare's stdout is quoted
    assert main(["compare", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]) == 0
    return capsys.readouterr().out


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _prose() -> str:
    """The README with the measured block removed, i.e. the hand-written sentences."""
    return MARKED.sub("", _readme())


def _field(out: str, pattern: str) -> re.Match:
    match = re.search(pattern, out)
    assert match, f"the compare output no longer looks like {pattern!r}"
    return match


def test_readme_compare_block_is_the_cli_output(tmp_path, capsys):
    match = BLOCK.search(_readme())
    assert match, "README is missing the COMPARE markers around a fenced block"
    assert match.group(1) == _measured(tmp_path, capsys), (
        "README compare output no longer matches `ahe demo --paired && ahe compare`")


def test_quoted_task_count_is_the_suite_that_ran(tmp_path, capsys):
    assert f"tasks compared : {N_TASKS}" in _measured(tmp_path, capsys)


def test_every_point_gain_in_the_prose_is_the_measured_delta(tmp_path, capsys):
    """The prose argues from one number, so that number must come from the block."""
    line = _field(_measured(tmp_path, capsys), r"delta=([-+0-9.]+)").group(1)
    gain = f"{abs(float(line)) * 100:.1f}"
    quoted = set(re.findall(r"([0-9.]+)-point gain", _prose()))
    assert quoted == {gain}, f"the README argues from {sorted(quoted)}, the run gives {gain}"


def test_the_prose_claimed_significance_holds_in_the_block(tmp_path, capsys):
    """The README reads the block as 'the interval clears zero and the exact test agrees'."""
    out = _measured(tmp_path, capsys)
    lo = float(_field(out, r"95% CI=\[([-+0-9.]+),").group(1))
    hi = float(_field(out, r"95% CI=\[[-+0-9.]+, ([-+0-9.]+)\]").group(1))
    p = float(_field(out, r"p=([0-9.]+)").group(1))
    assert min(lo, hi) > 0 or max(lo, hi) < 0, "the CI spans zero, so it does not 'clear' it"
    assert p < 0.05, "the prose says the exact paired test lands at p < 0.05"
