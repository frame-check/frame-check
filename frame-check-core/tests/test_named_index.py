"""Frames whose index may be named: keys may name index levels."""

import pytest
from frame_check_core.checker import Checker

PLAIN = 'import pandas as pd\ndf = pd.DataFrame({"a": [1], "k": [2]})\n'
NAMED = (
    "import pandas as pd\n"
    'idx = pd.Index([0], name="lvl")\n'
    'df = pd.DataFrame({"a": [1], "k": [2]}, index=idx)\n'
)


@pytest.mark.parametrize(
    "call",
    [
        'df.groupby("lvl").sum()',
        'df.sort_values("lvl")',
        'df.merge(df, on="lvl")',
        'df.join(df[["a"]], on="lvl", rsuffix="_r")',
        'pd.merge(df, df, on="lvl")',
    ],
)
def test_index_level_keys_are_not_reported_on_named_index(call: str):
    fc = Checker.check(NAMED + f"x = {call}\n")
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    "call",
    [
        'df.groupby("lvl").sum()',
        'df.sort_values("lvl")',
        'df.merge(df, on="lvl")',
    ],
)
def test_missing_keys_are_still_reported_on_plain_frames(call: str):
    fc = Checker.check(PLAIN + f"x = {call}\n")
    assert len(fc.diagnostics) == 1
    assert "column 'lvl' does not exist" in fc.diagnostics[0].message


@pytest.mark.parametrize(
    "setup",
    [
        'df = df.set_index("k")\n',
        'df.index.name = "lvl"\n',
        "df.index = other_index\n",
        'df = pd.read_csv("f.csv", usecols=["a", "k"], index_col="k")\n',
        "df = pd.DataFrame({'a': [1], 'k': [2]}, other_index)\n",
    ],
)
def test_index_may_become_named(setup: str):
    fc = Checker.check(PLAIN + setup + 'x = df.groupby("lvl").sum()\n')
    assert fc.diagnostics == []


def test_derived_frames_inherit_a_named_index():
    fc = Checker.check(NAMED + 'df2 = df.assign(b=1)\nx = df2.groupby("lvl").sum()\n')
    assert fc.diagnostics == []


def test_literal_or_range_index_is_unnamed():
    code = (
        "import pandas as pd\n"
        'df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])\n'
        'df2 = pd.DataFrame({"a": [1, 2]}, index=range(2))\n'
        'x = df.groupby("lvl").sum()\n'
        'y = df2.groupby("lvl").sum()\n'
    )
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 2


def test_reset_index_on_named_index_is_unknown():
    """The new columns are named after index levels we don't know."""
    fc = Checker.check(NAMED + 'out = df.reset_index()\nout["lvl"]\n')
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


def test_reset_index_on_plain_frame_adds_index():
    fc = Checker.check(PLAIN + 'out = df.reset_index()\nout["index"]\nout["lvl"]\n')
    assert set(fc.dfs["out"].columns) == {"a", "k", "index"}
    assert len(fc.diagnostics) == 1
