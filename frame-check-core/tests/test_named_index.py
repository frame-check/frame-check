"""Frames whose index may be named: keys may name index levels."""

import ast

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


SERIES = 's = pd.Series([1], index=pd.Index([0], name="k"))\n'


@pytest.mark.parametrize(
    "setup",
    [
        'df.index.rename("k", inplace=True)\n',
        'df.index.set_names("k", inplace=True)\n',
        'df.index.set_names(["k"], level=0, inplace=flag)\n',
        'df.rename_axis("k", inplace=True)\n',
        'df.index.names = ["k"]\n',
        # A dict of Series is indexed like the Series, name included
        SERIES + 'df = pd.DataFrame({"a": s, "b": [1]})\n',
        SERIES + 'df = pd.DataFrame(data={"a": s})\n',
        SERIES + 'df = pd.DataFrame.from_dict({"a": s})\n',
        SERIES + "df = pd.DataFrame(s.to_frame(), columns=['a'])\n",
        'df = pd.DataFrame(load(), columns=["a"])\n',
        'df = pd.DataFrame(data, columns=["a"])\n',
        # Parquet stores the index with its names
        'df = pd.read_parquet("f", columns=["a", "b"])\n',
    ],
)
def test_query_on_possibly_named_index_is_not_reported(setup: str):
    fc = Checker.check(PLAIN + setup + 'x = df.query("k > 0")\ny = df.eval("k")\n')
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    "call",
    [
        SERIES + 'x = pd.DataFrame({"a": s}).query("k >= 0")\n',
        'x = pd.read_parquet("f", columns=["a", "b"]).query("k > 0")\n',
        'x = df.rename_axis("k").query("k > 0")\n',
    ],
)
def test_query_on_inline_possibly_named_index_is_not_reported(call: str):
    fc = Checker.check(PLAIN + call)
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    "setup",
    [
        "",
        'df.index.rename("k")\n',
        'df.index.rename("k", inplace=False)\n',
        'df = pd.DataFrame([[1, 2]], columns=["a", "k"])\n',
        'df = pd.DataFrame({"a": range(3), "k": -1, "s": list("abc")})\n',
        'df = pd.DataFrame({"a": np.zeros(3), "k": [x for x in range(3)]})\n',
        'data = {"a": [1.5], "k": [2]}\ndf = pd.DataFrame(data)\n',
        'df = pd.DataFrame({"a": df["a"], "k": df["k"]})\n',
    ],
)
def test_query_on_unnamed_index_is_still_reported(setup: str):
    fc = Checker.check(PLAIN + setup + 'x = df.query("zz > 0")\n')
    assert len(fc.diagnostics) == 1
    assert "column 'zz' does not exist" in fc.diagnostics[0].message


def test_inline_query_on_literal_frame_is_still_reported():
    fc = Checker.check(
        'import pandas as pd\nx = pd.DataFrame({"a": [1]}).query("zz > 0")\n'
    )
    assert len(fc.diagnostics) == 1


@pytest.mark.parametrize(
    ("reader", "named"),
    [
        ('pd.read_feather("f")', True),
        ('pd.read_pickle("f")', True),
        ('pd.read_hdf("f")', True),
        ('pd.read_orc("f")', True),
        ('pd.read_json("f", orient="table")', True),
        ('pd.read_json("f", orient=orient)', True),
        ('pd.read_json("f")', False),
        ('pd.read_json("f", orient="records")', False),
    ],
)
def test_readers_restoring_a_named_index(reader: str, named: bool):
    """Untracked frames, so checked on the named-index state directly."""
    fc = Checker.check("import pandas as pd\n")
    assert fc._index_may_be_named(ast.parse(reader, mode="eval").body) is named
