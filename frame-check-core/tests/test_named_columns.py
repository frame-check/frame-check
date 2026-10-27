"""Frames whose columns axis may be named: melt names its variable column after it.

`df.melt()` without `var_name` (or with `var_name=None`) names the variable
column after the columns axis: "variable" when unnamed, the axis name when
set, and one column per level of a MultiIndex. When the axis may be named,
the melted frame must not be tracked (or it reports false positives).
"""

import pytest
from frame_check_core.checker import Checker

SETUP = 'import pandas as pd\ndf = pd.DataFrame({"a": [1], "b": [2]})\n'


def _pandas_columns(code: str) -> list:
    namespace: dict = {}
    exec(SETUP + code, namespace)  # noqa: S102
    return list(namespace["m"].columns)


@pytest.mark.parametrize(
    "code",
    [
        'df.columns.name = "feat"\nm = df.melt(id_vars=["a"])\n',
        'df.columns.names = ["feat"]\nm = df.melt(id_vars=["a"])\n',
        'df.columns.name = "feat"\nm = pd.melt(df, id_vars=["a"])\n',
        'df.columns.name = "feat"\nm = pd.melt(frame=df, id_vars=["a"])\n',
        'df.columns.name = "feat"\nm = df.melt(id_vars=["a"], var_name=None)\n',
        'df.columns.name = "feat"\nm = df.melt("a", None, None)\n',
        'df = df.rename_axis(columns="feat")\nm = df.melt(id_vars=["a"])\n',
        'df.rename_axis(columns="feat", inplace=True)\nm = df.melt(id_vars=["a"])\n',
        'df.columns.set_names("feat", inplace=True)\nm = df.melt(id_vars=["a"])\n',
        (
            'df = df.set_axis(pd.Index(["a", "b"], name="feat"), axis=1)\n'
            'm = df.melt(id_vars=["a"])\n'
        ),
        'm = df.rename_axis(columns="feat").melt(id_vars=["a"])\n',
        'df.columns.name = "feat"\nm = df.assign(c=1)[["a", "b"]].melt(id_vars="a")\n',
        'df.columns.name = "feat"\nd2 = df.copy()\nm = d2.melt(id_vars=["a"])\n',
        'df.columns.name = "feat"\nd2 = pd.DataFrame(df)\nm = d2.melt(id_vars=["a"])\n',
        'df.columns.name = "feat"\nm = df.merge(df, on="a").melt(id_vars=["a"])\n',
        'df.columns.name = "feat"\ndf = df.melt(id_vars=["a"])\nm = df\n',
    ],
)
def test_melt_of_named_columns_is_not_tracked(code: str):
    assert "feat" in _pandas_columns(code)
    fc = Checker.check(SETUP + code + 'm["feat"]\n')
    assert fc.diagnostics == []
    assert "m" not in fc.dfs


def test_melt_of_named_columns_in_a_branch_is_not_tracked():
    code = (
        SETUP + 'if flag:\n    df.columns.name = "feat"\nm = df.melt(id_vars=["a"])\n'
    )
    fc = Checker.check(code + 'm["feat"]\n')
    assert fc.diagnostics == []
    assert "m" not in fc.dfs


@pytest.mark.parametrize(
    "source",
    [
        'pd.read_csv("f.csv", usecols=["a", "b"])',
        'pd.read_parquet("f.parquet", columns=["a", "b"])',
        'pd.concat([pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]})], keys=["x", "y"])',
    ],
)
def test_melt_of_frames_with_unknown_columns_axis_is_not_tracked(source: str):
    # A file can store the axis name (parquet, pickle) or several header rows
    code = (
        f'import pandas as pd\ndf = {source}\nm = df.melt(id_vars=["a"])\nm["feat"]\n'
    )
    fc = Checker.check(code)
    assert fc.diagnostics == []
    assert "m" not in fc.dfs


def test_multiindex_columns_melt_to_one_column_per_level():
    # Why a possibly-MultiIndex axis can't use the default name either
    namespace: dict = {}
    exec(  # noqa: S102
        "import pandas as pd\n"
        'df = pd.DataFrame({("x", "a"): [1], ("x", "b"): [2]})\n'
        "m = df.melt()\n",
        namespace,
    )
    assert list(namespace["m"].columns) == ["variable_0", "variable_1", "value"]


@pytest.mark.parametrize(
    "code",
    [
        'm = df.melt(id_vars=["a"])\n',
        'm = pd.melt(df, id_vars=["a"])\n',
        # Assigning a list gives a new, unnamed axis
        'df.columns.name = "feat"\ndf.columns = ["a", "b"]\nm = df.melt(id_vars="a")\n',
        # A new frame with the same name has its own axis
        (
            'df.columns.name = "feat"\ndf = pd.DataFrame({"a": [1], "b": [2]})\n'
            'm = df.melt(id_vars="a")\n'
        ),
    ],
)
def test_melt_of_unnamed_columns_is_still_tracked(code: str):
    assert _pandas_columns(code) == ["a", "variable", "value"]
    fc = Checker.check(SETUP + code + 'm["variable"]\nm["value"]\nm["zz"]\n')
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith("Column 'zz' does not exist")


@pytest.mark.parametrize(
    "call",
    ['df.melt(id_vars=["a"], var_name="k")', 'pd.melt(df, "a", None, "k")'],
)
def test_explicit_var_name_is_tracked_on_named_columns(call: str):
    code = f'df.columns.name = "feat"\nm = {call}\n'
    assert _pandas_columns(code) == ["a", "k", "value"]
    fc = Checker.check(SETUP + code + 'm["k"]\nm["feat"]\n')
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith("Column 'feat' does not exist")


def test_missing_id_vars_are_still_reported_on_named_columns():
    fc = Checker.check(SETUP + 'df.columns.name = "feat"\nm = df.melt(id_vars="zz")\n')
    assert len(fc.diagnostics) == 1
    assert "column 'zz' does not exist" in fc.diagnostics[0].message
