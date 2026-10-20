"""Feature tests for reshaping: melt, get_dummies, pivot, stack and transpose.

melt is differential: the same expression runs through pandas and through the
checker, and the resulting column sets must match. The other operations have
data-dependent columns, so their results must be untracked (and must not
produce diagnostics); labels that pandas rejects with a KeyError are reported.
"""

import pytest
from frame_check_core.checker import Checker

SETUP = """
import pandas as pd
df = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 4], "c": [5, 6]})
"""


def _pandas_columns(expr: str) -> list:
    """Run `expr` with real pandas on the SETUP frame (test-only snippets)."""
    namespace: dict = {}
    exec(SETUP, namespace)  # noqa: S102
    return list(eval(expr, namespace).columns)


def _checker_columns(expr: str) -> set[str] | None:
    fc = Checker.check(SETUP + f"out = {expr}\n")
    assert fc.diagnostics == []
    tracker = fc.dfs.get("out")
    return None if tracker is None else set(tracker.columns)


MELT_CASES = [
    "df.melt()",
    'df.melt(id_vars="id")',
    'df.melt(id_vars=["id", "a"])',
    'df.melt(id_vars=("id",))',
    'df.melt(id_vars=["id"], value_vars=["a"])',
    'df.melt(value_vars=["a", "b"])',
    'df.melt(value_vars="a")',
    "df.melt(id_vars=[])",
    "df.melt(id_vars=None, value_vars=None)",
    'df.melt(id_vars="id", var_name="k", value_name="v")',
    'df.melt(id_vars="id", var_name=None)',
    'df.melt(id_vars="id", var_name="id")',
    'df.melt(id_vars="id", ignore_index=False)',
    'df.melt("id", ["a"], "k", "v")',
    'df.melt(id_vars=["a"], value_vars=["a", "b"])',
    'pd.melt(df, id_vars="id")',
    'pd.melt(df, "id", "a", "k")',
    'pd.melt(frame=df, id_vars=["id"], value_name="v")',
    'df.assign(z=1).melt(id_vars="z")',
    'df.melt(id_vars="id").drop(columns="value")',
]


@pytest.mark.parametrize("expr", MELT_CASES)
def test_melt_matches_pandas(expr: str):
    assert _checker_columns(expr) == set(_pandas_columns(expr))


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ('df.melt(id_vars="zz")', "df.melt(): column 'zz' does not exist"),
        ('df.melt(id_vars=["id", "zz"])', "df.melt(): column 'zz' does not exist"),
        ('df.melt(value_vars=["a", "zz"])', "df.melt(): column 'zz' does not exist"),
        ('df.melt(id_vars="zz", value_vars="yy")', "df.melt(): columns 'zz', 'yy'"),
        ('pd.melt(df, id_vars="zz")', "pd.melt(): column 'zz' does not exist"),
        ('pd.melt(df, "id", "zz")', "pd.melt(): column 'zz' does not exist"),
    ],
)
def test_melt_with_missing_label_is_reported(expr: str, message: str):
    with pytest.raises(KeyError):
        _pandas_columns(expr)
    fc = Checker.check(SETUP + f"out = {expr}\n")
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.splitlines()[0].startswith(message)


def test_melt_result_is_tracked():
    fc = Checker.check(SETUP + 'out = df.melt(id_vars="id")\nout["value"]\nout["a"]\n')
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith("Column 'a' does not exist")


@pytest.mark.parametrize(
    "expr",
    [
        'df.melt(id_vars="id", col_level=0)',  # MultiIndex columns
        'df.melt(id_vars="id", value_name=None)',  # NaN label
        'df.melt(id_vars="id", var_name=["x"])',  # ValueError
        "df.melt(id_vars=ids)",
        "df.melt(id_vars=[ids])",
        'df.melt(id_vars="id", var_name=name)',
        "df.melt(**opts)",
        "pd.melt(other, id_vars='id')",
    ],
)
def test_unresolvable_melt_stops_tracking(expr: str):
    fc = Checker.check(SETUP + f"out = {expr}\n")
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


def test_self_rebinding_melt_replaces_the_schema():
    code = SETUP + 'df = df.melt(id_vars="id")\ndf["value"]\ndf["a"]\n'
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith("Column 'a' does not exist")
