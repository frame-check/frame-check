"""Feature tests for reindex and filter (CRM-11, CRM-13), compared with pandas."""

import pandas as pd
import pytest
from frame_check_core.checker import Checker


def _pandas_columns(expr: str) -> set[str]:
    df = pd.DataFrame({"A": [1], "B": [2], "AB": [3]})
    return set(eval(expr, {"df": df}).columns)


def _checker_columns(expr: str) -> set[str] | None:
    code = f"""
import pandas as pd
df = pd.DataFrame({{"A": [1], "B": [2], "AB": [3]}})
out = {expr}
"""
    fc = Checker.check(code)
    assert fc.diagnostics == []
    tracker = fc.dfs.get("out")
    return None if tracker is None else set(tracker.columns)


REINDEX_CASES = [
    'df.reindex(columns=["A", "Q"])',
    'df.reindex(["B", "Q"], axis=1)',
    'df.reindex(["B"], axis="columns")',
    "df.reindex(index=[0, 1])",
    "df.reindex([0, 1])",
]


@pytest.mark.support(code="#CRM-13")
@pytest.mark.parametrize("expr", REINDEX_CASES)
def test_crm_13_reindex_matches_pandas(expr: str):
    assert _checker_columns(expr) == _pandas_columns(expr)


def test_reindex_then_read_dropped_column():
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1], "B": [2]})
df = df.reindex(columns=["A"])
df["B"]
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert "Column 'B' does not exist" in fc.diagnostics[0].message


def test_reindex_with_unresolved_labels_is_not_tracked():
    assert _checker_columns("df.reindex(columns=wanted)") is None


FILTER_CASES = [
    'df.filter(items=["A", "Q"])',
    'df.filter(["B"])',
    'df.filter(like="A")',
    'df.filter(regex="^A")',
    'df.filter(regex="B$")',
    "df.filter(items=[0], axis=0)",
    'df.filter(like="A", axis=1)',
]


@pytest.mark.support(code="#CRM-11")
@pytest.mark.parametrize("expr", FILTER_CASES)
def test_crm_11_filter_matches_pandas(expr: str):
    assert _checker_columns(expr) == _pandas_columns(expr)


@pytest.mark.parametrize(
    "expr",
    ["df.filter(items=wanted)", "df.filter(regex=pattern)", 'df.filter(regex="[")'],
)
def test_unresolvable_filter_is_not_tracked(expr: str):
    assert _checker_columns(expr) is None


@pytest.mark.parametrize(
    ("call", "missing"),
    [
        ('df.groupby("Z").sum()', "'Z'"),
        ('df.groupby(by="Z")["A"].mean()', "'Z'"),
    ],
)
def test_groupby_with_missing_key_is_reported(call: str, missing: str):
    with pytest.raises(KeyError):
        eval(call, {"df": pd.DataFrame({"A": [1], "B": [2]})})
    fc = Checker.check(
        f'import pandas as pd\ndf = pd.DataFrame({{"A": [1], "B": [2]}})\nx = {call}\n'
    )
    assert len(fc.diagnostics) == 1
    assert f"column {missing} does not exist" in fc.diagnostics[0].message


@pytest.mark.parametrize(
    "call",
    [
        'df.groupby("A").sum()',
        'df.groupby(["A", df["B"]]).sum()',
        'df.groupby(["A", "Z"]).sum()',  # a list may be group labels
        "df.groupby(level=0).sum()",
        "df.groupby(key_func).sum()",
        'df["C"] = df.groupby("A")["B"].transform("mean")',
    ],
)
def test_groupby_valid_keys_are_not_reported(call: str):
    stmt = call if call.startswith("df[") else f"x = {call}"
    fc = Checker.check(
        f'import pandas as pd\ndf = pd.DataFrame({{"A": [1], "B": [2]}})\n{stmt}\n'
    )
    assert fc.diagnostics == []
