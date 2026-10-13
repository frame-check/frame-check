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
