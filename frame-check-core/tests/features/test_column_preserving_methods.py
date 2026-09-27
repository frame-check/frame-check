"""Tests for DataFrame methods that keep all columns."""

import pytest
from frame_check_core.checker import Checker


@pytest.mark.parametrize(
    "call",
    [
        "copy()",
        "head(5)",
        "tail()",
        "sample(n=2)",
        "sort_values('A')",
        "sort_index()",
        "drop_duplicates()",
        "dropna()",
        "fillna(0)",
        "ffill()",
        "bfill()",
        "replace(1, 2)",
        "astype(float)",
        "round(2)",
        "abs()",
        "clip(0, 1)",
        "query('A > 0')",
        "nlargest(3, 'A')",
        "nsmallest(3, 'A')",
    ],
)
def test_method_keeps_columns(call: str):
    code = f"""
import pandas as pd
df = pd.DataFrame({{"A": [1], "B": [2]}})
df2 = df.{call}
df2["X"]
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df2"].columns.keys()) == {"A", "B"}
    assert len(fc.diagnostics) == 1


def test_inplace_variant_returns_none():
    """df2 = df.fillna(0, inplace=True) binds None, not a DataFrame."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df2 = df.fillna(0, inplace=True)
"""
    fc = Checker.check(code)
    assert "df2" not in fc.dfs
    assert set(fc.dfs["df"].columns.keys()) == {"A"}


def test_chain_through_column_preserving_methods():
    """Chains through common methods keep being checked."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1], "B": [2]})
top = df.dropna().sort_values("A").head(10).assign(C=lambda x: x["A"] + x["Z"])
"""
    fc = Checker.check(code)
    assert set(fc.dfs["top"].columns.keys()) == {"A", "B", "C"}
    assert len(fc.diagnostics) == 1
    assert "column 'Z' does not exist" in fc.diagnostics[0].message


@pytest.mark.support(code="#CAM-16")
def test_cam_16_from_query_results():
    """df['new'] = df.query('A > 0')['B']"""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1], "B": [2]})
df["new"] = df.query("A > 0")["B"]
df["other"] = df.query("A > 0")["Missing"]
df["new"]
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B", "new", "other"}
    assert len(fc.diagnostics) == 1
    assert "Column 'Missing' does not exist" in fc.diagnostics[0].message
