"""Feature tests for column assignment methods (CAM)."""

import pytest
from frame_check_core.checker import Checker

# --- CAM-1: Direct assignment ---


@pytest.mark.support(code="#CAM-1")
def test_cam_1_direct_assignment():
    """df["c"] = [7, 8, 9]"""
    code = """
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df["c"] = [7, 8, 9]
df["c"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert sorted(df.columns.keys()) == ["a", "b", "c"]
    assert len(fc.diagnostics) == 0


# --- CAM-7: assign method ---


@pytest.mark.support(code="#CAM-7")
def test_cam_7_assign_method():
    """df = df.assign(A=[1, 2, 3])"""
    code = """
import pandas as pd
df = pd.DataFrame({})
df = df.assign(A=[1, 2, 3])
df["A"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"A"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CAM-7-1")
def test_cam_7_1_assign_subscript():
    """df.assign(A=[1, 2, 3])["A"] - chained subscript access"""
    code = """
import pandas as pd
df = pd.DataFrame({})
df.assign(A=[1, 2, 3])["A"]
df.assign(A=[1, 2, 3])["B"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    # assign returns a new frame; df itself is unchanged
    assert set(df.columns.keys()) == set()
    assert len(fc.diagnostics) == 1
    assert "Column 'B' does not exist" in fc.diagnostics[0].message


@pytest.mark.support(code="#CAM-7-1-1")
def test_cam_7_1_1_chained_read_of_missing_column():
    """df.drop(columns="A")["A"] is reported against the chained frame."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1], "B": [2]})
total = df.drop(columns="A")["A"].sum()
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith(
        "Column 'A' does not exist on DataFrame 'df.drop(...)'."
    )


@pytest.mark.support(code="#CAM-7-2")
def test_cam_7_2_assign_chain():
    """df = df.assign(A=[1, 2, 3]).assign(B=[4, 5, 6]) - chained assign"""
    code = """
import pandas as pd
df = pd.DataFrame({})
df = df.assign(A=[1, 2, 3]).assign(B=[4, 5, 6])
df["A"]
df["B"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"A", "B"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CAM-7-3")
def test_cam_7_3_chain_with_other_methods():
    """df2 = df.assign(C=1).drop(columns="A").rename(columns={"B": "b"})"""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1], "B": [2]})
df2 = df.assign(C=1).drop(columns="A").rename(columns={"B": "b"})
df["A"]
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B"}
    assert set(fc.dfs["df2"].columns.keys()) == {"b", "C"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CAM-7-4")
def test_cam_7_4_chain_reports_errors_in_intermediate_steps():
    """Errors inside a chain are reported for the step that raises."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df = df.assign(B=1).drop(columns="X").assign(C=2)
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B", "C"}
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith(
        "df.assign(...).drop(): column 'X' does not exist"
    )


@pytest.mark.support(code="#CAM-7-5")
def test_cam_7_5_chain_keeps_column_dependencies():
    """df = df.assign(...).assign(...) keeps dependencies on the tracker."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df["C"] = df["A"]
df = df.assign(B=1).assign(D=2)
"""
    fc = Checker.check(code)
    assert fc.dfs["df"].columns["C"] == {"A"}


@pytest.mark.support(code="#CAM-7-6")
def test_cam_7_6_chain_on_untracked_frame_is_ignored():
    """Chains rooted at unknown frames or unregistered methods are skipped."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
other = load().assign(B=1)
df2 = df.set_index("A").assign(B=1)
"""
    fc = Checker.check(code)
    assert "other" not in fc.dfs
    assert "df2" not in fc.dfs
    assert len(fc.diagnostics) == 0


# --- CAM-8: Multiple assign ---


@pytest.mark.support(code="#CAM-8")
def test_cam_8_multiple_assign():
    """df = df.assign(B=1, C=2)"""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df = df.assign(B=1, C=2)
df["B"]
df["C"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"A", "B", "C"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CAM-8-1")
def test_cam_8_1_lambda_uses_columns_from_same_assign():
    """Callables see the columns created by earlier keywords."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df = df.assign(B=1, C=2, D=lambda x: x["A"] + x["B"] + x["C"])
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B", "C", "D"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CAM-8-2")
def test_cam_8_2_lambda_uses_later_column_is_reported():
    """A callable can't read a column created by a later keyword."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df = df.assign(D=lambda x: x["B"] * 2, B=1)
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith(
        "df.assign(): column 'B' does not exist on DataFrame 'df'."
    )


@pytest.mark.support(code="#CAM-8-3")
def test_cam_8_3_lambda_with_typo_suggests_column():
    """A typo inside an assign lambda gets a suggestion."""
    code = """
import pandas as pd
df = pd.DataFrame({"price": [1], "quantity": [2]})
df = df.assign(total=lambda d: d["price"] * d["quantiy"])
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].name_suggestion == "quantity"


@pytest.mark.support(code="#CAM-8-4")
def test_cam_8_4_lambda_across_chained_assign():
    """df.assign(B=1).assign(C=lambda x: x['B'] * 2)"""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df = df.assign(B=1).assign(C=lambda x: x["B"] * 2)
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B", "C"}
    assert len(fc.diagnostics) == 0


# --- CAM-9: insert method ---


@pytest.mark.support(code="#CAM-9")
def test_cam_9_insert_method():
    """df.insert(0, "A", [1, 2, 3])"""
    code = """
import pandas as pd
df = pd.DataFrame({})
df.insert(0, "A", [1, 2, 3])
df["A"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"A"}
    assert len(fc.diagnostics) == 0


# --- CAM-10: setitem with list ---


@pytest.mark.support(code="#CAM-10")
def test_cam_10_setitem_with_list():
    """df[["c", "d"]] = [[7, 8, 9], [10, 11, 12]]"""
    code = """
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df[["c", "d"]] = [[7, 8, 9], [10, 11, 12]]
df["c"]
df["d"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert sorted(df.columns.keys()) == ["a", "b", "c", "d"]
    assert len(fc.diagnostics) == 0
