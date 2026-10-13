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


# --- CAM-2: Attribute assignment ---


@pytest.mark.support(code="#CAM-2")
def test_cam_2_attribute_assignment_does_not_create_columns():
    """df.new = value sets a plain attribute (pandas warns), not a column."""
    code = """
import pandas as pd
df = pd.DataFrame({"a": [1]})
df.a = [2]
df.new = [5]
df["a"]
df["new"]
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"a"}
    assert len(fc.diagnostics) == 1
    assert "Column 'new' does not exist" in fc.diagnostics[0].message


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


@pytest.mark.support(code="#CAM-8-5")
def test_cam_8_5_assign_with_unpacked_literal_dict():
    """df.assign(**{"B": 1}) adds B (was a false positive)."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df = df.assign(**{"B": 1}, C=2)
df["B"]
df["C"]
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B", "C"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CAM-8-6")
def test_cam_8_6_assign_with_unresolved_mapping_stops_tracking():
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df = df.assign(**new_columns)
df["anything"]
"""
    fc = Checker.check(code)
    assert "df" not in fc.dfs
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


# --- Features that need no special handling ---
# Assigning any right-hand side to df["col"] adds the column, and the
# columns read on the right-hand side are still checked.

SOURCE = """
import pandas as pd
df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "combined": ["x y", "z w"]})
other_df = pd.DataFrame({"A": [5, 6]})
"""


@pytest.mark.support(code="#CAM-11")
def test_cam_11_from_dictionary():
    """df = pd.DataFrame({'A': [1,2], 'B': [3,4]})"""
    fc = Checker.check(SOURCE + 'df["A"]\ndf["B"]\n')
    assert {"A", "B"} <= set(fc.dfs["df"].columns)
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    ("code", "new_columns"),
    [
        pytest.param('df["C"] = df["A"].copy()', {"C"}, id="CAM-18"),
        pytest.param('df["C"] = df["A"].apply(lambda x: x * 2)', {"C"}, id="CAM-19"),
        pytest.param(
            'df[["first", "last"]] = df["combined"].str.split(expand=True)',
            {"first", "last"},
            id="CAM-20",
        ),
        pytest.param(
            'df["C"] = df.groupby("A")["B"].transform("mean")', {"C"}, id="CAM-21"
        ),
    ],
)
def test_assignment_from_derived_series(code: str, new_columns: set[str]):
    fc = Checker.check(
        SOURCE + code + "\n" + "".join(f'df["{c}"]\n' for c in new_columns)
    )
    assert new_columns <= set(fc.dfs["df"].columns)
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    "code",
    [
        'df["C"] = df["Z"].copy()',
        'df["C"] = df["Z"].apply(lambda x: x * 2)',
        'df[["first", "last"]] = df["Z"].str.split(expand=True)',
    ],
)
def test_assignment_from_derived_series_checks_its_source(code: str):
    fc = Checker.check(SOURCE + code + "\n")
    assert len(fc.diagnostics) == 1
    assert "'Z' does not exist" in fc.diagnostics[0].message


@pytest.mark.support(code="#CAM-18")
def test_cam_18_copy_from_another_column():
    """df['B'] = df['A'].copy()"""
    fc = Checker.check(SOURCE + 'df["C"] = df["A"].copy()\ndf["C"]\n')
    assert fc.diagnostics == []


@pytest.mark.support(code="#CAM-19")
def test_cam_19_transform_with_apply():
    """df['B'] = df['A'].apply(lambda x: x*2)"""
    fc = Checker.check(SOURCE + 'df["C"] = df["A"].apply(lambda x: x * 2)\ndf["C"]\n')
    assert fc.diagnostics == []


@pytest.mark.support(code="#CAM-20")
def test_cam_20_expanding_columns():
    """df[['A','B']] = df['combined'].str.split(expand=True)"""
    code = 'df[["first", "last"]] = df["combined"].str.split(expand=True)\n'
    fc = Checker.check(SOURCE + code + 'df["last"]\n')
    assert fc.diagnostics == []


@pytest.mark.support(code="#CAM-21")
def test_cam_21_from_groupby():
    """df['new'] = df.groupby('A')['B'].transform('mean')"""
    code = 'df["new"] = df.groupby("A")["B"].transform("mean")\ndf["new"]\n'
    fc = Checker.check(SOURCE + code)
    assert fc.diagnostics == []


@pytest.mark.support(code="#CAM-25")
def test_cam_25_update_method():
    """df.update(other_df[['A']]) changes values, not columns."""
    fc = Checker.check(SOURCE + 'df.update(other_df[["A"]])\ndf["A"]\n')
    assert set(fc.dfs["df"].columns) == {"A", "B", "combined"}
    assert fc.diagnostics == []
