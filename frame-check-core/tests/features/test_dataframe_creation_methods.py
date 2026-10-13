"""Feature tests for DataFrame creation methods (DCMS)."""

from pathlib import Path

import pytest
from frame_check_core.checker import Checker

CSV_TEST_FILE = (Path(__file__).parent.parent / "data" / "csv_file.csv").as_posix()


# --- DCMS-1: Dictionary of Lists ---


@pytest.mark.support(code="#DCMS-1")
def test_dcms_1_dictionary_of_lists():
    """pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})"""
    code = """
import pandas as pd
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df['col1']
df['col2']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"col1", "col2"}
    assert len(fc.diagnostics) == 0


# --- DCMS-2: List of Dictionaries ---


@pytest.mark.support(code="#DCMS-2")
def test_dcms_2_list_of_dictionaries():
    """pd.DataFrame([{'col1': 1, 'col2': 3}, {'col1': 2, 'col2': 4}])"""
    code = """
import pandas as pd
df = pd.DataFrame([{'col1': 1, 'col2': 3}, {'col1': 2, 'col2': 4}])
df['col1']
df['col2']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"col1", "col2"}
    assert len(fc.diagnostics) == 0


# --- DCMS-6: From CSV ---


@pytest.mark.support(code="#DCMS-6")
def test_dcms_6_read_csv_usecols():
    """pd.read_csv('file.csv', usecols=["a","b"])"""
    code = f"""
import pandas as pd
df = pd.read_csv("{CSV_TEST_FILE}", usecols=['a', 'b', 'c'])
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b", "c"}


@pytest.mark.support(code="#DCMS-6-1")
def test_dcms_6_1_read_csv_usecols_indirect():
    """pd.read_csv with usecols from variable"""
    code = f"""
import pandas as pd
cols = ['a', 'b', 'c']
df = pd.read_csv("{CSV_TEST_FILE}", usecols=cols)
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b", "c"}


# --- DCMS-9: From excel ---


@pytest.mark.support(code="#DCMS-9")
def test_dcms_9_read_excel_usecols():
    """pd.read_excel('file.csv', usecols=["a","b","c"])"""
    code = f"""
import pandas as pd
df = pd.read_excel("{CSV_TEST_FILE}", usecols=["a", "b", "c"])
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b", "c"}


@pytest.mark.support(code="#DCMS-9-1")
def test_dcms_9_1_read_excel_usecols_indirect():
    """pd.read_excel with usecols from variable"""
    code = f"""
import pandas as pd
cols = ['a', 'b', 'c']
df = pd.read_excel("{CSV_TEST_FILE}", usecols=cols)
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b", "c"}


# --- DCMS-7: From JSON ---


@pytest.mark.support(code="#DCMS-7")
def test_dcms_7_read_json():
    """pd.read_json('file.json') - cannot determine columns statically."""
    code = """
import pandas as pd
df = pd.read_json('data.json')
"""
    fc = Checker.check(code)
    # read_json handler returns None, so DataFrame might not be tracked
    # Just verify no errors are raised
    assert len(fc.diagnostics) == 0


# --- DCMS-10: From Parquet ---


@pytest.mark.support(code="#DCMS-10")
def test_dcms_10_read_parquet_columns():
    """pd.read_parquet('file.parquet', columns=['a', 'b'])"""
    code = """
import pandas as pd
df = pd.read_parquet('data.parquet', columns=['a', 'b'])
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b"}


@pytest.mark.support(code="#DCMS-10-1")
def test_dcms_10_1_read_parquet_columns_indirect():
    """pd.read_parquet with columns from variable"""
    code = """
import pandas as pd
cols = ['a', 'b']
df = pd.read_parquet('data.parquet', columns=cols)
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b"}


# --- DCMS-14: Copy constructor ---


@pytest.mark.support(code="#DCMS-14")
def test_dcms_14_copy_constructor():
    """pd.DataFrame(other_df)"""
    code = """
import pandas as pd
other_df = pd.DataFrame({"a": [1], "b": [2]})
df = pd.DataFrame(other_df)
df2 = pd.DataFrame(other_df[["a"]].assign(c=1))
df["a"]
df2["b"]
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"a", "b"}
    assert set(fc.dfs["df2"].columns.keys()) == {"a", "c"}
    assert len(fc.diagnostics) == 1
    assert "Column 'b' does not exist on DataFrame 'df2'." in fc.diagnostics[0].message


@pytest.mark.support(code="#DCMS-14-1")
def test_dcms_14_1_copy_of_untracked_frame_is_not_tracked():
    code = """
import pandas as pd
df = pd.DataFrame(load())
"""
    fc = Checker.check(code)
    assert "df" not in fc.dfs


# --- pd.DataFrame(...) with columns= (DCMS-4, DCMS-5, DCMS-12, DCMS-13, DCMS-3) ---


def _pandas_columns(expr: str) -> set[str]:
    import numpy as np
    import pandas as pd

    namespace = {"pd": pd, "np": np, "other": pd.DataFrame({"A": [1], "B": [2]})}
    return set(eval(expr, namespace).columns)


def _checker_columns(expr: str) -> set[str] | None:
    code = f"""
import numpy as np
import pandas as pd
other = pd.DataFrame({{"A": [1], "B": [2]}})
out = {expr}
"""
    fc = Checker.check(code)
    assert fc.diagnostics == []
    tracker = fc.dfs.get("out")
    return None if tracker is None else set(tracker.columns)


CONSTRUCTOR_CASES = [
    pytest.param('pd.DataFrame([[1, 2]], columns=["c1", "c2"])', id="DCMS-5"),
    pytest.param('pd.DataFrame(np.array([[1, 2]]), columns=["c1", "c2"])', id="DCMS-4"),
    pytest.param('pd.DataFrame(index=["x"], columns=["c1", "c2"])', id="DCMS-12"),
    pytest.param('pd.DataFrame({"col1": 1}, index=[0, 1])', id="DCMS-13"),
    pytest.param(
        'pd.DataFrame({"c1": pd.Series([1]), "c2": pd.Series([2])})', id="DCMS-3"
    ),
    pytest.param('pd.DataFrame({"a": [1], "b": [2]}, columns=["a", "z"])', id="dict"),
    pytest.param('pd.DataFrame([{"a": 1, "b": 2}], columns=["b"])', id="records"),
    pytest.param('pd.DataFrame(other, columns=["A", "Q"])', id="frame"),
    pytest.param('pd.DataFrame([[1, 2]], None, ["c1", "c2"])', id="positional"),
    pytest.param("pd.DataFrame()", id="empty"),
]


@pytest.mark.parametrize("expr", CONSTRUCTOR_CASES)
def test_constructor_matches_pandas(expr: str):
    assert _checker_columns(expr) == _pandas_columns(expr)


@pytest.mark.support(code="#DCMS-5")
def test_dcms_5_list_of_lists():
    """pd.DataFrame([[1, 2], [3, 4]], columns=['col1', 'col2'])"""
    expr = "pd.DataFrame([[1, 2], [3, 4]], columns=['col1', 'col2'])"
    assert _checker_columns(expr) == {"col1", "col2"}


@pytest.mark.support(code="#DCMS-4")
def test_dcms_4_numpy_array():
    """pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=['col1', 'col2'])"""
    expr = "pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=['col1', 'col2'])"
    assert _checker_columns(expr) == {"col1", "col2"}


@pytest.mark.support(code="#DCMS-12")
def test_dcms_12_from_index():
    """pd.DataFrame(index=['a', 'b'], columns=['col1', 'col2'])"""
    expr = "pd.DataFrame(index=['a', 'b'], columns=['col1', 'col2'])"
    assert _checker_columns(expr) == {"col1", "col2"}


@pytest.mark.support(code="#DCMS-13")
def test_dcms_13_from_scalar():
    """pd.DataFrame({'col1': 1}, index=[0, 1, 2])"""
    assert _checker_columns("pd.DataFrame({'col1': 1}, index=[0, 1, 2])") == {"col1"}


@pytest.mark.support(code="#DCMS-3")
def test_dcms_3_dictionary_of_series():
    """pd.DataFrame({'col1': pd.Series([1, 2]), 'col2': pd.Series([3, 4])})"""
    expr = "pd.DataFrame({'col1': pd.Series([1, 2]), 'col2': pd.Series([3, 4])})"
    assert _checker_columns(expr) == {"col1", "col2"}


def test_unresolved_columns_argument_is_not_tracked():
    assert _checker_columns("pd.DataFrame([[1]], columns=names)") is None


FROM_CASES = [
    pytest.param('pd.DataFrame.from_dict({"c1": [1], "c2": [2]})', id="from_dict"),
    pytest.param(
        'pd.DataFrame.from_dict({"r1": [1, 2]}, orient="index", columns=["c1", "c2"])',
        id="from_dict-index",
    ),
    pytest.param(
        'pd.DataFrame.from_records([("a", 1)], columns=["c1", "c2"])',
        id="from_records",
    ),
    pytest.param('pd.DataFrame.from_records([{"a": 1, "b": 2}])', id="records-dicts"),
    pytest.param(
        'pd.DataFrame.from_records([{"a": 1, "b": 2}], exclude=["b"])', id="exclude"
    ),
    pytest.param(
        'pd.DataFrame.from_records([{"a": 1, "b": 2}], index="a")', id="index"
    ),
    pytest.param(
        'pd.DataFrame.from_records([("x", 1)], columns=["c1", "c2"], index="c1")',
        id="index-columns",
    ),
]


@pytest.mark.parametrize("expr", FROM_CASES)
def test_from_constructors_match_pandas(expr: str):
    assert _checker_columns(expr) == _pandas_columns(expr)


@pytest.mark.support(code="#DCMS-15")
def test_dcms_15_from_records():
    """pd.DataFrame.from_records([('a', 1), ('b', 2)], columns=['col1', 'col2'])"""
    expr = "pd.DataFrame.from_records([('a', 1), ('b', 2)], columns=['col1', 'col2'])"
    assert _checker_columns(expr) == {"col1", "col2"}


@pytest.mark.support(code="#DCMS-16")
def test_dcms_16_from_dict():
    """pd.DataFrame.from_dict({'col1': [1, 2], 'col2': [3, 4]})"""
    expr = "pd.DataFrame.from_dict({'col1': [1, 2], 'col2': [3, 4]})"
    assert _checker_columns(expr) == {"col1", "col2"}


@pytest.mark.parametrize(
    "expr",
    [
        'pd.DataFrame.from_dict({"r1": [1, 2]}, orient="index")',
        "pd.DataFrame.from_dict(data)",
        "pd.DataFrame.from_records(rows)",
    ],
)
def test_unresolved_from_constructors_are_not_tracked(expr: str):
    assert _checker_columns(expr) is None


def test_read_on_from_dict_is_checked():
    code = """
import pandas as pd
df = pd.DataFrame.from_dict({"c1": [1]})
df["c2"]
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert "Column 'c2' does not exist on DataFrame 'df'." in fc.diagnostics[0].message
