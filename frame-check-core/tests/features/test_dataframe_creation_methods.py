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
