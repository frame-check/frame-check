"""Feature tests for column removal methods (CRM)."""

import pytest
from frame_check_core.checker import Checker

# --- CRM-1: del statement ---


@pytest.mark.support(code="#CRM-1")
def test_crm_1_del_statement():
    """del df['A']"""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
del df['A']
df['B']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"B"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CRM-1-1")
def test_crm_1_1_del_then_read_deleted_column():
    """Reading a deleted column is reported."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
del df['A']
df['A']
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert "'A' does not exist" in fc.diagnostics[0].message


@pytest.mark.support(code="#CRM-1-2")
def test_crm_1_2_del_multiple_targets():
    """del df['A'], df['B']"""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
del df['A'], df['B']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"C"}


# --- CRM-2: drop method ---


@pytest.mark.support(code="#CRM-2")
def test_crm_2_drop_method():
    """df = df.drop('A', axis=1)"""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df = df.drop('A', axis=1)
df['B']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"B"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CRM-2-1")
def test_crm_2_1_drop_then_read_dropped_column():
    """Reading a dropped column is reported."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df = df.drop('A', axis='columns')
df['A']
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert "'A' does not exist" in fc.diagnostics[0].message


@pytest.mark.support(code="#CRM-2-2")
def test_crm_2_2_drop_inplace():
    """df.drop(columns='A', inplace=True)"""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df.drop(columns='A', inplace=True)
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"B"}


@pytest.mark.support(code="#CRM-2-3")
def test_crm_2_3_drop_into_new_frame_keeps_source():
    """df2 = df.drop(columns='A') leaves df untouched."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df2 = df.drop(columns='A')
df['A']
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B"}
    assert set(fc.dfs["df2"].columns.keys()) == {"B"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CRM-2-4")
def test_crm_2_4_drop_rows_leaves_columns():
    """df.drop(0) and df.drop(index=[0]) drop rows, not columns."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df = df.drop(0)
df = df.drop(index=[0])
df = df.drop('A')
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"A", "B"}


@pytest.mark.support(code="#CRM-2-5")
def test_crm_2_5_drop_opaque_labels_no_false_positives():
    """Unresolvable labels leave the columns untouched."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df = df.drop(columns=get_cols())
df['A']
df['B']
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B"}
    assert len(fc.diagnostics) == 0


# --- CRM-3: drop with columns ---


@pytest.mark.support(code="#CRM-3")
def test_crm_3_drop_with_columns():
    """df = df.drop(columns=['A', 'B'])"""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
df = df.drop(columns=['A', 'B'])
df['C']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"C"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CRM-3-1")
def test_crm_3_1_drop_with_columns_variable():
    """df = df.drop(columns=cols) with cols defined as a list"""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
cols = ['A', 'B']
df = df.drop(columns=cols)
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"C"}


# --- CRM-4: drop multiple ---


@pytest.mark.support(code="#CRM-4")
def test_crm_4_drop_multiple():
    """df = df.drop(['A','B'], axis=1)"""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
df = df.drop(['A', 'B'], axis=1)
df['C']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"C"}
    assert len(fc.diagnostics) == 0


# --- CRM-5: pop method ---


@pytest.mark.support(code="#CRM-5")
def test_crm_5_pop_method():
    """removed = df.pop('A')"""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
removed = df.pop('A')
df['B']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"B"}
    assert "removed" not in fc.dfs
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CRM-5-1")
def test_crm_5_1_pop_standalone_then_read():
    """df.pop('A') as a statement removes 'A'; reading it is reported."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df.pop('A')
df['A']
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"B"}
    assert len(fc.diagnostics) == 1


@pytest.mark.support(code="#CRM-5-2")
def test_crm_5_2_pop_rebinding_frame_name_stops_tracking():
    """df = df.pop('A') rebinds df to a Series, so it is no longer checked."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df = df.pop('A')
df['anything']
"""
    fc = Checker.check(code)
    assert "df" not in fc.dfs
    assert len(fc.diagnostics) == 0


# --- Removing non-existent columns (raises KeyError at runtime) ---


@pytest.mark.support(code="#CRM-1-3")
def test_crm_1_3_del_missing_column_is_reported():
    """del df['X'] on a missing column is reported with a suggestion."""
    code = """
import pandas as pd
df = pd.DataFrame({'Name': [1]})
del df['Nmae']
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    diag = fc.diagnostics[0]
    assert diag.message.startswith("del: column 'Nmae' does not exist")
    assert diag.name_suggestion == "Name"
    assert diag.region.start.row == 4


@pytest.mark.support(code="#CRM-2-6")
def test_crm_2_6_drop_missing_columns_is_reported():
    """df.drop(columns=[...]) with missing labels is reported."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df = df.drop(columns=['A', 'X'])
df['B']
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith(
        "df.drop(): column 'X' does not exist on DataFrame 'df'."
    )


@pytest.mark.support(code="#CRM-2-7")
def test_crm_2_7_drop_missing_columns_errors_ignore():
    """errors='ignore' silently skips missing labels."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1], 'B': [2]})
df.drop(columns=['A', 'X'], errors='ignore', inplace=True)
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"B"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CRM-5-3")
def test_crm_5_3_pop_missing_column_is_reported():
    """df.pop('X') on a missing column is reported."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1]})
x = df.pop('X')
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith("df.pop(): column 'X' does not exist")
