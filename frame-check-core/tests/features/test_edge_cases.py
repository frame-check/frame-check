"""Feature tests for Edge Cases (EC)."""

import pytest
from frame_check_core.checker import Checker

# --- EC-2: Rename operation ---


@pytest.mark.support(code="#EC-2")
def test_ec_2_rename_operation_columns():
    """df.rename"""
    code = """
import pandas as pd
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df = df.rename(columns={'col1': 'cola', 'col2': 'colb'})
df['cola']
df['colb']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"cola", "colb"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#EC-2-1")
@pytest.mark.xfail(reason="Standalone method calls not implemented", strict=True)
def test_ec_2_1_rename_inplace():
    """df.rename(columns=..., inplace=True)"""
    code = """
import pandas as pd
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df.rename(columns={'col1': 'cola'}, inplace=True)
df['cola']
df['col2']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"cola", "col2"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#EC-2-2")
def test_ec_2_2_rename_mapper_axis_int():
    """df.rename({...}, axis=1)"""
    code = """
import pandas as pd
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df = df.rename({'col1': 'cola'}, axis=1)
df['cola']
df['col2']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"cola", "col2"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#EC-2-3")
def test_ec_2_3_rename_mapper_axis_columns_str():
    """df.rename({...}, axis='columns')"""
    code = """
import pandas as pd
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df = df.rename({'col1': 'cola'}, axis='columns')
df['cola']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"cola", "col2"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#EC-2-4")
def test_ec_2_4_rename_index_only_leaves_columns():
    """df.rename(index={...}) must not touch columns"""
    code = """
import pandas as pd
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df = df.rename(index={0: 'first'})
df['col1']
df['col2']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"col1", "col2"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#EC-2-5")
def test_ec_2_5_rename_nonexistent_column_is_noop():
    """Renaming a key not in df.columns leaves df unchanged (errors='ignore' default)."""
    code = """
import pandas as pd
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df = df.rename(columns={'missing': 'whatever'})
df['col1']
df['col2']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"col1", "col2"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#EC-2-6")
def test_ec_2_6_rename_with_opaque_mapping_no_false_positives():
    """When the mapping can't be resolved statically, columns are left untouched
    and access to existing columns must still pass."""
    code = """
import pandas as pd
import json
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
mapping = json.loads('{"col1": "cola"}')
df = df.rename(columns=mapping)
df['col1']
df['col2']
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"col1", "col2"}
    assert len(fc.diagnostics) == 0
