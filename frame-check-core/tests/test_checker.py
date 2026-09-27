"""Tests for the Checker."""

from pathlib import Path

from frame_check_core.checker import Checker

CSV_TEST_FILE = (Path(__file__).parent / "data" / "csv_file.csv").as_posix()


# --- Basic Checker.check() tests ---


def test_check_with_string_input():
    """Test Checker.check() with string input."""
    code = """
import pandas as pd
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
value = df['C']
"""
    checker = Checker.check(code)
    assert len(checker.dfs) == 1
    assert "df" in checker.dfs
    # Accessing non-existent column 'C' should produce a diagnostic
    assert len(checker.diagnostics) == 1


def test_check_with_file_input(tmp_path: Path):
    """Test Checker.check() with file input."""
    code = """
import pandas as pd
data = {'X': [1, 2], 'Y': [3, 4]}
df = pd.DataFrame(data)
result = df['Z']
"""
    test_file = tmp_path / "test_code.py"
    test_file.write_text(code)

    checker = Checker.check(test_file)
    assert len(checker.dfs) == 1
    assert "df" in checker.dfs
    # Accessing non-existent column 'Z' should produce a diagnostic
    assert len(checker.diagnostics) == 1


def test_check_valid_column_access():
    """Test that valid column access produces no diagnostics."""
    code = """
import pandas as pd
data = {'name': ['Alice', 'Bob'], 'age': [25, 30]}
df = pd.DataFrame(data)
names = df['name']
"""
    checker = Checker.check(code)
    assert len(checker.dfs) == 1
    tracker = checker.dfs.get("df")
    assert tracker is not None
    assert set(tracker.columns.keys()) == {"name", "age"}
    # Valid column access should produce no diagnostics
    assert len(checker.diagnostics) == 0


def test_check_multiple_dataframes():
    """Test checking code with multiple DataFrames."""
    code = """
import pandas as pd
data1 = {'A': [1, 2], 'B': [3, 4]}
df1 = pd.DataFrame(data1)
data2 = {'C': [5, 6], 'D': [7, 8]}
df2 = pd.DataFrame(data2)
val1 = df1['A']
val2 = df2['C']
val3 = df1['X']  # Invalid column
"""
    checker = Checker.check(code)
    assert len(checker.dfs) == 2
    assert "df1" in checker.dfs
    assert "df2" in checker.dfs
    assert set(checker.dfs["df1"].columns.keys()) == {"A", "B"}
    assert set(checker.dfs["df2"].columns.keys()) == {"C", "D"}
    # Only df1['X'] should produce a diagnostic
    assert len(checker.diagnostics) == 1


def test_check_empty_code():
    """Test checking empty code."""
    checker = Checker.check("")
    assert len(checker.dfs) == 0
    assert len(checker.pandas_aliases) == 0
    assert len(checker.diagnostics) == 0


def test_check_no_pandas_code():
    """Test checking code without pandas."""
    code = """
x = 5
y = x + 10
print(y)
"""
    checker = Checker.check(code)
    assert len(checker.dfs) == 0
    assert len(checker.pandas_aliases) == 0
    assert len(checker.diagnostics) == 0


# --- Import detection tests ---


def test_import_alias():
    code = "import pandas as pd"
    fc = Checker.check(code)
    assert fc.pandas_aliases == {"pd"}


def test_import_full():
    code = "import pandas"
    fc = Checker.check(code)
    assert fc.pandas_aliases == {"pandas"}


def test_check_pandas_alias():
    """Test that pandas imports with aliases are handled correctly."""
    code = """
import pandas as pd_alias
data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
df = pd_alias.DataFrame(data)
result = df['col1']
"""
    checker = Checker.check(code)
    assert len(checker.dfs) == 1
    assert "df" in checker.dfs
    assert "pd_alias" in checker.pandas_aliases
    assert len(checker.diagnostics) == 0


# --- DataFrame initialization tests ---


def test_frame_init_dict_arg():
    code = """
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b"}


def test_frame_init_list_of_dict_arg():
    code = """
import pandas as pd

df = pd.DataFrame([{"a": 1, "b": 4 }, {"a": 2, "b": 5 }, {"a": 3, "b": 6 }])
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b"}


def test_frame_init_dict_var_arg():
    code = """
import pandas as pd

data = {"a": [1, 2, 3], "b": [4, 5, 6]}
df = pd.DataFrame(data)
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b"}


# --- Frame history/tracking tests ---


def test_simple_frame_history():
    code = """
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert set(tracker.columns.keys()) == {"a", "b"}


# --- read_csv tests (non-feature tests) ---


def test_read_csv_no_usecols():
    code = f"""
import pandas as pd

df = pd.read_csv("{CSV_TEST_FILE}")
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == set()


def test_read_csv_usecols_with_var():
    code = f"""
import pandas as pd
a = 'a'
df = pd.read_csv("{CSV_TEST_FILE}", usecols=[a, 'b', 'c'])
"""
    fc = Checker.check(code)
    assert set(fc.dfs.keys()) == {"df"}
    tracker = fc.dfs.get("df")
    assert tracker is not None
    assert tracker.id_ == "df"
    assert set(tracker.columns.keys()) == {"a", "b", "c"}


# --- DataFrame method call semantics ---


def test_inplace_method_mutates_source_not_target():
    """`x = df.insert(...)` mutates df; x is None at runtime, not a DataFrame."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1]})
x = df.insert(0, 'B', [2])
df['B']
"""
    checker = Checker.check(code)
    assert set(checker.dfs["df"].columns) == {"A", "B"}
    assert "x" not in checker.dfs
    assert len(checker.diagnostics) == 0


def test_returned_dataframe_leaves_source_untouched():
    """`df2 = df.assign(...)` binds the new frame to df2 only."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1]})
df2 = df.assign(B=[2])
df['B']
"""
    checker = Checker.check(code)
    assert set(checker.dfs["df"].columns) == {"A"}
    assert set(checker.dfs["df2"].columns) == {"A", "B"}
    assert len(checker.diagnostics) == 1


def test_inplace_method_keeps_column_dependencies():
    """In-place updates preserve dependencies recorded for surviving columns."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1]})
df['C'] = df['A']
df.insert(0, 'B', [2])
"""
    checker = Checker.check(code)
    assert checker.dfs["df"].columns["C"] == {"A"}


def test_self_assigned_method_keeps_column_dependencies():
    """`df = df.method(...)` updates the tracker in place, keeping dependencies."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1]})
df['C'] = df['A']
df = df.assign(B=[2])
"""
    checker = Checker.check(code)
    assert checker.dfs["df"].columns["C"] == {"A"}


# --- Untracked variables (#135) ---


def test_assign_to_frame_without_schema_is_not_reported():
    """A frame read without usecols has no schema; assignments aren't flagged."""
    code = """
import pandas as pd
df = pd.read_csv("file.csv")
df["a"]
df["a"] = df["b"]
"""
    checker = Checker.check(code)
    assert len(checker.diagnostics) == 0


def test_assign_to_dict_is_not_reported():
    """Subscript assignment on a non-DataFrame variable isn't flagged."""
    code = """
config = {}
config["x"] = 1
config["y"] = config["x"]
"""
    checker = Checker.check(code)
    assert len(checker.diagnostics) == 0


def test_assign_from_untracked_variable_keeps_target_column():
    """df['C'] = d['x'] still adds 'C' when d is not a tracked DataFrame."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
d = {"x": 1}
df["C"] = d["x"]
df["C"]
"""
    checker = Checker.check(code)
    assert set(checker.dfs["df"].columns) == {"A", "C"}
    assert len(checker.diagnostics) == 0


def test_frame_init_dict_with_variable_key():
    """Dict keys that are variables resolve to their values (#37)."""
    code = """
import pandas as pd
col1 = "a"
df = pd.DataFrame({col1: [1, 2, 3], "b": [4, 5, 6]})
df["a"]
"""
    checker = Checker.check(code)
    assert set(checker.dfs["df"].columns) == {"a", "b"}
    assert len(checker.diagnostics) == 0


# --- Rebinding tracked frames ---


def test_rebinding_frame_to_unrelated_value_stops_tracking():
    """df = load() replaces the schema we knew about; don't use the stale one."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df = load()
df["B"]
"""
    checker = Checker.check(code)
    assert "df" not in checker.dfs
    assert len(checker.diagnostics) == 0


def test_rebinding_frame_to_derived_value_keeps_tracking():
    """df = df[mask] keeps the columns, so the frame is still checked."""
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
df = df[df["A"] > 0]
df["B"]
"""
    checker = Checker.check(code)
    assert set(checker.dfs["df"].columns) == {"A"}
    assert len(checker.diagnostics) == 1


# --- Bound names are declared ---


def test_bound_names_are_not_reported_as_undeclared():
    """Parameters, loop targets, with-targets and imports are known names."""
    code = """
import json
from collections import OrderedDict as od
def f(d, *, opts):
    d["x"] = 1
    opts["y"] = d["x"]
for row in rows:
    row["z"] = 1
with open("f") as fh:
    fh["w"] = 1
json["a"] = 1
od["b"] = 1
"""
    checker = Checker.check(code)
    assert len(checker.diagnostics) == 0


def test_unbound_name_is_still_reported_as_undeclared():
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1]})
unknown_df["column"] = df["A"]
"""
    checker = Checker.check(code)
    assert len(checker.diagnostics) == 1
    assert "not declared" in checker.diagnostics[0].message
