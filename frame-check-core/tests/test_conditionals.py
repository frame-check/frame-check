"""Tests for merging DataFrame schemas across if/elif/else branches."""

from frame_check_core.checker import Checker

HEADER = "import pandas as pd\n"


def check(body: str) -> Checker:
    return Checker.check(HEADER + body)


def test_columns_are_the_union_of_branches():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if c:
    df["B"] = 1
else:
    df["C"] = 2
x = df["A"]
y = df["B"]
z = df["C"]
""")
    assert checker.diagnostics == []
    assert set(checker.dfs["df"].columns) == {"A", "B", "C"}


def test_column_added_in_one_branch_then_read():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if c:
    df["B"] = 1
x = df["B"]
""")
    assert checker.diagnostics == []


def test_column_dropped_in_one_branch_may_still_exist():
    checker = check("""
df = pd.DataFrame({"A": [1], "B": [2]})
if c:
    del df["B"]
x = df["B"]
""")
    assert checker.diagnostics == []


def test_column_missing_in_all_branches_is_reported():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if c:
    df["B"] = 1
else:
    df["C"] = 2
x = df["Z"]
""")
    assert len(checker.diagnostics) == 1


def test_branches_do_not_leak_into_each_other():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if c:
    df["B"] = 1
else:
    x = df["B"]
""")
    assert len(checker.diagnostics) == 1


def test_error_inside_branch_is_reported():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if c:
    x = df["Z"]
""")
    assert len(checker.diagnostics) == 1


def test_if_test_is_checked_as_a_read():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if df["Z"].any():
    pass
""")
    assert len(checker.diagnostics) == 1
    ok = check("""
df = pd.DataFrame({"A": [1]})
if df["A"].any():
    pass
""")
    assert ok.diagnostics == []


def test_elif_test_is_checked_as_a_read():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if c:
    pass
elif df["Z"].any():
    pass
""")
    assert len(checker.diagnostics) == 1


def test_frame_created_in_only_one_branch_is_untracked():
    checker = check("""
if c:
    df = pd.DataFrame({"A": [1]})
x = df["Z"]
""")
    assert "df" not in checker.dfs
    assert checker.diagnostics == []


def test_frame_created_in_every_branch_is_the_union():
    checker = check("""
if c:
    df = pd.DataFrame({"A": [1]})
else:
    df = pd.DataFrame({"B": [1]})
x = df["A"]
y = df["B"]
z = df["Z"]
""")
    assert len(checker.diagnostics) == 1
    assert set(checker.dfs["df"].columns) == {"A", "B"}


def test_frame_rebound_to_unknown_in_one_branch_is_untracked():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if c:
    df = load()
x = df["Z"]
""")
    assert "df" not in checker.dfs
    assert checker.diagnostics == []


def test_branch_that_raises_is_excluded():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if "B" not in df:
    df["X"] = 1
    raise ValueError("no B")
x = df["X"]
""")
    assert len(checker.diagnostics) == 1


def test_else_that_returns_is_excluded():
    checker = check("""
def f(c):
    df = pd.DataFrame({"A": [1]})
    if c:
        df["B"] = 1
    else:
        df["C"] = 1
        return df
    x = df["B"]
    y = df["C"]
""")
    assert len(checker.diagnostics) == 1


def test_continue_and_break_are_excluded():
    checker = check("""
df = pd.DataFrame({"A": [1]})
for i in range(3):
    if c:
        df["B"] = 1
        continue
    elif d:
        df["C"] = 1
        break
    x = df["B"]
""")
    assert len(checker.diagnostics) == 1


def test_all_branches_exit():
    checker = check("""
def f(c):
    df = pd.DataFrame({"A": [1]})
    if c:
        return 1
    else:
        raise ValueError()
""")
    assert checker.diagnostics == []


def test_nested_ifs():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if c:
    if d:
        df["B"] = 1
    else:
        df["C"] = 1
else:
    df["D"] = 1
a = df["B"]
b = df["C"]
e = df["D"]
z = df["Z"]
""")
    assert len(checker.diagnostics) == 1
    assert set(checker.dfs["df"].columns) == {"A", "B", "C", "D"}


def test_elif_chain():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if a:
    df["B"] = 1
elif b:
    df["C"] = 1
elif c:
    df["D"] = 1
else:
    df["E"] = 1
cols = [df["B"], df["C"], df["D"], df["E"]]
z = df["Z"]
""")
    assert len(checker.diagnostics) == 1


def test_if_inside_function():
    checker = check("""
def f(c):
    df = pd.DataFrame({"A": [1]})
    if c:
        df["B"] = 1
    x = df["B"]
    y = df["Z"]
""")
    assert len(checker.diagnostics) == 1


def test_if_inside_function_on_outer_frame():
    checker = check("""
df = pd.DataFrame({"A": [1]})

def f(c):
    if c:
        df["B"] = 1

x = df["B"]
""")
    assert checker.diagnostics == []


def test_dependencies_are_unioned():
    checker = check("""
df = pd.DataFrame({"A": [1], "B": [2]})
if c:
    df["C"] = df["A"]
else:
    df["C"] = df["B"]
""")
    assert checker.dfs["df"].columns["C"] == {"A", "B"}


def test_differing_definitions_become_unknown():
    checker = check("""
df = pd.DataFrame({"A": [1], "B": [2]})
if c:
    cols = ["A"]
else:
    cols = ["B"]
x = df[cols]
""")
    assert "cols" not in checker.definitions
    assert checker.diagnostics == []


def test_identical_definitions_are_kept():
    checker = check("""
if c:
    cols = ["A"]
else:
    cols = ["A"]
""")
    assert checker.definitions["cols"] == ["A"]


def test_named_index_is_unioned():
    checker = check("""
df = pd.DataFrame({"A": [1]})
if c:
    df = df.set_index("A")
""")
    assert "df" in checker._named_index


def test_pandas_test_suite_example():
    checker = check("""
if exp == "a":
    expected = pd.DataFrame({"A": [1], "B": [2], "C": [5]})
elif exp == "empty_cross":
    expected = pd.DataFrame(columns=["A_x", "B", "A_y", "C"])
if how == "outer":
    expected = expected.sort_values("A")
""")
    assert checker.diagnostics == []
