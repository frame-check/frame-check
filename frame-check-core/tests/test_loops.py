"""Tests for loop bodies, whose later iterations see what earlier ones did."""

from frame_check_core.checker import Checker

HEADER = "import pandas as pd\n"


def check(body: str) -> Checker:
    return Checker.check(HEADER + body)


def test_else_branch_reads_column_added_by_an_earlier_iteration():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for i in range(3):
    if i == 0:
        df["b"] = 1
    else:
        df["c"] = df["b"] + 1
""")
    assert checker.diagnostics == []


def test_elif_branch_rebinds_with_column_from_an_earlier_iteration():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for i in range(3):
    if i == 0:
        df["b"] = 1
    elif i == 1:
        df = df.assign(c=df["b"])
""")
    assert checker.diagnostics == []


def test_while_loop_driven_by_a_flag():
    checker = check("""
df = pd.DataFrame({"a": [1]})
first = True
while cond():
    if first:
        df["b"] = 1
        first = False
    else:
        print(df["b"])
""")
    assert checker.diagnostics == []


def test_async_for_loop():
    checker = check("""
async def main(rows):
    df = pd.DataFrame({"a": [1]})
    async for row in rows:
        if row:
            df["b"] = 1
        else:
            print(df["b"])
""")
    assert checker.diagnostics == []


def test_read_before_the_statement_that_adds_the_column():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for i in range(3):
    if i > 0:
        print(df["b"])
    df["b"] = 1
""")
    assert checker.diagnostics == []


def test_write_depending_on_a_column_added_later_in_the_body():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for i in range(3):
    if i > 1:
        print(df["c"])
    if i > 0:
        df["c"] = df["b"]
    df["b"] = 1
""")
    assert checker.diagnostics == []


def test_nested_loops():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for x in range(2):
    for i in range(3):
        if i == 0:
            df["b"] = 1
        else:
            df["c"] = df["b"] + 1
    for j in range(2):
        if j > 0:
            print(df["d"])
        df["d"] = df["c"]
""")
    assert checker.diagnostics == []


def test_deeply_nested_loops_stay_quiet():
    loops = "".join(f"{'    ' * k}for i{k} in range(2):\n" for k in range(30))
    body = "    " * 30
    checker = check(
        'df = pd.DataFrame({"a": [1]})\n'
        + loops
        + f'{body}if i0:\n{body}    print(df["b"])\n{body}df["b"] = 1\n'
    )
    assert checker.diagnostics == []


def test_missing_column_inside_a_loop_is_reported_once():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for i in range(3):
    if i == 0:
        df["b"] = 1
    print(df["zz"])
""")
    assert len(checker.diagnostics) == 1
    assert "zz" in checker.diagnostics[0].message


def test_missing_column_in_nested_loop_is_reported_once():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for i in range(3):
    for j in range(3):
        df["b"] = df["zz"] + 1
        df.drop(columns=["yy"])
while cond():
    df = df.assign(c=df["xx"])
""")
    assert len(checker.diagnostics) == 3


def test_column_never_added_is_reported_in_while_loop():
    checker = check("""
df = pd.DataFrame({"a": [1]})
first = True
while cond():
    if first:
        df["b"] = 1
        first = False
    else:
        print(df["c"])
""")
    assert len(checker.diagnostics) == 1


def test_loop_without_frames_leaves_state_alone():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for i in range(3):
    print(i)
x = df["b"]
""")
    assert len(checker.diagnostics) == 1


def test_columns_after_the_loop():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for i in range(3):
    df["b"] = 1
x = df["b"]
y = df["zz"]
""")
    assert len(checker.diagnostics) == 1
    assert set(checker.dfs["df"].columns) == {"a", "b"}


def test_frame_rebound_to_unknown_in_the_body_is_untracked():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for path in paths:
    print(df["b"])
    df = pd.read_csv(path)
""")
    assert checker.diagnostics == []


def test_loop_carried_definition_is_dropped():
    checker = check("""
df = pd.DataFrame({"a": [1]})
col = "zz"
for i in range(3):
    if i > 0:
        df.drop(columns=col)
    col = "a"
""")
    assert checker.diagnostics == []


def test_loop_else_is_checked():
    checker = check("""
df = pd.DataFrame({"a": [1]})
for i in range(3):
    df["b"] = 1
else:
    print(df["b"])
    print(df["zz"])
""")
    assert len(checker.diagnostics) == 1


def test_loop_in_function():
    checker = check("""
def f():
    df = pd.DataFrame({"a": [1]})
    for i in range(3):
        if i == 0:
            df["b"] = 1
        else:
            print(df["b"])
    return df["b"]
""")
    assert checker.diagnostics == []
