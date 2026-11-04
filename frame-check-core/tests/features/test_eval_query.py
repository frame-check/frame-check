"""Feature tests for column names inside `df.eval()` / `df.query()` (CAM-15).

Most cases are differential: the same snippet runs through pandas and through
the checker. Every diagnostic must correspond to an error pandas really raises.
"""

import pytest
from frame_check_core.checker import Checker
from pandas.errors import UndefinedVariableError

SETUP = """
import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "my col": [1, 2, 3], "s": list("abc")})
threshold = 1
"""

COLUMNS = {"A", "B", "my col", "s"}


def _pandas_namespace(code: str) -> dict:
    """Run a snippet with real pandas (test-only code)."""
    namespace: dict = {}
    exec(SETUP + code, namespace)  # noqa: S102
    return namespace


def _tracked(code: str, name: str) -> set[str] | None:
    fc = Checker.check(SETUP + code)
    assert fc.diagnostics == []
    tracker = fc.dfs.get(name)
    return None if tracker is None else set(tracker.columns)


@pytest.mark.support(code="#CAM-15")
def test_cam_15_eval_inplace_assignment():
    code = 'df.eval("C = A * 2", inplace=True)\ndf["C"]\n'
    fc = Checker.check(SETUP + code)
    assert fc.diagnostics == []
    assert set(fc.dfs["df"].columns) == COLUMNS | {"C"}
    assert set(_pandas_namespace(code)["df"].columns) == COLUMNS | {"C"}


EVAL_FRAME_CASES = [
    "C = A * 2",
    "C = A + 1\nD = C * 2",
    "\n  C = A + 1\n\n  D = C * 2\n",
    "A = A + 1",
    "C = A + 1\nA = C",
    "C = `my col` * 2",
    "C = @threshold",
    "C = 1",
    "C = abs(A) + sqrt(B)",
    "C = A.sum() + B.mean()",
    "C = s.str.len()",
    "C = index",
    "C = A > B and A < 3",
    "C = A * 2  # comment",
]


@pytest.mark.parametrize("expr", EVAL_FRAME_CASES)
def test_cam_15_eval_assignment_returns_new_frame(expr: str):
    code = f"out = df.eval({expr!r})\n"
    expected = set(_pandas_namespace(code)["out"].columns)
    assert _tracked(code, "out") == expected
    # The original frame isn't modified
    assert _tracked(code, "df") == COLUMNS


@pytest.mark.parametrize("expr", EVAL_FRAME_CASES)
def test_cam_15_eval_assignment_inplace(expr: str):
    code = f"df.eval({expr!r}, inplace=True)\n"
    expected = set(_pandas_namespace(code)["df"].columns)
    assert _tracked(code, "df") == expected


@pytest.mark.parametrize("expr", ["A + B", "A > 1", "`my col` + 1", "A.sum()", "1"])
def test_cam_15_eval_expression_returns_series_not_frame(expr: str):
    code = f"out = df.eval({expr!r})\n"
    assert not hasattr(_pandas_namespace(code)["out"], "columns")
    assert _tracked(code, "out") is None
    assert _tracked(code, "df") == COLUMNS


def test_cam_15_eval_series_is_not_a_frame_for_later_accesses():
    code = 'out = df.eval("A + B")\nout["x"]\n'
    assert Checker.check(SETUP + code).diagnostics == []


def test_cam_15_eval_assignment_then_read_is_checked():
    fc = Checker.check(SETUP + 'out = df.eval("C = A * 2")\nout["C"]\nout["D"]\n')
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith("Column 'D' does not exist")


@pytest.mark.parametrize(
    ("method", "expr", "missing"),
    [
        ("eval", "C = Z * 2", "Z"),
        ("eval", "Z + 1", "Z"),
        ("eval", "A + B + Z", "Z"),
        ("eval", "C = A\nD = Z", "Z"),
        ("eval", "C = C + 1", "C"),
        ("eval", "C = A\nD = E + C", "E"),
        ("eval", "A.clip(lower=Z)", "Z"),
        ("eval", "np.sqrt(A)", "np"),  # modules aren't visible: use @np
        ("eval", "A + threshold", "threshold"),  # locals need @
        ("eval", "`no col` + 1", "no col"),
        ("query", "Z > 0", "Z"),
        ("query", "A > 0 and Z < 1", "Z"),
        ("query", "A > threshold", "threshold"),
        ("query", "s.str.contains(Z)", "Z"),
        ("query", "`no col` > 0", "no col"),
        ("query", "A in [Z, 2]", "Z"),
    ],
)
def test_cam_15_missing_name_is_reported(method: str, expr: str, missing: str):
    code = f"out = df.{method}({expr!r})\n"
    with pytest.raises(UndefinedVariableError):
        _pandas_namespace(code)
    fc = Checker.check(SETUP + code)
    assert len(fc.diagnostics) == 1
    first_line = fc.diagnostics[0].message.splitlines()[0]
    assert first_line == (
        f"df.{method}(): column '{missing}' does not exist on DataFrame 'df'."
    )


@pytest.mark.parametrize(
    "code",
    [
        'df.eval("C = Z", inplace=True)',
        'print(df.query("Z > 0"))',
        'df.query("Z > 0", inplace=True)',
        'df[df.eval("Z > 0")]',
        'df.assign(C=1).query("Z > 0")',
    ],
)
def test_cam_15_missing_name_reported_in_any_position(code: str):
    with pytest.raises(UndefinedVariableError):
        _pandas_namespace(code + "\n")
    fc = Checker.check(SETUP + code + "\n")
    assert len(fc.diagnostics) == 1


def test_cam_15_columns_created_by_the_frame_chain_are_visible():
    code = 'out = df.assign(C=1).query("C > 0 and A > 0")\n'
    assert _tracked(code, "out") == COLUMNS | {"C"}
    fc = Checker.check(SETUP + 'out = df.drop(columns="A").query("A > 0")\n')
    assert len(fc.diagnostics) == 1


NO_ERROR_QUERIES = [
    "A > @threshold",
    "A > @threshold and B < @threshold + 10",
    "@threshold < A",
    "A in @threshold_list",
    "`my col` > 1",
    "`my col` > @threshold",
    "`A` > 0",
    "index > 0",
    "ilevel_0 > 0",
    "A > columns.size",
    "A > 0 and clevel_0.size > 0",
    "A > 1 and not B < 2",
    "A.isin([1, 2])",
    "A > A.mean()",
    "s.str.upper() == 'A'",
    "s == 'Z'",
    "s == '@Z'",
    "s == 'it`s'",
    "A > 1  # comment",
    "A.between(1, 2)",
    "abs(A) > 0",
    "A < inf",
    "A == B or B == A",
]


@pytest.mark.parametrize("expr", NO_ERROR_QUERIES)
def test_cam_15_valid_query_is_not_reported(expr: str):
    code = f"threshold_list = [1, 2]\nout = df.query({expr!r})\n"
    namespace = _pandas_namespace(code)
    assert set(namespace["out"].columns) == COLUMNS
    assert _tracked(code, "out") == COLUMNS


@pytest.mark.parametrize("expr", NO_ERROR_QUERIES)
def test_cam_15_valid_eval_is_not_reported(expr: str):
    code = f"threshold_list = [1, 2]\nout = df.eval({expr!r})\n"
    _pandas_namespace(code)
    assert Checker.check(SETUP + code).diagnostics == []


def test_cam_15_backtick_with_escaped_backtick():
    setup = 'df = pd.DataFrame({"a`b": [1, 2], "x": [1, 2]})\n'
    code = setup + 'out = df.query("`a``b` > 1")\n'
    exec("import pandas as pd\n" + code, {})  # noqa: S102
    fc = Checker.check("import pandas as pd\n" + code)
    assert fc.diagnostics == []
    assert set(fc.dfs["out"].columns) == {"a`b", "x"}


@pytest.mark.parametrize(
    "line",
    [
        'df.query("A > 0", inplace=True)',
        'df.query("A > 0", engine="python", parser="pandas")',
    ],
)
def test_cam_15_query_keeps_columns(line: str):
    assert _tracked(line + "\n", "df") == COLUMNS


def test_cam_15_query_result_keeps_columns():
    code = 'df = df.query("A > 0")\n'
    assert _tracked(code, "df") == COLUMNS
    assert set(_pandas_namespace(code)["df"].columns) == COLUMNS


def test_cam_15_query_keeps_added_columns():
    code = 'df["C"] = 1\ndf = df.query("C > 0")\ndf["C"]\n'
    assert _tracked(code, "df") == COLUMNS | {"C"}


@pytest.mark.parametrize(
    "code",
    [
        'out = df.set_index("A").query("A > 1")',
        'out = df.set_index("A").eval("A + B")',
        'idx = df.set_index("A")\nout = idx.query("A > 1")',
        'idx = df.rename_axis("lvl")\nout = idx.query("lvl > 1")',
        'idx = df.rename_axis("lvl")\nout = idx.eval("lvl + A")',
    ],
)
def test_cam_15_index_level_names_are_not_reported(code: str):
    full = code + "\n"
    namespace = _pandas_namespace(full)
    assert namespace["out"] is not None
    assert Checker.check(SETUP + full).diagnostics == []


def test_cam_15_missing_name_on_frame_with_named_index_is_not_reported():
    # `Z` could be an index level, which the checker doesn't track
    code = 'idx = df.set_index("A")\nout = idx.query("Z > 1")\n'
    assert Checker.check(SETUP + code).diagnostics == []


@pytest.mark.parametrize(
    "line",
    [
        "df.query(expr)",
        'df.query(f"A > {n}")',
        'df.query("A > " + str(n))',
        "df.query(expr, inplace=True)",
        'df.query("Z > 1", resolvers=[{"Z": 1}])',
        'df.eval("Z + 1", resolvers=[{"Z": 1}])',
        'df.query("A > (")',
        'df.query("A > 1; Z")',
        'df.query("""A > 1\nand Z""")',
        'df.query("A @ Z")',
        'df.query("[Z for Z in A]")',
        'df.query("Z > 1", parser="python")',
        "df.query(**opts)",
    ],
)
def test_cam_15_unanalyzable_expression_is_not_reported(line: str):
    fc = Checker.check(SETUP + f"expr = load()\nn = load()\n{line}\n")
    assert fc.diagnostics == []
    assert set(fc.dfs["df"].columns) == COLUMNS


@pytest.mark.parametrize(
    "line",
    [
        "out = df.eval(expr)",
        'out = df.eval(f"C = {n}")',
        'out = df.eval("C = (")',
        'out = df.eval("C = A; D = B")',
        'out = df.eval("C = A\\nA + B")',  # pandas raises: not all assignments
        'out = df.eval("`new col` = A")',  # stored under a mangled name
        'out = df.eval("C, D = A, B")',
        'out = df.eval("C.x = A")',
        'out = df.eval("C += A")',
        'out = df.eval("C = [x for x in A]")',
        'out = df.eval("C = A", target=other)',
        'out = df.eval("C = A", resolvers=[{"Z": 1}])',
        "out = df.eval(**opts)",
    ],
)
def test_cam_15_unanalyzable_eval_result_is_untracked(line: str):
    fc = Checker.check(SETUP + f"expr = load()\nn = load()\n{line}\n")
    assert fc.diagnostics == []
    assert "out" not in fc.dfs


@pytest.mark.parametrize(
    "line",
    [
        "df.eval(expr, inplace=True)",
        'df.eval("C = (", inplace=True)',
        'df.eval("C = A", inplace=flag)',
        'df.eval("`new col` = A", inplace=True)',
        "df.eval(**opts)",
    ],
)
def test_cam_15_unanalyzable_inplace_eval_stops_tracking(line: str):
    fc = Checker.check(SETUP + f"expr = load()\nflag = load()\n{line}\n")
    assert fc.diagnostics == []
    assert "df" not in fc.dfs


def test_cam_15_unknown_eval_without_inplace_keeps_original_columns():
    fc = Checker.check(SETUP + "out = df.eval(expr)\n")
    assert set(fc.dfs["df"].columns) == COLUMNS


def test_cam_15_eval_assignment_to_existing_column_keeps_columns():
    code = 'out = df.eval("A = B * 2")\n'
    assert _tracked(code, "out") == COLUMNS
