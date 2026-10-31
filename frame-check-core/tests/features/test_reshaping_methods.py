"""Feature tests for reshaping: melt, get_dummies, pivot, stack and transpose.

melt is differential: the same expression runs through pandas and through the
checker, and the resulting column sets must match. The other operations have
data-dependent columns, so their results must be untracked (and must not
produce diagnostics); labels that pandas rejects with a KeyError are reported.
"""

import pytest
from frame_check_core.checker import Checker

SETUP = """
import pandas as pd
df = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 4], "c": [5, 6]})
"""


def _pandas_columns(expr: str) -> list:
    """Run `expr` with real pandas on the SETUP frame (test-only snippets)."""
    namespace: dict = {}
    exec(SETUP, namespace)  # noqa: S102
    return list(eval(expr, namespace).columns)


def _checker_columns(expr: str) -> set[str] | None:
    fc = Checker.check(SETUP + f"out = {expr}\n")
    assert fc.diagnostics == []
    tracker = fc.dfs.get("out")
    return None if tracker is None else set(tracker.columns)


MELT_CASES = [
    "df.melt()",
    'df.melt(id_vars="id")',
    'df.melt(id_vars=["id", "a"])',
    'df.melt(id_vars=("id",))',
    'df.melt(id_vars=["id"], value_vars=["a"])',
    'df.melt(value_vars=["a", "b"])',
    'df.melt(value_vars="a")',
    "df.melt(id_vars=[])",
    "df.melt(id_vars=None, value_vars=None)",
    'df.melt(id_vars="id", var_name="k", value_name="v")',
    'df.melt(id_vars="id", var_name=None)',
    'df.melt(id_vars="id", ignore_index=False)',
    'df.melt("id", ["a"], "k", "v")',
    'df.melt(id_vars=["a"], value_vars=["a", "b"])',
    'pd.melt(df, id_vars="id")',
    'pd.melt(df, "id", "a", "k")',
    'pd.melt(frame=df, id_vars=["id"], value_name="v")',
    'df.assign(z=1).melt(id_vars="z")',
    'df.melt(id_vars="id").drop(columns="value")',
]


@pytest.mark.parametrize("expr", MELT_CASES)
def test_melt_matches_pandas(expr: str):
    assert _checker_columns(expr) == set(_pandas_columns(expr))


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ('df.melt(id_vars="zz")', "df.melt(): column 'zz' does not exist"),
        ('df.melt(id_vars=["id", "zz"])', "df.melt(): column 'zz' does not exist"),
        ('df.melt(value_vars=["a", "zz"])', "df.melt(): column 'zz' does not exist"),
        ('df.melt(id_vars="zz", value_vars="yy")', "df.melt(): columns 'zz', 'yy'"),
        ('pd.melt(df, id_vars="zz")', "pd.melt(): column 'zz' does not exist"),
        ('pd.melt(df, "id", "zz")', "pd.melt(): column 'zz' does not exist"),
    ],
)
def test_melt_with_missing_label_is_reported(expr: str, message: str):
    with pytest.raises(KeyError):
        _pandas_columns(expr)
    fc = Checker.check(SETUP + f"out = {expr}\n")
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.splitlines()[0].startswith(message)


@pytest.mark.parametrize(
    "expr",
    [
        'df.melt(id_vars="id", var_name="id")',
        'df.melt(id_vars="id", value_name="id")',
        'df.melt(var_name="v", value_name="v")',
    ],
)
def test_melt_with_repeated_label_is_not_tracked(expr: str):
    """pandas 3.0 gives duplicate columns, 3.1 raises ValueError."""
    assert _checker_columns(expr) is None


def test_melt_result_is_tracked():
    fc = Checker.check(SETUP + 'out = df.melt(id_vars="id")\nout["value"]\nout["a"]\n')
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith("Column 'a' does not exist")


@pytest.mark.parametrize(
    "expr",
    [
        'df.melt(id_vars="id", col_level=0)',  # MultiIndex columns
        'df.melt(id_vars="id", value_name=None)',  # NaN label
        'df.melt(id_vars="id", var_name=["x"])',  # ValueError
        "df.melt(id_vars=ids)",
        "df.melt(id_vars=[ids])",
        'df.melt(id_vars="id", var_name=name)',
        "df.melt(**opts)",
        "pd.melt(other, id_vars='id')",
    ],
)
def test_unresolvable_melt_stops_tracking(expr: str):
    fc = Checker.check(SETUP + f"out = {expr}\n")
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


def test_self_rebinding_melt_replaces_the_schema():
    code = SETUP + 'df = df.melt(id_vars="id")\ndf["value"]\ndf["a"]\n'
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith("Column 'a' does not exist")


# p has 4 rows, more than any key list below: pandas groups by a list of keys
# as long as the frame as an array of labels (see pivot_table_error)
PIVOT_SETUP = """
import pandas as pd
p = pd.DataFrame(
    {"i": [1, 1, 2, 2], "c": ["x", "y", "x", "y"], "v": [1, 2, 3, 4], "w": [1, 2, 3, 4]}
)
named = pd.DataFrame({"c": ["x", "y"], "v": [1, 2]}, index=pd.Index([1, 1], name="i"))
"""


def _pandas_run(setup: str, expr: str) -> object:
    namespace: dict = {}
    exec(setup, namespace)  # noqa: S102
    return eval(expr, namespace)


def _check(expr: str, setup: str = PIVOT_SETUP) -> Checker:
    return Checker.check(setup + f"out = {expr}\n")


@pytest.mark.support(code="#CAM-24")
@pytest.mark.parametrize(
    "expr",
    [
        'pd.get_dummies(df, columns=["a"])',
        'pd.get_dummies(df, columns=["a", "b"], prefix="p", drop_first=True)',
        'pd.get_dummies(df, "p", "_", False, ["a"])',
        "pd.get_dummies(df)",
        'pd.get_dummies(df["a"])',
        'pd.get_dummies(df, columns="a")',
        "pd.get_dummies(df, columns=cols)",
        "pd.get_dummies(df, **opts)",
    ],
)
def test_cam_24_get_dummies_result_is_unknown(expr: str):
    fc = _check(expr, SETUP)
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        (
            'pd.get_dummies(df, columns=["zz"])',
            "pd.get_dummies(): column 'zz' does not exist on DataFrame 'df'.",
        ),
        (
            'pd.get_dummies(df, columns=["a", "zz"])',
            "pd.get_dummies(): column 'zz' does not exist on DataFrame 'df'.",
        ),
        (
            'pd.get_dummies(df, None, None, "_", ["zz"])',
            "pd.get_dummies(): column 'zz' does not exist on DataFrame 'df'.",
        ),
    ],
)
def test_get_dummies_with_missing_column_is_reported(expr: str, message: str):
    with pytest.raises(KeyError):
        _pandas_run(SETUP, expr)
    fc = _check(expr, SETUP)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.splitlines()[0] == message


@pytest.mark.support(code="#CAM-22")
@pytest.mark.parametrize(
    "expr",
    [
        'p.pivot(index="i", columns="c", values="v")',
        'p.pivot(index="i", columns="c")',
        'p.pivot(columns="c", values=["v", "w"])',
        'p.pivot(index=["i"], columns=["c"], values="v")',
        'pd.pivot(p, index="i", columns="c", values="v")',
    ],
)
def test_cam_22_pivot_result_is_unknown(expr: str):
    _pandas_run(PIVOT_SETUP, expr)
    fc = _check(expr)
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


def test_pivot_with_unresolved_labels_is_not_validated():
    fc = _check('p.pivot(index=idx, columns=cols, values="v")')
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    ("expr", "missing"),
    [
        ('p.pivot(index="zz", columns="c", values="v")', "zz"),
        ('p.pivot(index="i", columns="zz", values="v")', "zz"),
        ('p.pivot(index="i", columns="c", values="zz")', "zz"),
        ('p.pivot(index="i", columns="c", values=["v", "zz"])', "zz"),
        ('p.pivot(index=["i", "zz"], columns="c")', "zz"),
        ('p.pivot(columns=["zz"])', "zz"),
        ('pd.pivot(p, index="i", columns="c", values="zz")', "zz"),
    ],
)
def test_pivot_with_missing_label_is_reported(expr: str, missing: str):
    with pytest.raises(KeyError):
        _pandas_run(PIVOT_SETUP, expr)
    fc = _check(expr)
    assert len(fc.diagnostics) == 1
    assert f"column '{missing}' does not exist" in fc.diagnostics[0].message
    assert ("pd.pivot()" if "pd.pivot(" in expr else "p.pivot()") in (
        fc.diagnostics[0].message
    )


def test_pivot_does_not_accept_index_level_names():
    # Unlike pivot_table, so a name missing from the columns is reported even
    # on a frame with a named index
    expr = 'named.pivot(index="i", columns="c", values="v")'
    with pytest.raises(KeyError):
        _pandas_run(PIVOT_SETUP, expr)
    assert len(_check(expr).diagnostics) == 1


@pytest.mark.support(code="#EC-7")
@pytest.mark.parametrize(
    "expr",
    [
        'p.pivot_table(index="i", columns="c", values="v")',
        'p.pivot_table(index="i", values=["v", "w"], aggfunc="sum")',
        'p.pivot_table("v", "i", "c")',
        'p.pivot_table(index=["i"], columns="c", values="v", margins=True)',
        'pd.pivot_table(p, index="i", columns="c", values="v")',
        'pd.pivot_table(p, "v", "i", "c")',
        'p.pivot_table(index="i", values="v", aggfunc=agg)',
    ],
)
def test_ec_7_pivot_table_result_is_unknown(expr: str):
    if "agg)" not in expr:
        _pandas_run(PIVOT_SETUP, expr)
    fc = _check(expr)
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    "expr",
    [
        'p.pivot_table(index="zz", columns="c", values="v")',
        'p.pivot_table(index="i", columns="zz", values="v")',
        'p.pivot_table(index="i", columns="c", values="zz")',
        'p.pivot_table(index="i", values=["v", "zz"])',
        'p.pivot_table(index=["i", "zz"], values="v")',
        'p.pivot_table("zz", "i", "c")',
        'p.pivot_table("v", "zz")',
        'pd.pivot_table(p, index="zz", values="v")',
        'pd.pivot_table(p, "v", "i", "zz")',
    ],
)
def test_pivot_table_with_missing_label_is_reported(expr: str):
    with pytest.raises(KeyError):
        _pandas_run(PIVOT_SETUP, expr)
    fc = _check(expr)
    assert len(fc.diagnostics) == 1
    assert "column 'zz' does not exist" in fc.diagnostics[0].message


def test_pivot_table_accepts_index_level_names():
    # The labels name index levels, which aren't tracked: nothing to report
    for expr in (
        'named.pivot_table(index="i", columns="c", values="v")',
        'named.pivot_table(index="i", values="v")',
        'pd.pivot_table(named, index="i", values="v")',
    ):
        _pandas_run(PIVOT_SETUP, expr)
        fc = _check(expr)
        assert fc.diagnostics == [], expr
        assert "out" not in fc.dfs


def test_pivot_table_columns_list_is_not_validated():
    # pandas raises for an unknown label in a list of columns, but the list
    # is conservatively left unchecked
    expr = 'p.pivot_table(index="i", columns=["c", "zz"], values="v")'
    with pytest.raises(KeyError):
        _pandas_run(PIVOT_SETUP, expr)
    fc = _check(expr)
    assert fc.diagnostics == []
    assert "out" not in fc.dfs


def test_pivot_table_key_list_as_long_as_the_frame_is_an_array():
    # The one case where pandas accepts unknown labels: index + columns has
    # exactly as many keys as the frame has rows, so it is used as an array
    # of group labels. This is why a columns list isn't validated.
    setup = 'import pandas as pd\np = pd.DataFrame({"i": [1, 2], "v": [1, 2]})\n'
    _pandas_run(setup, 'p.pivot_table(index="i", columns=["zz"], values="v")')
    fc = _check('p.pivot_table(index="i", columns=["zz"], values="v")', setup)
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    "call",
    [
        'df.pivot(index="id", columns="a", values="b")',
        'df.pivot_table(index="id", columns="a", values="b")',
        'pd.pivot_table(df, index="id", values="b")',
        'pd.get_dummies(df, columns=["a"])',
        'pd.get_dummies(df, columns=["zz"])',
        "pd.get_dummies(df)",
    ],
)
def test_self_rebinding_unknown_result_drops_the_schema(call: str):
    fc = Checker.check(SETUP + f'df = {call}\ndf["a"]\ndf["zz"]\n')
    assert "df" not in fc.dfs
    # Only the invalid get_dummies call itself is reported, not the reads
    assert all("get_dummies()" in d.message for d in fc.diagnostics)


@pytest.mark.support(code="#EC-8")
@pytest.mark.parametrize(
    "expr",
    [
        "df.stack()",
        "df.unstack()",
        "df.stack(future_stack=True)",
        "df.unstack(level=0)",
        "df.stack().unstack()",
        "df.stack().unstack(level=0).reset_index()",
        'df.set_index("id").unstack()',
        'df.set_index(["id", "a"]).unstack("a")',
    ],
)
def test_ec_8_stack_and_unstack_results_are_unknown(expr: str):
    _pandas_run(SETUP, expr)
    fc = _check(expr, SETUP)
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


@pytest.mark.support(code="#CAM-23")
@pytest.mark.parametrize(
    "expr",
    [
        "df.unstack()",
        'df.groupby(["id", "a"])["b"].sum().unstack()',
        'df.set_index(["id", "a"]).unstack()',
    ],
)
def test_cam_23_unstack_result_is_unknown(expr: str):
    _pandas_run(SETUP, expr)
    fc = Checker.check(SETUP + f"out = {expr}\nout['x']\n")
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


@pytest.mark.support(code="#EC-9")
@pytest.mark.parametrize("expr", ["df.T", "df.transpose()", "df.transpose().T"])
def test_ec_9_transpose_result_is_unknown(expr: str):
    _pandas_run(SETUP, expr)
    fc = Checker.check(SETUP + f"out = {expr}\nout['x']\nout.loc['a']\n")
    assert "out" not in fc.dfs
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    "expr",
    ['df.T["x"]', 'df.T.loc["x"]', 'df.transpose()["x"]', 'df.T.assign(z=1)["x"]'],
)
def test_reading_a_transposed_frame_is_not_validated(expr: str):
    fc = Checker.check(SETUP + f"{expr}\n")
    assert fc.diagnostics == []


@pytest.mark.parametrize(
    "stmt",
    [
        "df = df.T",
        "df = df.transpose()",
        "df = df.T.copy()",
        "df = df.T.reset_index()",
        "df = df.stack()",
        "df = df.unstack()",
        "df = df.stack().unstack()",
        "df = df.T[df.T.columns[:1]]",
        'df = df.pivot_table(index="id", values="a").reset_index()',
        'df = df.pivot(columns="a").droplevel(0, axis=1)',
        "df = pd.get_dummies(df).astype(int)",
    ],
)
def test_self_rebinding_reshape_drops_the_schema(stmt: str):
    fc = Checker.check(SETUP + f'{stmt}\ndf["x"]\ndf["a"]\n')
    assert "df" not in fc.dfs
    assert fc.diagnostics == []


def test_derived_frames_still_follow_their_schema():
    # Only the transposing expressions are dropped
    fc = Checker.check(SETUP + 'df = df.assign(z=1).copy()\ndf["z"]\ndf["x"]\n')
    assert len(fc.diagnostics) == 1
