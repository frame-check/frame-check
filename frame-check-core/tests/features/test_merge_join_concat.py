"""Feature tests for merge, join and concat (CAM-12, CAM-13, CAM-14).

Most cases are differential: the same expression runs through pandas and
through the checker, and the resulting column sets must match.
"""

import pytest
from frame_check_core.checker import Checker

SETUP = """
import pandas as pd
L = pd.DataFrame({"key": [1], "a": [1], "v": [1]})
R = pd.DataFrame({"key": [1], "b": [2], "v": [2]})
R2 = pd.DataFrame({"k2": [1], "b": [2], "v": [2]})
"""


def _pandas_columns(expr: str) -> set[str]:
    """Run `expr` with real pandas on the SETUP frames (test-only snippets)."""
    namespace: dict = {}
    exec(SETUP, namespace)  # noqa: S102
    return set(eval(expr, namespace).columns)


def _checker_columns(expr: str) -> set[str] | None:
    fc = Checker.check(SETUP + f"out = {expr}\n")
    assert fc.diagnostics == []
    tracker = fc.dfs.get("out")
    return None if tracker is None else set(tracker.columns)


MERGE_CASES = [
    'L.merge(R, on="key")',
    "L.merge(R)",
    'L.merge(R, on=["key", "v"])',
    'L.merge(R2, left_on="key", right_on="k2")',
    'L.merge(R, on="key", suffixes=("_l", "_r"))',
    'L.merge(R, on="key", suffixes=(None, "_r"))',
    'L.merge(R, how="cross")',
    'L.merge(R, on="key", indicator=True)',
    'L.merge(R, on="key", indicator="src")',
    "L.merge(R, left_index=True, right_index=True)",
    'L.merge(R, "inner", "key")',
    'pd.merge(L, R, "left", ["key", "v"])',
    'L.merge(R2, "inner", None, "key", "k2")',
    'pd.merge(L, R, on="key")',
    'pd.merge(L[["key", "a"]], R.assign(c=1), on="key")',
    'L.merge(R, on="key").drop(columns="v_x")',
]


@pytest.mark.support(code="#CAM-14")
@pytest.mark.parametrize("expr", MERGE_CASES)
def test_cam_14_merge_matches_pandas(expr: str):
    assert _checker_columns(expr) == _pandas_columns(expr)


@pytest.mark.support(code="#CAM-14-1")
def test_cam_14_1_merge_then_read():
    code = SETUP + 'df = L.merge(R, on="key")\ndf["v"]\ndf["v_x"]\n'
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith("Column 'v' does not exist")


@pytest.mark.support(code="#CAM-14-2")
@pytest.mark.parametrize(
    ("expr", "message"),
    [
        (
            'L.merge(R, on="zz")',
            "L.merge(): column 'zz' does not exist on DataFrame 'L'.",
        ),
        (
            'L.merge(R, on="a")',
            "L.merge(): column 'a' does not exist on DataFrame 'R'.",
        ),
        (
            'L.merge(R2, left_on="key", right_on="kk")',
            "L.merge(): column 'kk' does not exist on DataFrame 'R2'.",
        ),
        (
            'pd.merge(L, R, on="b")',
            "pd.merge(): column 'b' does not exist on DataFrame 'L'.",
        ),
    ],
)
def test_cam_14_2_merge_with_missing_key_is_reported(expr: str, message: str):
    with pytest.raises(KeyError):
        _pandas_columns(expr)
    fc = Checker.check(SETUP + f"out = {expr}\n")
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.splitlines()[0] == message


@pytest.mark.support(code="#CAM-14-3")
@pytest.mark.parametrize(
    "expr",
    [
        "L.merge(load())",
        "L.merge(R, on=keys)",
        "L.merge(R, suffixes=sfx)",
        'L.merge(R, left_on="key", right_index=True)',
    ],
)
def test_cam_14_3_unresolvable_merge_stops_tracking(expr: str):
    fc = Checker.check(SETUP + f"out = {expr}\n")
    assert "out" not in fc.dfs
    assert fc.diagnostics == []
