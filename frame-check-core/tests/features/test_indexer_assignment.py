"""Feature tests for label-based indexer assignment (CAM-3, CAM-5, CAM-17)."""

import pytest
from frame_check_core.checker import Checker

SETUP = """
import pandas as pd
df = pd.DataFrame({"A": [1, -1], "B": [2, 3]})
"""


def _columns_after(stmt: str) -> set[str] | None:
    fc = Checker.check(SETUP + stmt + "\n")
    tracker = fc.dfs.get("df")
    return None if tracker is None else set(tracker.columns)


@pytest.mark.support(code="#CAM-3")
def test_cam_3_loc_assignment_adds_column():
    """df.loc[:, 'new'] = value"""
    fc = Checker.check(SETUP + 'df.loc[:, "new"] = 1\ndf["new"]\n')
    assert set(fc.dfs["df"].columns) == {"A", "B", "new"}
    assert fc.diagnostics == []


@pytest.mark.support(code="#CAM-5")
def test_cam_5_at_assignment_adds_column():
    """df.at[row, 'col'] = value"""
    fc = Checker.check(SETUP + 'df.at[0, "cell"] = 1\ndf["cell"]\n')
    assert set(fc.dfs["df"].columns) == {"A", "B", "cell"}
    assert fc.diagnostics == []


@pytest.mark.support(code="#CAM-17")
def test_cam_17_conditional_assignment_adds_column():
    """df.loc[df['A'] > 0, 'flag'] = 1"""
    fc = Checker.check(SETUP + 'df.loc[df["A"] > 0, "flag"] = 1\ndf["flag"]\n')
    assert set(fc.dfs["df"].columns) == {"A", "B", "flag"}
    assert fc.diagnostics == []


@pytest.mark.support(code="#CAM-17-1")
def test_cam_17_1_condition_is_still_checked():
    """The row mask is a read: a missing column there is reported."""
    fc = Checker.check(SETUP + 'df.loc[df["Z"] > 0, "flag"] = 1\n')
    assert len(fc.diagnostics) == 1
    assert "Column 'Z' does not exist" in fc.diagnostics[0].message


@pytest.mark.parametrize(
    ("stmt", "expected"),
    [
        ('df.loc[:, ["x", "y"]] = 0', {"A", "B", "x", "y"}),
        ('cols = ["x"]\ndf.loc[:, cols] = 0', {"A", "B", "x"}),
        ('df.loc[0, "A"] = 5', {"A", "B"}),
        ('df.loc[df["A"] > 0] = 0', {"A", "B"}),
        ("df.loc[:, :] = 0", {"A", "B"}),
        ('df.loc[:, df.columns.str.startswith("A")] = 0', {"A", "B"}),
        ("df.iloc[:, 0] = 1", {"A", "B"}),
        ("df.iat[0, 0] = 1", {"A", "B"}),
        ("df.loc[:, name] = 0", None),
    ],
)
def test_indexer_assignment_columns(stmt: str, expected: set[str] | None):
    """Existing labels, slices, row-only and positional indexers add nothing;
    an unresolvable label stops tracking (it may add any column)."""
    assert _columns_after(stmt) == expected


def test_right_hand_side_is_checked_before_the_column_exists():
    fc = Checker.check(SETUP + 'df.loc[:, "n"] = df["n"]\n')
    assert len(fc.diagnostics) == 1
    assert "Column 'n' does not exist" in fc.diagnostics[0].message
