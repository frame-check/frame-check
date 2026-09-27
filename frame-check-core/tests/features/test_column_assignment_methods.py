"""Feature tests for column assignment methods (CAM)."""

import pytest
from frame_check_core.checker import Checker

# --- CAM-1: Direct assignment ---


@pytest.mark.support(code="#CAM-1")
def test_cam_1_direct_assignment():
    """df["c"] = [7, 8, 9]"""
    code = """
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df["c"] = [7, 8, 9]
df["c"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert sorted(df.columns.keys()) == ["a", "b", "c"]
    assert len(fc.diagnostics) == 0


# --- CAM-7: assign method ---


@pytest.mark.support(code="#CAM-7")
def test_cam_7_assign_method():
    """df = df.assign(A=[1, 2, 3])"""
    code = """
import pandas as pd
df = pd.DataFrame({})
df = df.assign(A=[1, 2, 3])
df["A"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"A"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CAM-7-1")
@pytest.mark.xfail(reason="Not implemented", strict=True)
def test_cam_7_1_assign_subscript():
    """df.assign(A=[1, 2, 3])["A"] - chained subscript access"""
    code = """
import pandas as pd
df = pd.DataFrame({})
df.assign(A=[1, 2, 3])["A"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"A"}
    assert len(fc.diagnostics) == 0


@pytest.mark.support(code="#CAM-7-2")
@pytest.mark.xfail(reason="Not implemented", strict=True)
def test_cam_7_2_assign_chain():
    """df = df.assign(A=[1, 2, 3]).assign(B=[4, 5, 6]) - chained assign"""
    code = """
import pandas as pd
df = pd.DataFrame({})
df = df.assign(A=[1, 2, 3]).assign(B=[4, 5, 6])
df["A"]
df["B"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"A", "B"}
    assert len(fc.diagnostics) == 0


# --- CAM-9: insert method ---


@pytest.mark.support(code="#CAM-9")
def test_cam_9_insert_method():
    """df.insert(0, "A", [1, 2, 3])"""
    code = """
import pandas as pd
df = pd.DataFrame({})
df.insert(0, "A", [1, 2, 3])
df["A"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert set(df.columns.keys()) == {"A"}
    assert len(fc.diagnostics) == 0


# --- CAM-10: setitem with list ---


@pytest.mark.support(code="#CAM-10")
def test_cam_10_setitem_with_list():
    """df[["c", "d"]] = [[7, 8, 9], [10, 11, 12]]"""
    code = """
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df[["c", "d"]] = [[7, 8, 9], [10, 11, 12]]
df["c"]
df["d"]
"""
    fc = Checker.check(code)
    df = fc.dfs.get("df")
    assert df is not None
    assert sorted(df.columns.keys()) == ["a", "b", "c", "d"]
    assert len(fc.diagnostics) == 0
