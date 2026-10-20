"""Tests for names imported with `from pandas import ...`."""

import pytest
from frame_check_core.checker import Checker


@pytest.mark.parametrize(
    "code",
    [
        'from pandas import DataFrame\ndf = DataFrame({"a": [1]})\n',
        'from pandas import DataFrame as DF\ndf = DF({"a": [1]})\n',
        'from pandas import read_csv\ndf = read_csv("f.csv", usecols=["a"])\n',
        'from pandas import DataFrame\ndf = DataFrame.from_dict({"a": [1]})\n',
        (
            'from pandas import DataFrame, merge\nl = DataFrame({"k": [1], "a": [1]})\n'
            'r = DataFrame({"k": [1]})\ndf = merge(l, r, on="k").drop(columns="k")\n'
        ),
        (
            'from pandas import DataFrame, concat\nx = DataFrame({"a": [1]})\n'
            "df = concat([x, x])\n"
        ),
    ],
)
def test_imported_pandas_functions_create_frames(code: str):
    fc = Checker.check(code + 'df["a"]\ndf["missing"]\n')
    assert set(fc.dfs["df"].columns) == {"a"}
    assert len(fc.diagnostics) == 1
    assert (
        "Column 'missing' does not exist on DataFrame 'df'."
        in fc.diagnostics[0].message
    )


def test_errors_in_imported_function_calls_are_reported():
    code = """
from pandas import DataFrame, merge
l = DataFrame({"k": [1]})
r = DataFrame({"k": [1]})
out = merge(l, r, on="kk")
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert fc.diagnostics[0].message.startswith(
        "merge(): column 'kk' does not exist on DataFrame 'l'."
    )


def test_chain_from_imported_constructor_in_an_expression():
    code = """
from pandas import DataFrame
print(DataFrame({"a": [1]}).assign(b=lambda d: d["z"]))
"""
    fc = Checker.check(code)
    assert len(fc.diagnostics) == 1
    assert "column 'z' does not exist" in fc.diagnostics[0].message


@pytest.mark.parametrize(
    "code",
    [
        # Not from pandas
        'from mylib import DataFrame\ndf = DataFrame({"a": [1]})\n',
        'from polars import DataFrame\ndf = DataFrame({"a": [1]})\n',
        # Unregistered pandas function
        'from pandas import read_sql\ndf = read_sql("q", con)\n',
    ],
)
def test_other_imports_are_not_tracked(code: str):
    fc = Checker.check(code + 'df["missing"]\n')
    assert "df" not in fc.dfs
    assert fc.diagnostics == []
