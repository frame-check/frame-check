"""Tests for function, lambda and comprehension scopes."""

import pytest
from frame_check_core.checker import Checker

HEADER = 'import pandas as pd\ndf = pd.DataFrame({"A": [1]})\n'


@pytest.mark.parametrize(
    "body",
    [
        # #51: a local frame doesn't replace the module-level one
        'def f():\n    df = pd.DataFrame({"B": [4]})\n    return df\nprint(df["A"])\n',
        # Parameters shadow enclosing frames
        'def clean(df):\n    return df["Z"]\n',
        'def outer():\n    def inner(df):\n        return df["Z"]\n',
        'out = df.assign(B=1, C=lambda df: df["B"])\n',
        'vals = [df["Z"] for df in frames]\n',
        # Functions may run later: enclosing frames are quiet inside them
        'def f():\n    return df["Later"]\ndf["Later"] = 1\n',
        'def f():\n    return df.drop(columns="Later")\n',
        # Columns a function may add are kept; removals are ignored
        'def add():\n    df["X"] = 1\nadd()\nprint(df["X"])\n',
        'def f():\n    del df["A"]\nprint(df["A"])\n',
        # Rebinding through a global declaration stops tracking
        "def reload():\n    global df\n    df = load()\nreload()\nprint(df['New'])\n",
    ],
)
def test_no_false_positive(body: str):
    fc = Checker.check(HEADER + body)
    assert [d.message for d in fc.diagnostics] == []


@pytest.mark.parametrize(
    "body",
    [
        # Frames local to a function are checked
        'def f():\n    local = pd.DataFrame({"A": [1]})\n    return local["Z"]\n',
        # Enclosing frames are still checked after a function definition
        'def f():\n    pass\nprint(df["Z"])\n',
        # Lambdas and comprehensions run in place, so enclosing frames are
        # checked inside them
        'vals = [df["Z"] for _ in range(3)]\n',
        'out = df.assign(C=lambda x: df["Z"])\n',
    ],
)
def test_error_reported(body: str):
    fc = Checker.check(HEADER + body)
    assert len(fc.diagnostics) == 1
    assert "'Z' does not exist" in fc.diagnostics[0].message


def test_function_scope_does_not_leak():
    """Frames and definitions created in a function stay local to it."""
    code = HEADER + 'def f():\n    local = pd.DataFrame({"B": [1]})\n    cols = ["B"]\n'
    fc = Checker.check(code)
    assert set(fc.dfs) == {"df"}
    assert "cols" not in fc.definitions


def test_global_frame_mutated_in_function_merges_columns():
    code = HEADER + 'def add():\n    df["X"] = df["A"]\n'
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns) == {"A", "X"}
