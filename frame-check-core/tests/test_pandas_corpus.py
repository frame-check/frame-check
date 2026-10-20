"""Robustness: check pandas' own test suite, real-world pandas code.

The checker must never crash, whatever the code looks like. (Diagnostics
aren't asserted: many tests there trigger KeyError on purpose, and the
files change with the pandas version.)
"""

import ast
from pathlib import Path

import pandas
from frame_check_core.checker import Checker

PANDAS_TESTS = Path(pandas.__file__).parent / "tests"


def test_checker_does_not_crash_on_pandas_test_suite():
    crashes = []
    for path in sorted(PANDAS_TESTS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        try:
            Checker.check(tree)
        except Exception as e:  # noqa: BLE001 - collect every crash
            crashes.append(f"{path.relative_to(PANDAS_TESTS)}: {type(e).__name__}: {e}")
    assert crashes == []
