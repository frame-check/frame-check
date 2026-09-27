"""
Extractors for identifying DataFrame column references in Python AST.

This package provides extractors for various AST expression patterns.
Each extractor handles a specific pattern type (subscripts, binary operations, etc.)
and returns structured `ColumnRef` objects.

Usage:
    The main entry point is `Extractor.extract()`:

    >>> from frame_check_core.extractors import Extractor
    >>> import ast
    >>> expr = ast.parse("df['A'] + df['B']", mode="eval").body
    >>> refs = Extractor.extract(expr)
    >>> [ref.col_names[0] for ref in refs]
    ['A', 'B']

    Or use the module-level convenience function:

    >>> from frame_check_core.extractors import extract
    >>> extract(expr)
    [ColumnRef(...), ColumnRef(...)]

Adding new extractors:
    1. Create a new module in this package with your extractor function
    2. Import it in registry.py
    3. Add it to the EXTRACTORS list in registry.py

The EXTRACTORS list in registry.py defines which extractors are used
and in what order. Extractors earlier in the list are tried first.
"""

import ast

from frame_check_core.refs import ColumnRef

# Individual extractors (for direct use)
from .binop import extract_column_refs_from_binop
from .column import extract_column_ref, extract_single_column_ref

# Registry
from .registry import Extractor

__all__ = [
    # Types
    "ColumnRef",
    # Registry
    "Extractor",
    "extract",
    # Individual extractors (for direct use)
    "extract_column_ref",
    "extract_column_refs_from_binop",
    "extract_single_column_ref",
]


def extract(node: ast.expr) -> list[ColumnRef] | None:
    """
    Extract column references from any recognized RHS expression pattern.

    This function is a convenience wrapper for `Extractor.extract()`
    which tries all registered extractors in list order.

    Args:
        node: The AST expression to analyze (typically the RHS of an assignment).

    Returns:
        A list of `ColumnRef` objects if the pattern is recognized.
        Returns `None` if:
        - The expression doesn't match any known pattern
        - The expression contains unsupported operands (e.g., constants, variables)

    Example:
        >>> import ast
        >>> # Simple column reference
        >>> expr = ast.parse("df['A']", mode="eval").body
        >>> refs = extract(expr)
        >>> len(refs)
        1
        >>> refs[0].col_names
        ['A']

        >>> # Binary operation
        >>> expr = ast.parse("df['A'] * df['B']", mode="eval").body
        >>> refs = extract(expr)
        >>> sorted(ref.col_names[0] for ref in refs)
        ['A', 'B']

        >>> # Unsupported pattern
        >>> expr = ast.parse("df['A'] + 1", mode="eval").body
        >>> extract(expr) is None
        True
    """
    return Extractor.extract(node)
