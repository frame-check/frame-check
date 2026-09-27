"""
Diagnostic message generators for DataFrame column validation errors.

This module provides factory functions that create `Diagnostic` objects
for various error conditions detected during AST analysis. Each function
generates a user-friendly error message with:

- Clear description of the error
- Suggestions for similar column names (using Jaro-Winkler similarity)
- List of available columns for reference

The diagnostic messages follow a consistent format designed to be
expressive and actionable, helping users quickly identify and fix
issues in their DataFrame operations.

Example CLI output:
    my_script.py:10:4: Cannot assign to df['Total']: column 'Amt' does not exist. Did you mean 'Amount'?
       |
    10 | df['Total'] = df['Amt']
       |               ^^^^^^^^^
       |
       = available: Amount, Price, Quantity
"""

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from frame_check_core.util.col_similarity import zero_deps_jaro_winkler

from .region import CodeRegion


@dataclass(frozen=True, slots=True)
class IllegalAccess:
    """A call that references columns missing from a DataFrame."""

    missing: list[str]
    # Label of the frame the columns are missing from, when it isn't the
    # frame the method is called on (e.g. the right side of a merge)
    frame: str | None = None
    # Columns of that frame, for the "available columns" hint
    available: list[str] | None = None


class Severity(StrEnum):
    ERROR = "error"


@dataclass(kw_only=True, frozen=True, slots=True)
class CodeSource:
    path: Path | None = field(default=None)
    code: str = ""

    @property
    def is_traceable(self) -> bool:
        """Check if the code is traceable to a source file or code string."""
        return self.path is not None or self.code != ""


@dataclass(kw_only=True)
class Diagnostic:
    message: str
    severity: Severity
    region: CodeRegion
    hint: list[str] | None = None
    name_suggestion: str | None = None
    definition_region: CodeRegion | None = None
    data_src_region: CodeRegion | None = None


def _format_columns(cols: list[str], max_display: int = 8) -> str:
    """
    Format a list of column names for display in diagnostic messages.

    For short lists, displays all columns. For longer lists, shows the
    first 3 and last 2 columns with a count of omitted columns in between.

    Args:
        cols: List of column names to format.
        max_display: Maximum columns to show before truncating (default: 8).

    Returns:
        A formatted string of quoted column names.

    Example:
        >>> _format_columns(['A', 'B', 'C'])
        "'A', 'B', 'C'"

        >>> _format_columns(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        "'A', 'B', 'C', ...+5 more..., 'I', 'J'"
    """
    sorted_cols = sorted(cols)
    if len(sorted_cols) <= max_display:
        return ", ".join(f"'{c}'" for c in sorted_cols)
    # Show first 3, count, last 2: 'A', 'B', 'C', ...+10 more..., 'Y', 'Z'
    first = ", ".join(f"'{c}'" for c in sorted_cols[:3])
    last = ", ".join(f"'{c}'" for c in sorted_cols[-2:])
    remaining = len(sorted_cols) - 5
    return f"{first}, ...+{remaining} more..., {last}"


def df_is_not_declared(node: ast.Subscript) -> Diagnostic:
    """
    Create a diagnostic for referencing an undeclared DataFrame.

    Called when code references a DataFrame variable that hasn't been
    declared or tracked by the checker.

    Args:
        node: The subscript AST node containing the undeclared DataFrame
            reference (e.g., the node for `unknown_df['col']`).

    Returns:
        A Diagnostic with an error message indicating the DataFrame
        is not declared.

    Example:
        For `unknown_df['A']`, produces:
        "DataFrame 'unknown_df' is not declared."
    """
    assert isinstance(node.value, ast.Name), f"Expected Name, got {type(node.value)}"
    return Diagnostic(
        message=f"DataFrame '{node.value.id}' is not declared.",
        severity=Severity.ERROR,
        region=CodeRegion.from_ast_node(node=node.value),
    )


def wrong_assignment(
    write_col: str,
    missing_cols: list[str],
    write_node: ast.Subscript,
    df_name: str,
    available_cols: list[str],
) -> Diagnostic:
    """
    Create a diagnostic for an assignment referencing non-existent columns.

    Called when an assignment like `df['C'] = df['A'] + df['B']` references
    columns that don't exist on the DataFrame.

    Args:
        write_col: The column being assigned to (e.g., 'C').
        missing_cols: List of column names that don't exist but are referenced.
        write_node: The AST node for the assignment target (for location info).
        df_name: The name of the DataFrame variable (e.g., 'df').
        available_cols: List of columns that actually exist on the DataFrame.

    Returns:
        A Diagnostic with:
        - Error message describing the invalid assignment
        - Suggestions for similar column names (if any)
        - List of available columns

    Example:
        For `df['Total'] = df['Ammount']` where 'Amount' exists, the CLI shows:

        my_script.py:5:16: Cannot assign to df['Total']: column 'Ammount' does not exist. Did you mean 'Amount'?
          |
        5 | df['Total'] = df['Ammount']
          |               ^^^^^^^^^^^^^
          |
          = available: Amount, Price, Quantity
    """
    lines: list[str] = []

    # Header: what's being assigned
    if len(missing_cols) == 1:
        lines.append(
            f"Cannot assign to {df_name}[{write_col!r}]: "
            f"column '{missing_cols[0]}' does not exist."
        )
    else:
        formatted = ", ".join(f"'{col}'" for col in missing_cols)
        lines.append(
            f"Cannot assign to {df_name}[{write_col!r}]: "
            f"columns {formatted} do not exist."
        )

    # Suggestions for each missing column
    suggestions: list[str] = []
    first_suggestion: str | None = None
    for col in missing_cols:
        if similar := zero_deps_jaro_winkler(col, available_cols):
            suggestions.append(f"'{col}' -> '{similar}'")
            if first_suggestion is None:
                first_suggestion = similar

    if suggestions:
        lines.append(f"  Did you mean: {', '.join(suggestions)}?")

    # Show available columns
    if available_cols:
        lines.append(f"  Available columns: {_format_columns(available_cols)}")

    return Diagnostic(
        message="\n".join(lines),
        severity=Severity.ERROR,
        region=CodeRegion.from_ast_node(node=write_node),
        name_suggestion=first_suggestion,
    )


def wrong_read(
    col_name: str,
    node: ast.Subscript,
    df_name: str,
    available_cols: list[str],
) -> Diagnostic:
    """
    Create a diagnostic for reading a non-existent column.

    Called when code reads from a column that doesn't exist, such as
    `print(df['X'])` where 'X' is not a valid column.

    Args:
        col_name: The column name being read (e.g., 'X').
        node: The AST node for the subscript (for location info).
        df_name: The name of the DataFrame variable (e.g., 'df').
        available_cols: List of columns that actually exist on the DataFrame.

    Returns:
        A Diagnostic with:
        - Error message describing the invalid read
        - Suggestion for a similar column name (if any)
        - List of available columns

    Example:
        For `print(df['Nmae'])` where 'Name' exists, the CLI shows:

        my_script.py:5:7: Column 'Nmae' does not exist on DataFrame 'df'. Did you mean 'Name'?
          |
        5 | print(df['Nmae'])
          |       ^^^^^^^^^^
          |
          = available: Age, Email, Name
    """
    lines: list[str] = [f"Column '{col_name}' does not exist on DataFrame '{df_name}'."]

    # Suggestion if similar column exists
    similar = zero_deps_jaro_winkler(col_name, available_cols)
    if similar:
        lines.append(f"  Did you mean: '{similar}'?")

    # Show available columns
    if available_cols:
        lines.append(f"  Available columns: {_format_columns(available_cols)}")

    return Diagnostic(
        message="\n".join(lines),
        severity=Severity.ERROR,
        region=CodeRegion.from_ast_node(node=node),
        name_suggestion=similar,
    )


def missing_columns(
    action: str,
    missing_cols: list[str],
    node: ast.expr,
    df_name: str,
    available_cols: list[str],
) -> Diagnostic:
    """
    Create a diagnostic for an operation on non-existent columns.

    Called when an operation that raises `KeyError` for unknown labels, such
    as `df.drop(columns=['X'])`, `df.pop('X')` or `del df['X']`, targets
    columns that don't exist on the DataFrame.

    Args:
        action: Short description of the operation (e.g., 'df.drop()').
        missing_cols: Column names that don't exist but are referenced.
        node: The AST node of the operation (for location info).
        df_name: The name of the DataFrame variable (e.g., 'df').
        available_cols: List of columns that actually exist on the DataFrame.

    Returns:
        A Diagnostic with the error message, suggestions for similar column
        names (if any) and the list of available columns.

    Example:
        For `df.drop(columns=['Nmae'])` where 'Name' exists, produces:
        "df.drop(): column 'Nmae' does not exist on DataFrame 'df'."
    """
    if len(missing_cols) == 1:
        subject = f"column '{missing_cols[0]}' does not exist"
    else:
        formatted = ", ".join(f"'{col}'" for col in missing_cols)
        subject = f"columns {formatted} do not exist"
    lines: list[str] = [f"{action}: {subject} on DataFrame '{df_name}'."]

    suggestions: list[str] = []
    first_suggestion: str | None = None
    for col in missing_cols:
        if similar := zero_deps_jaro_winkler(col, available_cols):
            suggestions.append(f"'{col}' -> '{similar}'")
            if first_suggestion is None:
                first_suggestion = similar

    if suggestions:
        lines.append(f"  Did you mean: {', '.join(suggestions)}?")

    if available_cols:
        lines.append(f"  Available columns: {_format_columns(available_cols)}")

    return Diagnostic(
        message="\n".join(lines),
        severity=Severity.ERROR,
        region=CodeRegion.from_ast_node(node=node),
        name_suggestion=first_suggestion,
    )
