"""
AST-based checker for validating DataFrame column operations.

This module provides the core `Checker` class that walks Python AST nodes
and validates DataFrame column accesses and assignments. It detects:

- DataFrame creation via pandas functions (e.g., `pd.read_csv()`, `pd.DataFrame()`)
- References to non-existent columns in read operations (e.g., `print(df['X'])`)
- References to non-existent columns in assignments (e.g., `df['C'] = df['X']`)
- References to undeclared DataFrames

The checker uses a `Tracker` to maintain the known state of each DataFrame's
columns and validates operations against this state.

Example:
    >>> import ast
    >>> from frame_check_core.checker import Checker, format_diagnostic
    >>>
    >>> code = '''
    ... import pandas as pd
    ... df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    ... df['C'] = df['A'] + df['B']
    ... '''
    >>>
    >>> checker = Checker()
    >>> checker.visit(ast.parse(code))
    >>> for diag in checker.diagnostics:
    ...     print(format_diagnostic(diag, "example.py"))
"""

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Self

from frame_check_core import diagnostic
from frame_check_core.extractors import extract, extract_single_column_ref

# Ensure pandas and dataframe handlers are registered
from frame_check_core.handlers import pandas as _pandas  # noqa: F401
from frame_check_core.handlers.models import DF, PD, Result, get_result
from frame_check_core.tracker import Relaxed, Strict, Tracker


def format_diagnostic(
    diag: diagnostic.Diagnostic,
    file_path: Path | str = "<unknown>",
) -> str:
    """
    Format a diagnostic with file location prefix.

    Produces output in the standard compiler diagnostic format:
    `file:line:col: message`

    Args:
        diag: The diagnostic to format.
        file_path: Path to the source file (for display purposes).

    Returns:
        A formatted string with location prefix and diagnostic message.

    Example:
        >>> diag = ...  # Diagnostic at line 10, column 4
        >>> format_diagnostic(diag, "my_script.py")
        "my_script.py:10:4: Column 'X' does not exist..."
    """
    loc = diag.region.start
    return f"{file_path}:{loc.row}:{loc.col}: {diag.message}"


_generic_visit = ast.NodeVisitor.generic_visit


class Checker(ast.NodeVisitor):
    """
    AST visitor that validates DataFrame column operations.

    Walks the AST and checks that all column references point to columns
    that exist (or will exist) on their respective DataFrames. Collects
    diagnostics for any invalid operations found.

    Attributes:
        diagnostics: List of diagnostics collected during AST traversal.
        dfs: Mapping of DataFrame variable names to their column trackers.
        pandas_aliases: Set of aliases used for pandas (e.g., {'pd', 'pandas'}).
        definitions: Mapping of variable names to their resolved values.

    Example:
        >>> import ast
        >>> checker = Checker()
        >>> checker.visit(ast.parse("import pandas as pd\\ndf = pd.DataFrame({'A': [1]})"))
        >>> 'df' in checker.dfs
        True
    """

    def __init__(self) -> None:
        """
        Initialize the checker with empty state.

        Creates empty diagnostics list, DataFrame trackers, and import
        tracking. DataFrames are discovered dynamically by analyzing
        pandas function calls like `pd.read_csv()` or `pd.DataFrame()`.
        """
        self._dispatch: dict[type[ast.AST], Callable[[Checker, ast.AST], None]] = {}
        self._skip_subscripts: set[int] = set()
        self.diagnostics: list[diagnostic.Diagnostic] = []
        self.dfs: dict[str, Tracker[Strict] | Tracker[Relaxed]] = {}
        self.pandas_aliases: set[str] = set()
        self.definitions: dict[str, Result] = {}

    @classmethod
    def check(cls, code: str | Path | ast.Module) -> Self:
        """
        Check the given code for DataFrame column access issues.

        This is the main entry point for the Checker. Parses the provided
        code, analyzes it for potential column access errors in pandas
        DataFrames, and generates diagnostics.

        Args:
            code: The code to check. Can be a string of Python code,
                a file path to a Python file or a AST module object.

        Returns:
            An instance of Checker with the analysis completed and
            diagnostics generated.

        Example:
            >>> checker = Checker.check("import pandas as pd\\ndf = pd.DataFrame({'A': [1]})")
            >>> len(checker.dfs)
            1
        """
        checker = cls()
        if isinstance(code, Path):
            source = code.read_text()
            tree = ast.parse(source, filename=str(code))
        elif isinstance(code, ast.Module):
            tree = code
        else:
            tree = ast.parse(code)
        checker.visit(tree)
        return checker

    def visit(self, node: ast.AST) -> None:
        """
        Dispatch to the `visit_<NodeType>` method, caching the lookup per class.

        `ast.NodeVisitor.visit` builds the method name and resolves it with
        `getattr` for every node; caching by node class avoids that work.
        """
        cls = node.__class__
        try:
            visitor = self._dispatch[cls]
        except KeyError:
            visitor = getattr(type(self), "visit_" + cls.__name__, _generic_visit)
            self._dispatch[cls] = visitor
        visitor(self, node)

    def _skip_leaf(self, node: ast.AST) -> None:
        """Leaf nodes can't contain column references, so don't descend."""

    # `ast.NodeVisitor.visit_Constant` runs deprecation shims on every
    # constant, and `Name` would otherwise visit its `ctx` child.
    visit_Constant = _skip_leaf
    visit_Name = _skip_leaf

    def visit_Import(self, node: ast.Import) -> None:
        """
        Track pandas imports.

        Detects `import pandas` or `import pandas as pd` style imports
        and records the alias used to reference pandas.

        Args:
            node: The import AST node.
        """
        for alias in node.names:
            if alias.name == "pandas":
                # import pandas or import pandas as pd
                self.pandas_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Track pandas imports from 'from' statements.

        Detects `from pandas import DataFrame` style imports. Currently
        tracks but doesn't fully support this pattern.

        Args:
            node: The import-from AST node.
        """
        # TODO: Handle `from pandas import DataFrame` etc.
        self.generic_visit(node)

    def _try_create_dataframe(self, node: ast.Assign) -> bool:
        """
        Attempt to detect and register a DataFrame creation.

        Handles patterns like:
        - `df = pd.read_csv("file.csv", usecols=["A", "B"])`
        - `df = pd.DataFrame({"col1": [1], "col2": [2]})`

        Args:
            node: The assignment AST node to analyze.

        Returns:
            True if a DataFrame was created and registered, False otherwise.
        """
        if len(node.targets) != 1:
            return False

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return False

        df_name = target.id

        # Match: df = pd.something(...)
        match node.value:
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id=module_name), attr=method_name),
                args=args,
                keywords=keywords,
            ):
                # Check if this is a pandas call
                if module_name not in self.pandas_aliases:
                    return False

                # Try to get a handler for this method
                method = PD.get_method(method_name)
                if method is None:
                    return False

                # Call the handler to extract columns
                created_df, _error = method(args, keywords, self.definitions)
                if created_df is None:
                    return False

                # Register the new DataFrame
                self.dfs[df_name] = Tracker.new_with_columns(
                    df_name, columns=list(created_df.columns)
                )
                return True

        return False

    def _try_dataframe_method(self, node: ast.Assign) -> bool:
        """
        Attempt to detect and handle DataFrame method calls.

        Handles patterns like:
        - `df = df.assign(new_col=values)`
        - `df.insert(1, "new_col", values)`

        Args:
            node: The assignment AST node to analyze.

        Returns:
            True if a DataFrame method was handled, False otherwise.
        """
        if len(node.targets) != 1:
            return False

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return False

        result_name = target.id

        # Match: df = df.register(...) or df2 = df.register(...)
        match node.value:
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id=source_df_name), attr=method_name),
                args=args,
                keywords=keywords,
            ):
                # Check if source is a known DataFrame
                if source_df_name not in self.dfs:
                    return False

                tracker = self.dfs[source_df_name]
                current_columns = set(tracker.columns.keys())

                # Create a temporary DF to use the method registry
                temp_df = DF(current_columns)
                method = temp_df.get_method(method_name)
                if method is None:
                    return False

                # Call the handler
                updated_df, returned_df, _error = method(
                    args, keywords, self.definitions
                )

                # If method returns a new DataFrame (like assign), use returned
                # Otherwise use updated (for in-place modifications)
                result_df = returned_df if returned_df is not None else updated_df

                # Update or create the result tracker
                self.dfs[result_name] = Tracker.new_with_columns(
                    result_name, columns=list(result_df.columns)
                )
                return True

        return False

    def visit_Assign(self, node: ast.Assign) -> None:
        """
        Validate assignment statements.

        Handles three types of assignments:
        1. DataFrame creation: `df = pd.read_csv(...)`
        2. DataFrame method calls: `df = df.assign(...)`
        3. Column assignments: `df['col'] = expr` or `df[['a', 'b']] = expr`

        Also tracks simple variable assignments for later resolution.

        Args:
            node: The assignment AST node to validate.

        Note:
            Successfully processed subscript nodes are added to
            `_skip_subscripts` to prevent duplicate diagnostics
            in `visit_Subscript`.
        """
        # Try DataFrame creation first
        if self._try_create_dataframe(node):
            self.generic_visit(node)
            return

        # Try DataFrame method calls
        if self._try_dataframe_method(node):
            self.generic_visit(node)
            return

        # Track simple variable assignments for definition resolution
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            self.definitions[var_name] = get_result(node.value, self.definitions)

        # Handle column assignments: df['col'] = expr or df[['a', 'b']] = expr
        if len(node.targets) != 1:
            return self.generic_visit(node)

        target_ref = extract_single_column_ref(node.targets[0])
        if target_ref is None:
            return self.generic_visit(node)

        if target_ref.df_name not in self.dfs:
            self.diagnostics.append(diagnostic.df_is_not_declared(target_ref.node))
            return self.generic_visit(node)

        tracker = self.dfs[target_ref.df_name]

        read_refs = extract(node.value)
        if read_refs is None:
            # Unknown RHS pattern - just add the column(s) without dependencies
            for col_name in target_ref.col_names:
                tracker.try_add(col_name)
            self._skip_subscripts.add(id(target_ref.node))
            return self.generic_visit(node)

        # Validate all referenced DataFrames exist
        for ref in read_refs:
            if ref.df_name not in self.dfs:
                self.diagnostics.append(diagnostic.df_is_not_declared(ref.node))
                return self.generic_visit(node)

        # RHS refs are always single-column
        read_cols = [r.col_names[0] for r in read_refs]

        # Try to add the first column with dependencies, report error if missing
        if missing := tracker.try_add(target_ref.col_names[0], depends_on=read_cols):
            self.diagnostics.append(
                diagnostic.wrong_assignment(
                    write_col=", ".join(target_ref.col_names),
                    missing_cols=missing,
                    write_node=target_ref.node,
                    df_name=target_ref.df_name,
                    available_cols=list(tracker.columns.keys()),
                )
            )
        else:
            # First column added successfully, add the rest
            for col_name in target_ref.col_names[1:]:
                tracker.try_add(col_name, depends_on=read_cols)

        # Mark subscripts as handled to avoid duplicate diagnostics
        self._skip_subscripts.add(id(target_ref.node))
        self._skip_subscripts.update(id(r.node) for r in read_refs)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """
        Validate a column read operation.

        Handles read access of the form `df['col']` in any context
        (function calls, expressions, etc.). Validates that:

        1. The DataFrame is declared
        2. The column exists on the DataFrame

        Skips processing for:
        - Subscripts already handled in `visit_Assign`
        - Non-column subscripts (e.g., `list[0]`)
        - Multi-column reads (e.g., `df[['a', 'b']]`)

        Args:
            node: The subscript AST node to validate.
        """
        # Skip if already handled in visit_Assign
        if id(node) in self._skip_subscripts:
            return self.generic_visit(node)

        ref = extract_single_column_ref(node)
        if ref is None:
            return self.generic_visit(node)

        if ref.df_name not in self.dfs:
            # DataFrame not declared - might be a non-DataFrame subscript, skip silently
            return self.generic_visit(node)

        tracker = self.dfs[ref.df_name]

        # Only validate single-column reads for now
        if len(ref.col_names) != 1:
            return self.generic_visit(node)

        if missing := tracker.try_get(ref.col_names[0]):
            self.diagnostics.append(
                diagnostic.wrong_read(
                    col_name=missing,
                    node=ref.node,
                    df_name=ref.df_name,
                    available_cols=list(tracker.columns.keys()),
                )
            )

        self.generic_visit(node)


if __name__ == "__main__":
    file_path = "example.py"
    py = """
import pandas as pd

df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df['C'] = df['A'] + df['B']
df['D'] = df['X']  # Error: X doesn't exist
df['E'] = df['A']
print(df['Y'])  # Error: Y doesn't exist
    """

    checker = Checker.check(py)
    for diag in checker.diagnostics:
        print(format_diagnostic(diag, file_path))
