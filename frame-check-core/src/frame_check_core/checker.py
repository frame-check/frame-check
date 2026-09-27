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
from collections.abc import Callable, Iterable
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


def _parameter_names(args: ast.arguments) -> set[str]:
    """Return the names of all parameters in a function signature."""
    names = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _function_locals(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[set[str], set[str]]:
    """
    Return the names local to a function, and those declared global/nonlocal.

    Only statements can bind names in a function body (walrus targets
    aside), so walking statement blocks is enough and much cheaper than
    visiting every expression node.
    """
    local_names = _parameter_names(node.args)
    declared: set[str] = set()
    _collect_bound_names(node.body, local_names, declared)
    return local_names - declared, declared


def _collect_bound_names(
    stmts: list[ast.stmt], names: set[str], declared: set[str]
) -> None:
    """Add the names bound by `stmts` (and nested blocks) to `names`."""
    for stmt in stmts:
        match stmt:
            case ast.Assign(targets=targets):
                for target in targets:
                    _collect_target_names(target, names)
            case ast.AugAssign(target=target) | ast.AnnAssign(target=target):
                _collect_target_names(target, names)
            case ast.For(target=target) | ast.AsyncFor(target=target):
                _collect_target_names(target, names)
                _collect_bound_names(stmt.body, names, declared)
                _collect_bound_names(stmt.orelse, names, declared)
            case ast.With(items=items) | ast.AsyncWith(items=items):
                for item in items:
                    if item.optional_vars is not None:
                        _collect_target_names(item.optional_vars, names)
                _collect_bound_names(stmt.body, names, declared)
            case ast.If(body=body, orelse=orelse) | ast.While(body=body, orelse=orelse):
                _collect_bound_names(body, names, declared)
                _collect_bound_names(orelse, names, declared)
            case ast.Try() | ast.TryStar():
                _collect_bound_names(stmt.body, names, declared)
                for handler in stmt.handlers:
                    if handler.name is not None:
                        names.add(handler.name)
                    _collect_bound_names(handler.body, names, declared)
                _collect_bound_names(stmt.orelse, names, declared)
                _collect_bound_names(stmt.finalbody, names, declared)
            case ast.Match(cases=cases):
                for case in cases:
                    names.update(
                        child.name
                        for child in ast.walk(case.pattern)
                        if isinstance(child, (ast.MatchAs, ast.MatchStar))
                        and child.name is not None
                    )
                    _collect_bound_names(case.body, names, declared)
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                names.add(name)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                names.update(a.asname or a.name.partition(".")[0] for a in aliases)
            case ast.Global(names=global_names) | ast.Nonlocal(names=global_names):
                declared.update(global_names)


def _collect_target_names(target: ast.expr, names: set[str]) -> None:
    """Add the names bound by an assignment target (`a`, `a, *b`, ...)."""
    match target:
        case ast.Name(id=name):
            names.add(name)
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            for elt in elts:
                _collect_target_names(elt, names)
        case ast.Starred(value=value):
            _collect_target_names(value, names)


def _constant_strs(elts: list[ast.expr]) -> list[str] | None:
    """Return the values of a non-empty list of string constants, else None."""
    values = []
    for elt in elts:
        if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
            return None
        values.append(elt.value)
    return values or None


def _root_name(expr: ast.expr) -> str | None:
    """Return the name a frame expression starts from (`df` in `df[...].a()`)."""
    while True:
        match expr:
            case ast.Call(func=ast.Attribute(value=inner)) | ast.Subscript(value=inner):
                expr = inner
            case ast.Name(id=name):
                return name
            case _:
                return None


def _references_name(node: ast.AST, name: str) -> bool:
    """Check whether `name` is used anywhere inside `node`."""
    return any(
        isinstance(child, ast.Name) and child.id == name for child in ast.walk(node)
    )


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
        # _eval_frame results by node id, so each expression's diagnostics
        # are reported once however many code paths evaluate it
        self._frames: dict[int, tuple[str, set[str]] | None] = {}
        # Method calls already run through a handler (see visit_Call)
        self._evaluated_calls: set[int] = set()
        self.diagnostics: list[diagnostic.Diagnostic] = []
        self.dfs: dict[str, Tracker[Strict] | Tracker[Relaxed]] = {}
        self.pandas_aliases: set[str] = set()
        self.definitions: dict[str, Result] = {}
        # Every name bound anywhere (loop targets, parameters, imports, ...)
        self._bound: set[str] = set()
        # Enclosing-scope frames whose diagnostics are suppressed (see _report)
        self._quiet: set[str] = set()

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
    # constant.
    visit_Constant = _skip_leaf

    def visit_Name(self, node: ast.Name) -> None:
        """Record bound names; don't descend into the `ctx` child."""
        if type(node.ctx) is ast.Store:
            self._bound.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        """Record function and lambda parameters as bound names."""
        self._bound.add(node.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """
        Analyze a function body in its own scope.

        Parameters and names assigned in the body are local and shadow
        frames of the enclosing scope. Enclosing frames stay visible but
        quiet, since the function may run after they changed, and columns
        the body may add to them are merged back afterwards.

        Args:
            node: The function definition AST node.
        """
        self._bound.add(node.name)
        local_names, declared = _function_locals(node)
        self._visit_scope(node, local_names, deferred=True, declared=declared)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Analyze a lambda with its parameters shadowing enclosing frames."""
        self._visit_scope(node, _parameter_names(node.args), deferred=False)

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        """Analyze a comprehension with its targets shadowing enclosing frames."""
        local_names = {
            child.id
            for generator in node.generators
            for child in ast.walk(generator.target)
            if isinstance(child, ast.Name)
        }
        self._visit_scope(node, local_names, deferred=False)

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def _visit_scope(
        self,
        node: ast.AST,
        local_names: set[str],
        *,
        deferred: bool,
        declared: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        """
        Visit `node` in a nested scope where `local_names` shadow outer frames.

        Args:
            node: The scope's AST node.
            local_names: Names local to the scope.
            deferred: Whether the scope's code may run later (functions). If
                so, enclosing frames are visible as quiet copies, and columns
                the scope may add are merged into them afterwards. Otherwise
                (lambdas, comprehensions) they are checked as usual.
            declared: Names declared `global`/`nonlocal` in the scope.
        """
        saved_dfs, saved_definitions, saved_quiet = (
            self.dfs,
            self.definitions,
            self._quiet,
        )
        visible = {n: t for n, t in saved_dfs.items() if n not in local_names}
        copies = {}
        if deferred:
            copies = {n: t.copy() for n, t in visible.items()}
            visible = dict(copies)
            self._quiet = (saved_quiet - local_names) | copies.keys()
        else:
            self._quiet = saved_quiet - local_names
        self.dfs = visible
        self.definitions = {
            k: v for k, v in saved_definitions.items() if k not in local_names
        }
        try:
            self.generic_visit(node)
        finally:
            inner_dfs = self.dfs
            self.dfs, self.definitions, self._quiet = (
                saved_dfs,
                saved_definitions,
                saved_quiet,
            )

        # Columns the function may add are kept, and removals ignored, so
        # the enclosing frame is a superset of what it may be after a call.
        for name, copy in copies.items():
            if inner_dfs.get(name) is not copy:
                # Rebound through a global/nonlocal declaration: unknown now
                if name in declared:
                    saved_dfs.pop(name, None)
                continue
            outer = saved_dfs[name]
            for column in copy.columns.keys() - outer.columns.keys():
                outer.try_add(column)

    def _report(self, frame_name: str | None, diag: diagnostic.Diagnostic) -> None:
        """
        Record a diagnostic about the frame named `frame_name`.

        Frames from an enclosing scope seen inside a function body are quiet:
        the function may run after the frame changed, so their state there
        is uncertain.
        """
        if frame_name not in self._quiet:
            self.diagnostics.append(diag)

    def _is_known(self, name: str) -> bool:
        """Whether `name` refers to a variable the code binds somewhere."""
        return name in self.definitions or name in self._bound

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
            self._bound.add(alias.asname or alias.name.partition(".")[0])
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
        for alias in node.names:
            self._bound.add(alias.asname or alias.name)
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

    def _call_method(
        self, label: str, columns: Iterable[str], call: ast.Call, method_name: str
    ) -> tuple[DF, DF, DF | None] | None:
        """
        Run the registered handler for `method_name` on a frame's columns.

        Reports any error returned by the handler as a diagnostic.

        Args:
            label: How the frame is referred to in diagnostics (e.g. 'df').
            columns: The frame's columns before the call.
            call: The method call AST node.
            method_name: The DataFrame method being called.

        Returns:
            The (original, updated, returned) frames, or None if the method
            has no registered handler.
        """
        method = DF(columns).get_method(method_name)
        if method is None:
            return None
        self._evaluated_calls.add(id(call))

        updated_df, returned_df, error = method(
            call.args, call.keywords, self.definitions
        )
        if error is not None:
            self._report(
                _root_name(call),
                diagnostic.missing_columns(
                    action=f"{label}.{method_name}()",
                    missing_cols=error.missing,
                    node=call,
                    df_name=label,
                    available_cols=list(method.df.columns),
                ),
            )
        return method.df, updated_df, returned_df

    def _eval_frame(self, expr: ast.expr) -> tuple[str, set[str]] | None:
        """
        Evaluate the columns of a DataFrame-valued expression.

        Handles tracked names (`df`), column subsets (`df[["A", "B"]]`) and
        method chains rooted at one (`df.assign(A=1).drop(columns="B")`).
        In-place effects on the intermediate frames of a chain are
        discarded, like at runtime. Results are cached per node.

        Args:
            expr: The expression to evaluate.

        Returns:
            A (label, columns) pair, where the label describes the expression
            for diagnostics, or None if it isn't a known DataFrame.
        """
        key = id(expr)
        if key in self._frames:
            return self._frames[key]
        frame = self._frames[key] = self._eval_frame_uncached(expr)
        return frame

    def _eval_frame_uncached(self, expr: ast.expr) -> tuple[str, set[str]] | None:
        """Evaluate `expr` without the cache; see `_eval_frame`."""
        match expr:
            case ast.Name(id=name):
                tracker = self.dfs.get(name)
                if tracker is None:
                    return None
                return name, set(tracker.columns)
            case ast.Call(func=ast.Attribute(value=inner, attr=method_name)):
                frame = self._eval_frame(inner)
                if frame is None:
                    return None
                label, columns = frame
                result = self._call_method(label, columns, expr, method_name)
                if result is None or result[2] is None:
                    return None
                return f"{label}.{method_name}(...)", result[2].columns
            case ast.Subscript(value=inner, slice=ast.List(elts=elts)):
                # df[["A", "B"]]
                frame = self._eval_frame(inner)
                if frame is None:
                    return None
                return self._select(frame, elts, expr, "[[...]]")
            case ast.Subscript(
                value=ast.Attribute(value=inner, attr="loc"),
                slice=ast.Tuple(elts=[_, col_selector]),
            ):
                # df.loc[rows, ["A", "B"]] / df.loc[rows, :]
                frame = self._eval_frame(inner)
                if frame is None:
                    return None
                match col_selector:
                    case ast.List(elts=elts):
                        return self._select(frame, elts, expr, ".loc[...]")
                    case ast.Slice(lower=None, upper=None, step=None):
                        return f"{frame[0]}.loc[...]", frame[1]
                return None
            case _:
                return None

    def _select(
        self,
        frame: tuple[str, set[str]],
        elts: list[ast.expr],
        node: ast.expr,
        suffix: str,
    ) -> tuple[str, set[str]] | None:
        """
        Select the columns listed in `elts` from `frame`.

        Selecting unknown labels raises KeyError, so they are reported.

        Returns:
            The selected frame, or None if the labels aren't all string
            constants.
        """
        col_names = _constant_strs(elts)
        if col_names is None:
            return None
        label, columns = frame
        if missing := [col for col in col_names if col not in columns]:
            self._report(
                _root_name(node),
                diagnostic.missing_columns(
                    action=f"{label}{suffix}",
                    missing_cols=missing,
                    node=node,
                    df_name=label,
                    available_cols=list(columns),
                ),
            )
        return f"{label}{suffix}", set(col_names)

    def _bind_frame(self, target: str, value: ast.expr, columns: Iterable[str]) -> None:
        """
        Bind the columns of a DataFrame-valued expression to `target`.

        For `df = df...` (the value is derived from the target itself), the
        existing tracker is diffed instead of rebuilt, which also keeps
        column dependencies.
        """
        tracker = self.dfs.get(target)
        if tracker is not None and _root_name(value) == target:
            tracker.set_columns(set(columns))
        else:
            self.dfs[target] = Tracker.new_with_columns(target, columns=list(columns))

    def _try_dataframe_method(self, call: ast.expr, target: str | None) -> bool:
        """
        Attempt to detect and handle DataFrame method calls.

        Handles patterns like:
        - `df = df.assign(new_col=values)` (returns a new DataFrame)
        - `df.insert(1, "new_col", values)` (modifies `df` in place)
        - `s = df.pop("col")` (modifies `df` in place, returns a Series)
        - `df = df.assign(A=1).assign(B=2)` (method chains)

        In-place changes are applied to the source DataFrame's tracker, while
        a returned DataFrame is bound to `target`.

        Args:
            call: The expression that may be a DataFrame method call.
            target: Name the call result is assigned to, or None for a
                standalone expression statement.

        Returns:
            True if a DataFrame method was handled, False otherwise.
        """
        # Match: <frame>.method(...)
        match call:
            case ast.Call(func=ast.Attribute(value=source, attr=method_name)):
                pass
            case _:
                return False

        tracker = None
        columns: Iterable[str]
        if isinstance(source, ast.Name):
            tracker = self.dfs.get(source.id)
            if tracker is None:
                return False
            label, columns = source.id, tracker.columns.keys()
        else:
            # Method chain: the source is an intermediate frame
            frame = self._eval_frame(source)
            if frame is None:
                return False
            label, columns = frame

        result = self._call_method(label, columns, call, method_name)
        if result is None:
            return False
        original_df, updated_df, returned_df = result

        if tracker is not None and updated_df.columns != original_df.columns:
            tracker.set_columns(updated_df.columns)

        if target is None:
            return True

        if returned_df is not None:
            self._bind_frame(target, call, returned_df.columns)
        else:
            # The target now holds a non-DataFrame result (e.g. a Series or
            # None); stop tracking it so it can't produce false positives.
            self.dfs.pop(target, None)
        return True

    def visit_Call(self, node: ast.Call) -> None:
        """
        Check DataFrame method calls in any expression position.

        Calls not already handled as a statement or assignment, such as
        `return df.drop(columns="X")` or `print(df.sort_values("X"))`, are
        evaluated so errors in their arguments are reported. Their in-place
        effects are not applied, which keeps the frame a superset.

        Args:
            node: The call AST node.
        """
        if (
            isinstance(node.func, ast.Attribute)
            and id(node) not in self._evaluated_calls
        ):
            self._eval_frame(node)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        """
        Handle standalone expression statements.

        Applies in-place DataFrame method calls such as
        `df.insert(0, "A", values)` or `df.rename(columns=..., inplace=True)`.

        Args:
            node: The expression statement AST node.
        """
        self._try_dataframe_method(node.value, target=None)
        self.generic_visit(node)

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
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and self._try_dataframe_method(node.value, target=node.targets[0].id)
        ):
            self.generic_visit(node)
            return

        # Column subsets: df2 = df[["A", "B"]]
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
            and (frame := self._eval_frame(node.value)) is not None
        ):
            self._bind_frame(node.targets[0].id, node.value, frame[1])
            self.generic_visit(node)
            return

        # Track simple variable assignments for definition resolution
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            self.definitions[var_name] = get_result(node.value, self.definitions)
            # Rebinding a tracked frame to an unrelated value (df = load())
            # makes its schema stale. Values derived from the frame itself
            # (df = df[mask], df = df.sort_values(...)) keep being tracked.
            if var_name in self.dfs and not _references_name(node.value, var_name):
                del self.dfs[var_name]

        # Handle column assignments: df['col'] = expr or df[['a', 'b']] = expr
        if len(node.targets) != 1:
            return self.generic_visit(node)

        target_ref = extract_single_column_ref(node.targets[0])
        if target_ref is None:
            return self.generic_visit(node)

        if target_ref.df_name not in self.dfs:
            # A known variable without a tracked schema (e.g. a dict, or
            # pd.read_csv() without usecols) has nothing to check
            if not self._is_known(target_ref.df_name):
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

        # Validate all referenced DataFrames exist; known variables without a
        # tracked schema are skipped
        tracked_refs = []
        for ref in read_refs:
            if ref.df_name in self.dfs:
                tracked_refs.append(ref)
            elif not self._is_known(ref.df_name):
                self.diagnostics.append(diagnostic.df_is_not_declared(ref.node))
                return self.generic_visit(node)
        read_refs = tracked_refs

        # RHS refs are always single-column
        read_cols = [r.col_names[0] for r in read_refs]

        # Try to add the first column with dependencies, report error if missing
        if missing := tracker.try_add(target_ref.col_names[0], depends_on=read_cols):
            self._report(
                target_ref.df_name,
                diagnostic.wrong_assignment(
                    write_col=", ".join(target_ref.col_names),
                    missing_cols=missing,
                    write_node=target_ref.node,
                    df_name=target_ref.df_name,
                    available_cols=list(tracker.columns.keys()),
                ),
            )
        else:
            # First column added successfully, add the rest
            for col_name in target_ref.col_names[1:]:
                tracker.try_add(col_name, depends_on=read_cols)

        # Mark subscripts as handled to avoid duplicate diagnostics
        self._skip_subscripts.add(id(target_ref.node))
        self._skip_subscripts.update(id(r.node) for r in read_refs)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        """
        Handle column removal via the `del` statement.

        Handles `del df['col']` and `del df['a'], df['b']`, removing the
        columns from the tracked DataFrame.

        Args:
            node: The delete statement AST node.
        """
        for target in node.targets:
            ref = extract_single_column_ref(target)
            if ref is None:
                continue
            tracker = self.dfs.get(ref.df_name)
            if tracker is None:
                continue
            if missing := [c for c in ref.col_names if c not in tracker.columns]:
                self._report(
                    ref.df_name,
                    diagnostic.missing_columns(
                        action="del",
                        missing_cols=missing,
                        node=ref.node,
                        df_name=ref.df_name,
                        available_cols=list(tracker.columns),
                    ),
                )
            tracker.set_columns(tracker.columns.keys() - ref.col_names)
            # Not a read: don't validate the deleted column afterwards
            self._skip_subscripts.add(id(ref.node))
        self.generic_visit(node)

    def _check_chained_read(self, node: ast.Subscript) -> None:
        """
        Validate a single-column read on a method chain, e.g. `df.assign(A=1)["A"]`.

        Args:
            node: The subscript AST node whose value is a call.
        """
        match node.slice:
            case ast.Constant(value=str(col_name)):
                pass
            case _:
                return

        frame = self._eval_frame(node.value)
        if frame is None:
            return
        label, columns = frame
        if col_name not in columns:
            self._report(
                _root_name(node.value),
                diagnostic.wrong_read(
                    col_name=col_name,
                    node=node,
                    df_name=label,
                    available_cols=list(columns),
                ),
            )

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

        Args:
            node: The subscript AST node to validate.
        """
        # Skip if already handled in visit_Assign
        if id(node) in self._skip_subscripts:
            return self.generic_visit(node)

        # Column subsets: df[["A", "B"]], df.loc[:, ["A"]] (validated while
        # evaluating). Only reads: assigning to df.loc[:, [...]] may add columns.
        if isinstance(node.ctx, ast.Load) and (
            isinstance(node.slice, ast.List)
            or (
                isinstance(node.slice, ast.Tuple)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "loc"
            )
        ):
            self._eval_frame(node)
            return self.generic_visit(node)

        # Chained read: df.assign(A=1)["A"]
        if isinstance(node.value, ast.Call):
            self._check_chained_read(node)
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
            self._report(
                ref.df_name,
                diagnostic.wrong_read(
                    col_name=missing,
                    node=ref.node,
                    df_name=ref.df_name,
                    available_cols=list(tracker.columns.keys()),
                ),
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
