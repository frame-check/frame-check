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
from typing import Any, Self

from frame_check_core import diagnostic
from frame_check_core.diagnostic.region import CodeRegion
from frame_check_core.extractors import extract, extract_single_column_ref

# Ensure pandas and dataframe handlers are registered
from frame_check_core.handlers import pandas as _pandas  # noqa: F401
from frame_check_core.handlers.models import DF, PD, Result, get_result
from frame_check_core.tracker import Relaxed, Strict, Tracker

# Statements after which control doesn't continue with the next one
_EXITS = (ast.Return, ast.Raise, ast.Continue, ast.Break)

# Marks a name that has no definition (None is a valid definition)
_MISSING: Any = object()


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
) -> set[str]:
    """
    Return the names local to a function (excluding global/nonlocal ones).

    Only statements can bind names in a function body (walrus targets
    aside), so walking statement blocks is enough and much cheaper than
    visiting every expression node.
    """
    local_names = _parameter_names(node.args)
    declared: set[str] = set()
    _collect_bound_names(node.body, local_names, declared)
    return local_names - declared


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
            case ast.Call(func=ast.Name(id=name)):
                # DataFrame(...) with a name imported from pandas
                return name
            case ast.Name(id=name):
                return name
            case _:
                return None


# Methods whose keys may name index levels as well as columns
_INDEX_KEY_METHODS = frozenset({"groupby", "merge", "join", "sort_values"})

# Methods combining the frame with other frames passed as arguments
_COMBINING_METHODS = frozenset({"merge", "join"})

# Methods that may give a frame a named index
_INDEX_CHANGING_METHODS = frozenset(
    {"set_index", "rename_axis", "set_axis", "pivot", "pivot_table", "stack", "unstack"}
)


def _unnamed_index(value: ast.expr) -> bool:
    """Whether an `index=` argument certainly has no name (a literal, range)."""
    match value:
        case ast.List() | ast.Tuple() | ast.Constant(value=None):
            return True
        case ast.Call(func=ast.Name(id="range")):
            return True
    return False


def _index_attribute_root(target: ast.expr) -> str | None:
    """`df` for targets like `df.index`, `df.index.name`, `df.index.names`."""
    while isinstance(target, ast.Attribute):
        if target.attr == "index" and isinstance(target.value, ast.Name):
            return target.value.id
        target = target.value
    return None


def _columns_target(target: ast.expr) -> bool:
    """Whether `target` is `df.columns`."""
    match target:
        case ast.Attribute(value=ast.Name(), attr="columns"):
            return True
    return False


def _indexer_target(target: ast.expr) -> bool:
    """Whether `target` is `df.loc[rows, cols]` or `df.at[row, col]`."""
    match target:
        case ast.Subscript(
            value=ast.Attribute(value=ast.Name(), attr="loc" | "at"),
            slice=ast.Tuple(elts=[_, _]),
        ):
            return True
    return False


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
        # Frames whose index may carry names (see _index_may_be_named)
        self._named_index: set[str] = set()
        # Names imported from pandas: local name -> pandas name
        # (`from pandas import DataFrame, read_csv as rc`)
        self.pandas_names: dict[str, str] = {}
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
            visitor = getattr(
                type(self), "visit_" + cls.__name__, type(self).generic_visit
            )
            self._dispatch[cls] = visitor
        visitor(self, node)

    def generic_visit(self, node: ast.AST) -> None:
        """
        Visit all child nodes.

        Same as `ast.NodeVisitor.generic_visit`, without its `iter_fields`
        generator, which is the single largest cost of a check.
        """
        visit = self.visit
        for field in node._fields:
            value = getattr(node, field, None)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        visit(item)
            elif isinstance(value, ast.AST):
                visit(value)

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
        self._visit_scope(node, _function_locals(node), deferred=True)

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
                # Rebound through `global`, or changed in place in a way we
                # can't follow (e.g. df.columns = load()): unknown now
                saved_dfs.pop(name, None)
                continue
            outer = saved_dfs[name]
            for column in copy.columns.keys() - outer.columns.keys():
                outer.try_add(column)

    def _is_frame_root(self, name: str | None) -> bool:
        """Whether a chain starting at `name` can evaluate to a frame."""
        return name is not None and (
            name in self.dfs or name in self.pandas_aliases or name in self.pandas_names
        )

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

        Records names imported with `from pandas import DataFrame, read_csv`
        so calls like `DataFrame(...)` are handled like `pd.DataFrame(...)`.

        Args:
            node: The import-from AST node.
        """
        for alias in node.names:
            local_name = alias.asname or alias.name
            self._bound.add(local_name)
            if node.module == "pandas" and alias.name != "*":
                self.pandas_names[local_name] = alias.name
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

        if not self._is_pandas_call(node.value):
            return False

        frame = self._eval_frame(node.value)
        if frame is None:
            return False

        # Register the new DataFrame
        self.dfs[target.id] = Tracker.new_with_columns(
            target.id, columns=list(frame[1])
        )
        self._update_named_index(target.id, node.value)
        return True

    def _is_pandas_call(self, expr: ast.expr) -> bool:
        """Whether `expr` is a call like `pd.read_csv(...)`."""
        return (
            isinstance(expr, ast.Call) and self._pandas_function(expr.func) is not None
        )

    def _pandas_function(self, func: ast.expr) -> tuple[str, str] | None:
        """
        Resolve the function of a call to a pandas function, if it is one.

        Handles `pd.read_csv`, `pd.DataFrame.from_dict`, and names imported
        from pandas (`read_csv`, `DataFrame.from_dict`).

        Returns:
            (label for diagnostics, registry name) or None.
        """
        match func:
            case ast.Attribute(value=ast.Name(id=module), attr=name) if (
                module in self.pandas_aliases
            ):
                return f"{module}.{name}", name
            case ast.Attribute(
                value=ast.Attribute(value=ast.Name(id=module), attr="DataFrame"),
                attr=constructor,
            ) if module in self.pandas_aliases:
                return f"{module}.DataFrame.{constructor}", f"DataFrame.{constructor}"
            case ast.Name(id=local) if local in self.pandas_names:
                return local, self.pandas_names[local]
            case ast.Attribute(value=ast.Name(id=local), attr=constructor) if (
                self.pandas_names.get(local) == "DataFrame"
            ):
                return f"{local}.{constructor}", f"DataFrame.{constructor}"
        return None

    def _call_pandas(
        self, call: ast.Call, label: str, function_name: str
    ) -> set[str] | None:
        """
        Run the registered handler for a pandas function call.

        Args:
            call: The call AST node, e.g. for `pd.read_csv(...)`.
            label: How the call is referred to in diagnostics (e.g. 'pd.merge').
            function_name: The pandas function being called.

        Returns:
            The created frame's columns, or None if they can't be determined.
        """
        function = PD.get_method(function_name)
        if function is None:
            return None
        self._evaluated_calls.add(id(call))
        created_df, error = function(
            call.args, call.keywords, self.definitions, self._frame_arg
        )
        if function_name in _INDEX_KEY_METHODS and self._may_have_named_index(call):
            # Keys may name index levels, which aren't tracked
            error = None
        if error is not None and error.frame is not None:
            self._report(
                error.frame,
                diagnostic.missing_columns(
                    action=f"{label}()",
                    missing_cols=error.missing,
                    node=call,
                    df_name=error.frame,
                    available_cols=error.available or [],
                ),
            )
        return None if created_df is None else created_df.columns

    def _call_method(
        self, label: str, columns: Iterable[str], call: ast.Call, method_name: str
    ) -> tuple[DF, DF | None, DF | None] | None:
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
            call.args, call.keywords, self.definitions, self._frame_arg
        )
        if method_name in _INDEX_KEY_METHODS and self._may_have_named_index(call):
            # Keys may name index levels, which aren't tracked
            error = None
        if method_name == "reset_index" and self._may_have_named_index(call):
            # The index level names it turns into columns aren't known
            if returned_df is not None and returned_df.columns != method.df.columns:
                returned_df = None
            if updated_df is not None and updated_df.columns != method.df.columns:
                updated_df = None
        if error is not None:
            self._report(
                _root_name(call),
                diagnostic.missing_columns(
                    action=f"{label}.{method_name}()",
                    missing_cols=error.missing,
                    node=call,
                    region=CodeRegion.from_method_call(node=call),
                    df_name=error.frame or label,
                    available_cols=(
                        error.available
                        if error.available is not None
                        else list(method.df.columns)
                    ),
                ),
            )
        return method.df, updated_df, returned_df

    def _frame_arg(self, expr: ast.expr) -> DF | None:
        """Resolve a handler argument to a tracked frame, e.g. `other` in `df.merge(other)`."""
        frame = self._eval_frame(expr)
        return None if frame is None else DF(frame[1], label=frame[0])

    def _eval_frame(self, expr: ast.expr) -> tuple[str, set[str]] | None:
        """
        Evaluate the columns of a DataFrame-valued expression.

        Handles tracked names (`df`), pandas constructors (`pd.DataFrame(...)`),
        column subsets (`df[["A", "B"]]`) and method chains rooted at either
        (`df.assign(A=1).drop(columns="B")`).
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
            case ast.Call(func=func) if (
                pandas_function := self._pandas_function(func)
            ) is not None:
                # pd.DataFrame({...}), pd.read_csv(...), DataFrame.from_dict(...)
                label, function_name = pandas_function
                columns = self._call_pandas(expr, label, function_name)
                if columns is None:
                    return None
                return f"{label}(...)", columns
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
        self._update_named_index(target, value)

    def _update_named_index(self, target: str, value: ast.expr) -> None:
        """Record whether the frame bound to `target` may have a named index."""
        if self._index_may_be_named(value):
            self._named_index.add(target)
        else:
            self._named_index.discard(target)

    def _index_may_be_named(self, expr: ast.expr) -> bool:
        """
        Whether the frame `expr` evaluates to may have a named index.

        Its index levels can then be used as keys in groupby, merge, join or
        sort_values, and reset_index adds columns named after them. The
        index isn't tracked, so this is conservative: a constructor with a
        non-literal `index=`, `read_*(index_col=...)`, an index-changing
        method, or any frame derived from such a frame.

        Only the frame's spine is walked (the receiver chain down to its
        root, plus frame arguments of merges), not every subexpression.
        """
        while True:
            match expr:
                case ast.Call(func=func, args=args, keywords=keywords) if (
                    pandas_function := self._pandas_function(func)
                ) is not None:
                    return self._pandas_call_may_name_index(
                        pandas_function[1], args, keywords
                    )
                case ast.Call(
                    func=ast.Attribute(value=inner, attr=method),
                    args=args,
                    keywords=keywords,
                ):
                    if method in _INDEX_CHANGING_METHODS:
                        return True
                    if method in _COMBINING_METHODS and self._any_named_frame(
                        args, keywords
                    ):
                        return True
                    expr = inner
                case ast.Subscript(value=inner) | ast.Attribute(value=inner):
                    expr = inner
                case ast.Name(id=name):
                    return name in self._named_index
                case _:
                    return False

    def _pandas_call_may_name_index(
        self, name: str, args: list[ast.expr], keywords: list[ast.keyword]
    ) -> bool:
        """`_index_may_be_named` for a pandas function call."""
        index_args = {kw.arg: kw.value for kw in keywords}
        if name == "DataFrame" and len(args) > 1:
            index_args["index"] = args[1]
        for key in ("index", "index_col"):
            value = index_args.get(key)
            if value is not None and not _unnamed_index(value):
                return True
        # Built from other frames: pd.DataFrame(df), pd.merge(a, b), pd.concat([a, b])
        return self._any_named_frame(args, keywords)

    def _any_named_frame(
        self, args: list[ast.expr], keywords: list[ast.keyword]
    ) -> bool:
        """Whether a frame passed in `args`/`keywords` may have a named index."""
        values = [*args, *(kw.value for kw in keywords)]
        for value in values:
            elements = (
                value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            )
            if any(self._index_may_be_named(e) for e in elements):
                return True
        return False

    def _may_have_named_index(self, call: ast.Call) -> bool:
        """Whether a frame used in `call` may have a named index."""
        return any(
            isinstance(node, ast.Name) and node.id in self._named_index
            for node in ast.walk(call)
        )

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

        if tracker is not None:
            if updated_df is None:
                # Changed in place in a way we can't follow: stop tracking
                del self.dfs[label]
            elif updated_df.columns != original_df.columns:
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
            and self._is_frame_root(_root_name(node))
        ):
            # Only chains rooted at a tracked frame or pandas can produce a
            # frame; skipping the rest is cheap (np.where(...), s.str.lower())
            self._eval_frame(node)
        self.generic_visit(node)

    def _apply_unpacked_column_assignment(self, target: ast.Tuple | ast.List) -> None:
        """
        Add the columns written by `df["a"], df["b"] = ...`.

        The targets are writes, so they're excluded from read validation.
        """
        for element in target.elts:
            if isinstance(element, ast.Starred):
                element = element.value
            ref = extract_single_column_ref(element)
            if ref is None:
                continue
            tracker = self.dfs.get(ref.df_name)
            if tracker is None:
                continue
            for column in ref.col_names:
                tracker.try_add(column)
            self._skip_subscripts.add(id(ref.node))

    def _apply_dynamic_column_assignment(self, target: ast.expr) -> None:
        """
        Handle `df[key] = ...` where `key` isn't a literal label.

        A key that resolves to labels through definitions (`name = "x"`)
        adds them. A key derived from the frame itself (a boolean row mask
        like `df[df["A"] > 0] = 0`) adds no columns. Any other key (a loop
        variable, a call) may add any column, so the frame stops being
        tracked.
        """
        match target:
            case ast.Subscript(value=ast.Name(id=df_name), slice=key):
                pass
            case _:
                return
        tracker = self.dfs.get(df_name)
        if tracker is None:
            return

        labels = get_result(key, self.definitions)
        if isinstance(labels, str):
            labels = [labels]
        if (
            isinstance(labels, list)
            and labels
            and all(isinstance(c, str) for c in labels)
        ):
            for column in labels:
                if isinstance(column, str):
                    tracker.try_add(column)
        elif not _references_name(key, df_name):
            del self.dfs[df_name]

    def _apply_indexer_assignment(self, target: ast.expr) -> None:
        """
        Add the columns created by `df.loc[rows, cols] = ...` or `df.at[i, col] = ...`.

        pandas enlarges the frame with new column labels. A slice (`:`)
        selects existing columns. A selector that can't be resolved may add
        unknown columns, so the frame stops being tracked, unless it is
        derived from the frame itself (e.g. a mask over `df.columns`).
        """
        match target:
            case ast.Subscript(
                value=ast.Attribute(value=ast.Name(id=df_name)),
                slice=ast.Tuple(elts=[_, col_selector]),
            ):
                pass
            case _:
                return
        tracker = self.dfs.get(df_name)
        if tracker is None or isinstance(col_selector, ast.Slice):
            return

        labels = get_result(col_selector, self.definitions)
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(labels, list) and all(isinstance(c, str) for c in labels):
            for column in labels:
                if isinstance(column, str):
                    tracker.try_add(column)
        elif not _references_name(col_selector, df_name):
            del self.dfs[df_name]

    def _apply_columns_assignment(self, target: ast.expr, value: ast.expr) -> None:
        """
        Replace a frame's columns for `df.columns = [...]`.

        Names that can't be resolved statically (`df.columns.str.lower()`,
        a comprehension, ...) stop tracking the frame. A length mismatch
        raises at runtime, but isn't reported: the tracked columns can be a
        superset of the real ones, so the count isn't reliable.
        """
        if not isinstance(target, ast.Attribute) or not isinstance(
            target.value, ast.Name
        ):
            return
        df_name = target.value.id
        tracker = self.dfs.get(df_name)
        if tracker is None:
            return

        names = get_result(value, self.definitions)
        if isinstance(names, list) and all(isinstance(n, str) for n in names):
            tracker.set_columns({n for n in names if isinstance(n, str)})
        else:
            del self.dfs[df_name]

    def visit_If(self, node: ast.If) -> None:
        """
        Visit each branch from the same incoming state, then merge the results.

        A frame's columns after the `if` are the union of its columns in every
        branch that can fall through (a missing `else` is an empty branch). A
        frame that isn't tracked in all of them is not tracked afterwards, as
        its schema is unknown. Branches ending in `return`, `raise`,
        `continue` or `break` don't reach the code after the `if`, so they
        are left out of the merge. Diagnostics inside a branch are reported
        against that branch's own state.

        Only names mentioned in the branches can change, so only those are
        saved, restored and merged.

        Args:
            node: The if statement AST node.
        """
        self.visit(node.test)
        branches = (node.body, node.orelse)
        names = {
            child.id
            for branch in branches
            for stmt in branch
            for child in ast.walk(stmt)
            if type(child) is ast.Name
        }
        dfs, definitions, named = self.dfs, self.definitions, self._named_index
        before_dfs = {name: dfs.get(name) for name in names}
        before_defs = {name: definitions.get(name, _MISSING) for name in names}
        before_named = names & named

        outcomes = []
        for branch in branches:
            for name, tracker in before_dfs.items():
                if tracker is None:
                    dfs.pop(name, None)
                else:
                    dfs[name] = tracker.copy()
            for name, value in before_defs.items():
                if value is _MISSING:
                    definitions.pop(name, None)
                else:
                    definitions[name] = value
            named -= names
            named |= before_named
            for stmt in branch:
                self.visit(stmt)
            outcomes.append(
                (
                    bool(branch) and isinstance(branch[-1], _EXITS),
                    {name: dfs.get(name) for name in names},
                    {name: definitions.get(name, _MISSING) for name in names},
                    names & named,
                )
            )

        # Branches that exit don't flow on; if none flows on, the code after
        # the `if` is unreachable and any merge will do
        live = [outcome for outcome in outcomes if not outcome[0]] or outcomes

        named -= names
        for name in names:
            trackers = [outcome[1][name] for outcome in live]
            dfs.pop(name, None)
            if None not in trackers:
                self._merge_trackers(name, before_dfs[name], trackers)  # type: ignore[arg-type]

            values = [outcome[2][name] for outcome in live]
            definitions.pop(name, None)
            first = values[0]
            if first is not _MISSING and all(
                value is first or (value is not _MISSING and value == first)
                for value in values
            ):
                definitions[name] = first  # type: ignore[assignment]
        for outcome in live:
            named |= outcome[3]

    def _merge_trackers(
        self,
        name: str,
        original: "Tracker[Strict] | Tracker[Relaxed] | None",
        trackers: "list[Tracker[Strict] | Tracker[Relaxed]]",
    ) -> None:
        """
        Track `name` with the union of its branch outcomes `trackers`.

        The tracker held before the `if` is kept when there is one, so the
        enclosing function scope can still recognise it (see `_visit_scope`).
        Outcomes of different modes can't be combined; the frame is dropped.
        """
        base = original if original is not None else trackers[0]
        if any(tracker.mode != base.mode for tracker in trackers):
            return
        merged: dict[str, set[str]] = {}
        for tracker in trackers:
            for column, dependencies in tracker.columns.items():
                merged.setdefault(column, set()).update(dependencies)
        base.columns = merged
        self.dfs[name] = base

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
        # df.index = ..., df.index.name = ...: the index may now be named
        for target in node.targets:
            if (name := _index_attribute_root(target)) is not None:
                self._named_index.add(name)

        # Label-based indexer assignment: df.loc[rows, "col"] = v, df.at[i, "col"] = v
        if indexer_targets := [t for t in node.targets if _indexer_target(t)]:
            # The right-hand side is evaluated before the column exists
            self.generic_visit(node)
            for target in indexer_targets:
                self._apply_indexer_assignment(target)
            return

        # Replacing the schema: df.columns = ["a", "b"]
        if columns_targets := [t for t in node.targets if _columns_target(t)]:
            self.generic_visit(node)
            for target in columns_targets:
                self._apply_columns_assignment(target, node.value)
            return

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
            elif var_name in self.dfs and self._index_may_be_named(node.value):
                # e.g. df = df.set_index("a"): still tracked, index now named
                self._named_index.add(var_name)

        # Unpacking into columns: df["a"], df["b"] = ...
        if len(node.targets) == 1 and isinstance(
            node.targets[0], (ast.Tuple, ast.List)
        ):
            self._apply_unpacked_column_assignment(node.targets[0])
            return self.generic_visit(node)

        # Handle column assignments: df['col'] = expr or df[['a', 'b']] = expr
        if len(node.targets) != 1:
            return self.generic_visit(node)

        target_ref = extract_single_column_ref(node.targets[0])
        if target_ref is None:
            self.generic_visit(node)
            self._apply_dynamic_column_assignment(node.targets[0])
            return

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
        # Reads on the target frame are checked here and become dependencies;
        # reads on other frames (df["c"] = other["c"]) are checked against
        # their own frame by visit_Subscript
        read_refs = [r for r in tracked_refs if r.df_name == target_ref.df_name]

        # Every column read on the right-hand side, e.g. both in df[["A", "B"]]
        read_cols = [col for ref in read_refs for col in ref.col_names]

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
