import ast
import os
from pathlib import Path
from typing import Self, override

from .ast.models import (
    DF,
    PD,
    Result,
    get_result,
    is_assigning,
    set_assigning,
    set_result,
)
from .models.diagnostic import CodeSource, Diagnostic, Severity
from .models.history import (
    ColumnHistory,
    ColumnInstance,
    FrameHistory,
    FrameInstance,
    LineIdKey,
)
from .util.col_similarity import zero_deps_jaro_winkler


class FrameChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.import_aliases: dict[str, str] = {}
        """
        This dictionary maps imported module names to their aliases.
        Example: {'pandas': 'pd'}
        If pandas is imported without an alias, like `import pandas`,
        then it will be {'pandas': 'pandas'}
        """

        self.frames: FrameHistory = FrameHistory()
        self.column_accesses: ColumnHistory = ColumnHistory()
        self.definitions: dict[str, Result] = {}
        self.diagnostics: list[Diagnostic] = []
        self.source: CodeSource

    @classmethod
    def check(cls, code: str | ast.Module | Path) -> Self:
        """Check the given code for DataFrame column access issues.
        This is the main entry point for the FrameChecker.

        Parses the provided code, analyzes it for potential column access errors
        in pandas DataFrames, and generates diagnostics.

        Args:
            code (str | ast.Module | Path): The code to check. Can be a string of Python code,
                a file path to a Python file, or an already parsed AST module.

        Returns:
            Self: An instance of FrameChecker with the analysis completed and diagnostics generated.

        Raises:
            TypeError: If the code parameter is not one of the supported types.
        """
        checker = cls()
        if isinstance(code, (str, Path)) and os.path.isfile(str(code)):
            with open(str(code), "r") as f:
                source = f.read()
                tree = ast.parse(source)
            checker.source = CodeSource(path=Path(code), code=source)
        elif isinstance(code, str):
            tree = ast.parse(code)
            checker.source = CodeSource(code=code)
        elif isinstance(code, ast.Module):
            tree = code
            checker.source = CodeSource()
        else:
            raise TypeError(f"Unsupported type: {type(code)}")

        checker.visit(tree)
        # Generate diagnostics
        checker._generate_diagnostics()
        return checker

    def _generate_diagnostics(self):
        """Generate diagnostics from collected column accesses."""
        for access in self.column_accesses.values():
            if access.id not in access.frame.columns:
                data_line = f"DataFrame '{access.frame.id}' created at line {access.frame.defined_lino}"
                data_line += " with columns:"
                hints = [data_line]

                for col in sorted(access.frame.columns):
                    hints.append(f"  • {col}")
                message = f"Column '{access.id}' does not exist"
                similar_col = zero_deps_jaro_winkler(access.id, access.frame.columns)
                if similar_col:
                    message += f", did you mean '{similar_col}'?"
                else:
                    message += "."

                diagnostic = Diagnostic(
                    column_name=access.id,
                    message=message,
                    severity=Severity.ERROR,
                    location=(access.lineno, access.start_col),
                    underline_length=access.underline_length,
                    hint=hints,
                    definition_location=(access.frame.defined_lino, 0),
                )
                self.diagnostics.append(diagnostic)

    def get_cols_from_data_arg(self, arg: ast.AST | None) -> set[str]:
        cols: set[str] = set()

        if arg is None:
            return set()

        if isinstance(arg, ast.Dict):
            keys_nodes = arg.keys

            for k in keys_nodes:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    cols.add(k.value)
            return cols

        # If wrapped around Assign or other, try to get inner Dict
        if isinstance(arg, ast.Assign) and isinstance(arg.value, ast.Dict):
            inner_dict = arg.value
            keys = inner_dict.keys
            for key in keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    cols.add(key.value)
            return cols

        # If wrapped around List
        if isinstance(arg, ast.List):
            for elt in arg.elts:
                if isinstance(elt, ast.Dict):
                    keys = elt.keys
                    for k in keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            cols.add(k.value)
            return cols

        return set()

    def _maybe_create_df(self, node: ast.Assign) -> None:
        if not isinstance(node.value, ast.Call):
            return
        func = node.value.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            return
        method = PD.get_method(func.attr)
        if method is None:
            return
        created, error = method(node.value.args, node.value.keywords)
        if error is not None:
            pass  # TODO
        if created is not None:
            new_frame = FrameInstance.new(
                lineno=node.lineno,
                id=node.targets[0].id if isinstance(node.targets[0], ast.Name) else "",
                data_arg=None,
                keywords=[],
                columns=created.columns,
            )
            self.frames.add(new_frame)

    @override
    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                set_assigning(target)

        # Only visit the value expression to compute results, avoid walking targets
        self.visit(node.value)

        for target in node.targets:
            match target:
                case ast.Subscript(value=ast.Name(id=df_name), slice=subscript_slice):
                    df_name = df_name or ""
                    if last_frame := self.frames.get_before(node.lineno, df_name):
                        # CAM-1: direct column assignment to existing DataFrame
                        new_frame = last_frame.new_instance(
                            lineno=node.lineno, new_columns=subscript_slice
                        )
                        self.frames.add(new_frame)

                case ast.Name(id=id):
                    # any other value assignemnt like foo = "something"
                    self.definitions[id] = get_result(node.value)

        self._maybe_create_df(node)

    @override
    def visit_Import(self, node: ast.Import):
        # No need to traverse children for imports
        for alias in node.names:
            if alias.name == "pandas":
                # Use asname if available, otherwise use the module name
                self.import_aliases["pandas"] = alias.asname or alias.name

    @override
    def visit_Name(self, node):
        # Name has no children; skip generic traversal
        if node.id in self.definitions:
            set_result(node, self.definitions[node.id])

    @override
    def visit_Subscript(self, node: ast.Subscript):
        # Avoid generic traversal; we only need top-level info

        # ignore subscript if it is a column assignment
        if is_assigning(node):
            return

        if not isinstance(node.value, ast.Name):
            return

        frame_id = node.value.id
        if not self.frames.contains_id(frame_id):
            return

        if not isinstance(const := node.slice, ast.Constant) or not isinstance(
            const.value, str
        ):
            return

        # referencing a column by string constant
        frame = self.frames.get_before(node.lineno, frame_id)
        if frame is not None:
            start_col = node.value.end_col_offset or 0
            underline_length = (node.end_col_offset or 0) - start_col
            # record this column access
            self.column_accesses[LineIdKey(node.lineno, const.value)] = ColumnInstance(
                _node=node,
                lineno=node.lineno,
                id=const.value,
                frame=frame,
                start_col=start_col,
                underline_length=underline_length,
            )

    @override
    def visit_Call(self, node: ast.Call):
        # Visit inner call for chains and visit argument expressions so results are available
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Call
        ):
            self.visit(node.func.value)
        # Ensure arguments and keyword values are visited so Name results are resolved for parse_args
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

        if isinstance(node.func, ast.Attribute):
            frame_id = None
            df = None
            match node.func.value:
                case ast.Name():
                    frame_id = node.func.value.id
                    frame = self.frames.get_before(node.lineno, frame_id)
                    if frame is not None:
                        df = DF(frame.columns)
                case ast.Call():
                    result = get_result(node.func.value)
                    if isinstance(result, DF):
                        df = result

            if df is None:
                return

            method = df.get_method(node.func.attr)

            if method is None:
                return

            updated, returned, error = method(node.args, node.keywords)
            if error is not None:
                # self.column_accesses[LineIdKey(node.lineno, "")] = error
                pass
            if returned is not None:
                set_result(node, returned)
            if df.columns != updated.columns and frame is not None:
                new_frame = frame.new_instance(
                    lineno=node.lineno,
                    new_columns=updated.columns,
                )
                self.frames.add(new_frame)
