import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Union

from ..diagnostic import IllegalAccess


class _Unknown:
    pass


Unknown = _Unknown()  # A value that is either not supported or not provided.


@dataclass(frozen=True, slots=True)
class ColumnLambda:
    """A single-argument lambda, e.g. `lambda x: x["A"] + x["B"]`.

    `columns` are the labels the lambda reads from its argument, in order.
    """

    columns: tuple[str, ...]


Result = Union[
    str,
    bool,
    int,
    dict,
    list,
    ColumnLambda,
    "PD",
    "PDMethod",
    "DF",
    "DFMethod",
    _Unknown,
]

_ASSIGNING_ATTR = "_frame_checker_assigning"

# Resolves an argument expression to a tracked frame, if it is one
FrameResolver = Callable[[ast.expr], "DF | None"]


def is_assigning(node: ast.Subscript) -> bool:
    return getattr(node, _ASSIGNING_ATTR, False)


def set_assigning(node: ast.Subscript) -> None:
    setattr(node, _ASSIGNING_ATTR, True)


def get_value(
    node: ast.AST,
    definitions: dict[str, Result],
    frames: FrameResolver | None = None,
) -> Result:
    # Tracked frames passed as arguments: df.merge(other), pd.DataFrame(df)
    if (
        frames is not None
        and isinstance(node, (ast.Name, ast.Call, ast.Subscript))
        and (frame := frames(node)) is not None
    ):
        return frame

    match node:
        case ast.Constant(value=str(result)):
            return result

        # bool must precede int — bool is a subclass of int
        case ast.Constant(value=bool(result)):
            return result

        case ast.Constant(value=int(result)):
            return result

        case ast.List(elts=elts):
            elements = []
            for elt in elts:
                parsed_elt = get_value(elt, definitions, frames)
                elements.append(parsed_elt)
            return elements

        case ast.Name(id=name):
            # Look up variable value in definitions
            return definitions.get(name, Unknown)

        case ast.Dict(keys=keys, values=values):
            result_dict = {}
            for key_node, value_node in zip(keys, values):
                if key_node is None:
                    continue
                key = get_value(key_node, definitions, frames)
                value = get_value(value_node, definitions, frames)
                result_dict[key] = value
            return result_dict

        case ast.Lambda(args=ast.arguments(args=[ast.arg(arg=param)]), body=body) if (
            not node.args.posonlyargs
            and not node.args.kwonlyargs
            and node.args.vararg is None
            and node.args.kwarg is None
        ):
            return ColumnLambda(columns=_lambda_columns(body, param))

        case _:
            return Unknown


def _lambda_columns(body: ast.expr, param: str) -> tuple[str, ...]:
    """Collect the column labels read as `param["col"]` inside a lambda body."""
    columns: list[str] = []
    for child in ast.walk(body):
        match child:
            case ast.Subscript(
                value=ast.Name(id=name), slice=ast.Constant(value=str(column))
            ) if name == param and column not in columns:
                columns.append(column)
    return tuple(columns)


def get_result(
    node: ast.AST,
    definitions: dict[str, Result],
    frames: FrameResolver | None = None,
) -> Result:
    return get_value(node, definitions, frames)


def parse_args(
    args: list[ast.expr],
    keywords: list[ast.keyword],
    definitions: dict[str, Result],
    frames: FrameResolver | None = None,
) -> tuple[list[Result], dict[str, Result]]:
    argsv = [get_result(arg, definitions, frames) for arg in args]
    keywordsv = {
        kw.arg: get_result(kw.value, definitions, frames)
        for kw in keywords
        if kw.arg is not None
    }
    return argsv, keywordsv


def idx_or_key(
    args: list[Result],
    keywords: dict[str, Result],
    idx: int | None = None,
    key: str | None = None,
) -> Result:
    if idx is not None and len(args) > idx:
        return args[idx]
    if key is not None and key in keywords:
        return keywords[key]
    return Unknown


PDFuncResult = tuple[set[str] | None, IllegalAccess | None]
PDFunc = Callable[[list[Result], dict[str, Result]], PDFuncResult]
PDMethodResult = tuple["DF | None", IllegalAccess | None]


class PD:
    func_registry: ClassVar[dict[str, PDFunc]] = {}

    @classmethod
    def get_method(cls, method_name: str) -> "PDMethod | None":
        if method_name in cls.func_registry:
            return PDMethod(cls.func_registry[method_name])
        return None

    @classmethod
    def register(cls, name: str):
        def decorator(func: PDFunc):
            cls.func_registry[name] = func
            return func

        return decorator


class PDMethod:
    def __init__(self, func: PDFunc):
        self.func = func

    def __call__(
        self,
        args: list[ast.expr],
        keywords: list[ast.keyword],
        definitions: dict[str, Result] | None = None,
        frames: FrameResolver | None = None,
    ) -> PDMethodResult:
        """
        Returns a tuple with two elements:
        - The first element is the created dataframe, if any.
        - The second element is an `IllegalAccess` instance representing an error if the method call is illegal, or `None` if there is no error.
        """
        # If definitions is not provided, fall back to empty dict (for backward compatibility)
        if definitions is None:
            definitions = {}
        argsv, keywordsv = parse_args(args, keywords, definitions, frames)
        returned, error = self.func(argsv, keywordsv)
        return DF(returned) if returned is not None else None, error


DFFuncResult = tuple[set[str], set[str] | None, IllegalAccess | None]
DFFunc = Callable[[set[str], list[Result], dict[str, Result]], DFFuncResult]
DFMethodResult = tuple["DF", "DF | None", IllegalAccess | None]


class DF:
    """This represents a state of a DataFrame. It should be considered immutable."""

    func_registry: ClassVar[dict[str, DFFunc]] = {}

    def __init__(self, columns: Iterable[str], label: str | None = None):
        self.columns: set[str] = set(columns)
        # How the frame is referred to in diagnostics, e.g. 'df'
        self.label = label

    def get_method(self, method_name: str) -> "DFMethod | None":
        if method_name in self.func_registry:
            return DFMethod(self, self.func_registry[method_name])
        return None

    @classmethod
    def register(cls, name: str):
        def decorator(func: DFFunc):
            cls.func_registry[name] = func
            return func

        return decorator


class DFMethod:
    def __init__(self, df: DF, func: DFFunc):
        self.df = df
        self.func = func

    def __call__(
        self,
        args: list[ast.expr],
        keywords: list[ast.keyword],
        definitions: dict[str, Result] | None = None,
        frames: FrameResolver | None = None,
    ) -> DFMethodResult:
        """
        Returns a tuple with three elements:
        - The first element is the updated dataframe.
        - The second element is the returned dataframe, if any.
        - The third element is an `IllegalAccess` instance representing an error if the method call is illegal, or `None` if there is no error.
        """
        if definitions is None:
            definitions = {}
        argsv, keywordsv = parse_args(args, keywords, definitions, frames)
        updated, returned, error = self.func(self.df.columns.copy(), argsv, keywordsv)
        return DF(updated), DF(returned) if returned is not None else None, error
