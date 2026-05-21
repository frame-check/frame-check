import ast
from typing import Callable, Iterable, Union

from ..diagnostic import IllegalAccess


class _Unknown:
    pass


Unknown = _Unknown()  # A value that is either not supported or not provided.
Result = Union[str, bool, int, dict, list, "PD", "PDMethod", "DF", "DFMethod", _Unknown]

_ASSIGNING_ATTR = "_frame_checker_assigning"
_RESULT_ATTR = "_frame_checker_result_columns"


def is_assigning(node: ast.Subscript) -> bool:
    return getattr(node, _ASSIGNING_ATTR, False)


def set_assigning(node: ast.Subscript) -> None:
    setattr(node, _ASSIGNING_ATTR, True)


def get_value(node: ast.AST, definitions: dict[str, Result]) -> Result:
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
                parsed_elt = get_value(elt, definitions)
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
                key = get_result(key_node, definitions)
                value = get_result(value_node, definitions)
                result_dict[key] = value
            return result_dict

        case _:
            return Unknown


def get_result(node: ast.AST, definitions: dict[str, Result]) -> Result:
    if hasattr(node, _RESULT_ATTR):
        return getattr(node, _RESULT_ATTR)
    else:
        return get_value(node, definitions)


def set_result(node: ast.AST, result: Result) -> None:
    setattr(node, _RESULT_ATTR, result)


def parse_args(
    args: list[ast.expr],
    keywords: list[ast.keyword],
    definitions: dict[str, Result],
) -> tuple[list[Result], dict[str, Result]]:
    argsv = [get_result(arg, definitions) for arg in args]
    keywordsv = {
        kw.arg: get_result(kw.value, definitions)
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
    instance = None
    func_registry: dict[str, PDFunc] = {}

    def __new__(cls) -> "PD":
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

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
    ) -> PDMethodResult:
        """
        Returns a tuple with two elements:
        - The first element is the created dataframe, if any.
        - The second element is an `IllegalAccess` instance representing an error if the method call is illegal, or `None` if there is no error.
        """
        # If definitions is not provided, fall back to empty dict (for backward compatibility)
        if definitions is None:
            definitions = {}
        argsv, keywordsv = parse_args(args, keywords, definitions)
        returned, error = self.func(argsv, keywordsv)
        return DF(returned) if returned is not None else None, error


DFFuncResult = tuple[set[str], set[str] | None, IllegalAccess | None]
DFFunc = Callable[[set[str], list[Result], dict[str, Result]], DFFuncResult]
DFMethodResult = tuple["DF", "DF | None", IllegalAccess | None]


class DF:
    """This represents a state of a DataFrame. It should be considered immutable."""

    func_registry: dict[str, DFFunc] = {}

    def __init__(self, columns: Iterable[str]):
        self.columns: set[str] = set(columns)

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
    ) -> DFMethodResult:
        """
        Returns a tuple with three elements:
        - The first element is the updated dataframe.
        - The second element is the returned dataframe, if any.
        - The third element is an `IllegalAccess` instance representing an error if the method call is illegal, or `None` if there is no error.
        """
        if definitions is None:
            definitions = {}
        argsv, keywordsv = parse_args(args, keywords, definitions)
        updated, returned, error = self.func(self.df.columns.copy(), argsv, keywordsv)
        return DF(updated), DF(returned) if returned is not None else None, error
