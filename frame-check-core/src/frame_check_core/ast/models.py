import ast
from typing import Callable, Iterable, Union

from ..models.diagnostic import IllegalAccess


class _Unknown:
    pass


Unknown = _Unknown()  # A value that is either not supported or not provided.
Result = Union[str, dict, list, "PD", "PDMethod", "DF", "DFMethod", _Unknown]

_ASSIGNING_ATTR = "_frame_checker_assigning"
_RESULT_ATTR = "_frame_checker_result_columns"


def is_assigning(node: ast.Subscript) -> bool:
    return getattr(node, _ASSIGNING_ATTR, False)


def set_assigning(node: ast.Subscript) -> None:
    setattr(node, _ASSIGNING_ATTR, True)


def get_value(node: ast.AST) -> Result:
    # Fast, shallow extraction to avoid deep recursion:
    # - Strings are returned directly
    # - Lists of string constants are returned as a list[str]
    # - Dicts with string keys and shallow-parsable values are returned
    #   - Values are either string constants or lists of string constants
    # - Everything else is Unknown
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.List):
        values: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                values.append(elt.value)
            else:
                return Unknown
        return values

    if isinstance(node, ast.Dict):
        result_dict: dict[str, Result] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                continue
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
            else:
                return Unknown

            # Shallow parse values without recursion
            if isinstance(value_node, ast.Constant) and isinstance(
                value_node.value, str
            ):
                value: Result = value_node.value
            elif isinstance(value_node, ast.List):
                vals: list[str] | None = []
                for elt in value_node.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        vals.append(elt.value)
                    else:
                        vals = None
                        break
                value = vals if vals is not None else Unknown
            else:
                value = Unknown

            result_dict[key] = value
        return result_dict

    return Unknown


def get_result(node: ast.AST) -> Result:
    if hasattr(node, _RESULT_ATTR):
        return getattr(node, _RESULT_ATTR)
    else:
        return get_value(node)


def set_result(node: ast.AST, result: Result) -> None:
    setattr(node, _RESULT_ATTR, result)


def parse_args(
    args: list[ast.expr],
    keywords: list[ast.keyword],
    arg_indices: set[int] | None = None,
    keyword_names: set[str] | None = None,
) -> tuple[list[Result], dict[str, Result]]:
    # Targeted parsing to optionally limit work:
    # - If arg_indices is provided, only parse those positional args; others become Unknown
    # - If keyword_names is provided, only parse those keywords; others are skipped
    if arg_indices is None:
        argsv = [get_result(arg) for arg in args]
    else:
        argsv = [
            get_result(arg) if i in arg_indices else Unknown
            for i, arg in enumerate(args)
        ]

    if keyword_names is None:
        keywordsv = {
            kw.arg: get_result(kw.value) for kw in keywords if kw.arg is not None
        }
    else:
        keywordsv = {
            kw.arg: get_result(kw.value)
            for kw in keywords
            if kw.arg is not None and kw.arg in keyword_names
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
    def method(cls, name: str):
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
        arg_indices: set[int] | None = None,
        keyword_names: set[str] | None = None,
    ) -> PDMethodResult:
        """
        Returns a tuple with two elements:
        - The first element is the created dataframe, if any.
        - The second element is an `IllegalAccess` instance representing an error if the method call is illegal, or `None` if there is no error.
        """
        argsv, keywordsv = parse_args(
            args, keywords, arg_indices=arg_indices, keyword_names=keyword_names
        )
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
    def method(cls, name: str):
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
        arg_indices: set[int] | None = None,
        keyword_names: set[str] | None = None,
    ) -> DFMethodResult:
        """
        Returns a tuple with three elements:
        - The first element is the updated dataframe.
        - The second element is the returned dataframe, if any.
        - The third element is an `IllegalAccess` instance representing an error if the method call is illegal, or `None` if there is no error.
        """
        argsv, keywordsv = parse_args(
            args, keywords, arg_indices=arg_indices, keyword_names=keyword_names
        )
        updated, returned, error = self.func(self.df.columns.copy(), argsv, keywordsv)
        return DF(updated), DF(returned) if returned is not None else None, error
