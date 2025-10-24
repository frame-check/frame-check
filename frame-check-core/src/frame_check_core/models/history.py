import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple
from bisect import bisect_left


def get_column_values(
    col: str | ast.expr | Iterable[str],
) -> Iterable[str]:
    match col:
        case str():
            yield col
        case ast.Constant():
            match col.value:
                case str(value):
                    yield value
        case ast.List():
            for elt_node in col.elts:
                match elt_node:
                    case ast.Constant():
                        yield from get_column_values(elt_node)
                    case ast.List():
                        yield from get_column_values(elt_node)
        case Iterable():
            yield from col
        case _:
            yield from []


@dataclass(kw_only=True, frozen=True, slots=True)
class FrameInstance:
    """
    Represents an immutable instance of a data frame at a specific line number.

    This class stores information about a data frame, including its source location,
    identifier, data arguments, and columns.
    """

    lineno: int
    """
    Line number where this frame instance appears
    """

    defined_lino: int
    """
    Line number where this frame instance is first defined
    """

    id: str
    """
    Identifier for the frame
    """

    data_arg: ast.List | ast.Dict | None = None
    """
    Data argument used in the frame
    """

    keywords: list[ast.keyword] = field(default_factory=list)
    """
    Keyword arguments for the frame
    """

    columns: frozenset[str]
    """
    Set of column names in this frame
    """

    @classmethod
    def new(
        cls,
        *,
        lineno: int,
        id: str,
        data_arg: ast.List | ast.Dict | None = None,
        keywords: list[ast.keyword] = field(default_factory=list),
        columns: Iterable[str] | ast.expr,
    ) -> "FrameInstance":
        """
        Create a new FrameInstance with the given parameters.

        Args:
            lineno: Line number where the frame appears
            id: Identifier for the frame
            data_arg: Data argument used in the frame
            keywords: Keyword arguments for the frame
            columns: Column names to include in the frame

        Returns:
            A new FrameInstance with the specified properties
        """
        return cls(
            lineno=lineno,
            id=id,
            data_arg=data_arg,
            keywords=keywords,
            columns=frozenset(get_column_values(columns)),
            defined_lino=lineno,
        )

    def new_instance(
        self,
        *,
        lineno: int,
        new_columns: Iterable[str] | ast.expr,
    ) -> "FrameInstance":
        """
        Create a new FrameInstance based on the current instance with updated properties.

        This method creates a new frame instance that inherits properties from the current
        instance but with a new line number and additional columns.

        Args:
            lineno: New line number for the frame instance
            new_columns: Additional columns to merge with existing columns

        Returns:
            A new FrameInstance with updated properties
        """
        return FrameInstance(
            lineno=lineno,
            id=self.id,
            columns=self.columns.union(get_column_values(new_columns)),
            defined_lino=self.defined_lino,
        )


@dataclass(kw_only=True)
class ColumnInstance:
    _node: ast.Subscript
    lineno: int
    id: str
    frame: FrameInstance
    start_col: int
    underline_length: int


class LineIdKey(NamedTuple):
    lineno: int
    id: str


@dataclass(kw_only=True, slots=True)
class InstanceHistory[I: FrameInstance | ColumnInstance]:
    instances: dict[LineIdKey, I] = field(default_factory=dict)
    _by_id_lines: dict[str, list[int]] = field(default_factory=dict)
    _ids: set[str] = field(default_factory=set)

    def add(self, instance: I):
        key = LineIdKey(instance.lineno, instance.id)
        self.instances[key] = instance
        # index by id and lineno
        self._ids.add(key.id)
        lines = self._by_id_lines.setdefault(key.id, [])
        i = bisect_left(lines, key.lineno)
        if i == len(lines) or lines[i] != key.lineno:
            lines.insert(i, key.lineno)

    def get(self, id: str | None) -> list[I]:
        if id is None:
            return []
        lines = self._by_id_lines.get(id, [])
        return [self.instances[LineIdKey(line, id)] for line in lines]

    def get_at(self, lineno: int, id: str) -> I | None:
        return self.instances.get(LineIdKey(lineno, id))

    def get_before(self, lineno: int, id: str) -> I | None:
        lines = self._by_id_lines.get(id)
        if not lines:
            return None
        # find rightmost line < lineno
        i = bisect_left(lines, lineno) - 1
        if i >= 0:
            key = LineIdKey(lines[i], id)
            return self.instances.get(key)
        return None

    def instance_ids(self) -> set[str]:
        return set(self._ids)

    def values(self) -> list[I]:
        return list(self.instances.values())

    def __len__(self) -> int:
        return len(self.instances)

    def __setitem__(self, key: LineIdKey, value: I):
        self.instances[key] = value
        self._ids.add(key.id)
        lines = self._by_id_lines.setdefault(key.id, [])
        i = bisect_left(lines, key.lineno)
        if i == len(lines) or lines[i] != key.lineno:
            lines.insert(i, key.lineno)

    def __getitem__(self, key: LineIdKey) -> I:
        return self.instances[key]

    def __contains__(self, item: LineIdKey) -> bool:
        return item in self.instances

    def contains_id(self, id: str) -> bool:
        return id in self._ids


FrameHistory = InstanceHistory[FrameInstance]
ColumnHistory = InstanceHistory[ColumnInstance]
