import ast
from dataclasses import dataclass
from functools import cached_property
from typing import Self


@dataclass(kw_only=True, order=True, frozen=True)
class CodePosition:
    """Represents a point in source code."""

    row: int = 0
    col: int = 0

    def as_lsp_position(self) -> tuple[int, int]:
        """Convert to zero-indexed tuple (Usually for LSP)"""

        return (self.row - 1, self.col)

    def __str__(self) -> str:
        return f"{self.row}:{self.col}"


@dataclass(kw_only=True, order=True, frozen=True)
class CodeRegion:
    """
    Represents a rectangular region (end exclusive) in source code,
    bounded by a start (top left) and end (bottom right) position.
    """

    start: CodePosition
    end: CodePosition

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError("End position must not be before start position.")

    @cached_property
    def row_span(self) -> int:
        """Get the number of rows spanned by the region"""

        return self.end.row - self.start.row

    @cached_property
    def col_span(self) -> int:
        """Get the number of columns spanned by the region"""

        return self.end.col - self.start.col

    @cached_property
    def is_same_row(self) -> bool:
        """Check if the region is within the same row"""

        return self.row_span == 1

    @cached_property
    def is_same_column(self) -> bool:
        """Check if the region is within the same column"""

        return self.col_span == 1

    @cached_property
    def is_empty(self) -> bool:
        """Check if the region is empty"""

        return self.col_span == 0 and self.row_span == 0

    @classmethod
    def from_ast_node(cls, *, node: ast.stmt | ast.expr) -> Self:
        """Construct a CodeRegion from an AST node."""

        start_position = CodePosition(row=node.lineno, col=node.col_offset)

        # NOTE: lineno and end_lineno are not exclusive, so we add 1
        exclusive_end_row = (node.end_lineno or node.lineno) + 1
        exclusive_end_col_offset = node.end_col_offset or node.col_offset
        end_position = CodePosition(
            row=exclusive_end_row,
            col=exclusive_end_col_offset,
        )
        return cls(start=start_position, end=end_position)

    @classmethod
    def from_tuples(cls, *, start: tuple[int, int], end: tuple[int, int]) -> Self:
        """Shorthand to create a CodeRegion from start and end tuples."""

        start_position = CodePosition(row=start[0], col=start[1])
        end_position = CodePosition(row=end[0], col=end[1])
        return cls(start=start_position, end=end_position)
