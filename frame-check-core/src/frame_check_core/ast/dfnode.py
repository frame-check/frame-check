from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final


@dataclass(frozen=True, slots=True)
class Fix:
    df_node: "DFNode"
    column: str


class DFNode(ABC):
    def __init__(self, columns: frozenset[str]):
        self.columns = columns
        self.fix_cached: dict[str, list[Fix] | None] = {}

    @final
    def get_column_fixes(self, column: str) -> list[Fix] | None:
        """
        Get fixes for a specific column.
        Returns None if the column exists, or an empty list if no fixes are available.
        """
        if column in self.columns:
            return None
        if column in self.fix_cached:
            return self.fix_cached[column]
        fixes = self._generate_fixes(column)
        self.fix_cached[column] = fixes
        return fixes

    @abstractmethod
    def _generate_fixes(self, column: str) -> list[Fix] | None:
        """
        The method that will be called by get_column_fixes.
        In this method, we assume the column does not exist in self.columns.
        """
        pass


class FromData(DFNode):
    def _generate_fixes(self, column: str) -> list[Fix] | None:
        return [Fix(self, column)]


class ChosenCols(DFNode):
    def __init__(self, source: DFNode, chosen_columns: frozenset[str]):
        super().__init__(source.columns & chosen_columns)
        self.source = source
        self.chosen_columns = chosen_columns

    def _generate_fixes(self, column: str) -> list[Fix] | None:
        if column in self.chosen_columns:
            return self.source.get_column_fixes(column)
        else:
            return [Fix(self, column)]


class MapCols(DFNode):
    def __init__(self, source: DFNode, columns_map: dict[str, str]):
        super().__init__(
            frozenset(k for k, v in columns_map.items() if v in source.columns)
        )
        self.source = source
        self.columns_map = columns_map

    def _generate_fixes(self, column: str) -> list[Fix] | None:
        if column in self.columns_map:
            source_column = self.columns_map[column]
            return self.source.get_column_fixes(source_column)
        else:
            return [Fix(self, column)]


class Copy(DFNode):
    def __init__(self, source: DFNode):
        super().__init__(source.columns)
        self.source = source

    def _generate_fixes(self, column: str) -> list[Fix] | None:
        return self.source.get_column_fixes(column)


class AddCols(DFNode):
    def __init__(self, source: DFNode, added_columns: frozenset[str]):
        super().__init__(source.columns | added_columns)
        self.source = source
        self.added_columns = added_columns

    def _generate_fixes(self, column: str) -> list[Fix] | None:
        if column in self.added_columns:
            return None
        else:
            return self.source.get_column_fixes(column)


class RemoveCols(DFNode):
    def __init__(self, source: DFNode, removed_columns: frozenset[str]):
        super().__init__(source.columns - removed_columns)
        self.source = source
        self.removed_columns = removed_columns

    def _generate_fixes(self, column: str) -> list[Fix] | None:
        if column in self.removed_columns:
            return [Fix(self, column)]
        else:
            return self.source.get_column_fixes(column)


class MergeDFs(DFNode):
    def __init__(self, *dfs: DFNode):
        combined_columns = frozenset().union(*(df.columns for df in dfs))
        super().__init__(combined_columns)
        self.dfs = dfs

    def _generate_fixes(self, column: str) -> list[Fix] | None:
        fixes = []
        for df in self.dfs:
            df_fixes = df.get_column_fixes(column)
            if df_fixes:
                for fix in df_fixes:
                    fixes.append(fix)
        return fixes
