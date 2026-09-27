from typing import Literal, overload

Strict = Literal["strict"]
Relaxed = Literal["relaxed"]


class Tracker[M: Strict | Relaxed]:
    """
    Tracks column dependencies in a DataFrame-like structure.

    In 'strict' mode: All column dependencies must exist, returns Diagnostic on errors.
    In 'relaxed' mode: Missing dependencies are auto-created, never returns errors.
    """

    __slots__ = ("_is_strict", "columns", "id_", "mode")

    def __init__(self, id_: str, mode: M = "strict") -> None:  # type: ignore[assignment]
        self.id_ = id_
        self.columns: dict[str, set[str]] = {}
        self.mode = mode
        self._is_strict = mode == "strict"

    @overload
    def try_get(
        self: "Tracker[Relaxed]",
        column: str,
    ) -> None: ...

    @overload
    def try_get(
        self: "Tracker[Strict]",
        column: str,
    ) -> str | None: ...

    def try_get(self, column: str) -> str | None:
        """
        Access a column (read operation).

        Args:
            column: The column name to access

        Returns:
            None on success (or in relaxed mode)
            Name of the column if it doesn't exist in strict mode
        """
        columns = self.columns

        if column in columns:
            return None

        if self._is_strict:
            return column

        # Relaxed mode: auto-create the column
        columns[column] = set()
        return None

    @overload
    def try_add(
        self: "Tracker[Relaxed]",
        column: str,
        *,
        depends_on: str | list[str] | None = None,
    ) -> None: ...

    @overload
    def try_add(
        self: "Tracker[Strict]",
        column: str,
        *,
        depends_on: None = None,
    ) -> None: ...

    @overload
    def try_add(
        self: "Tracker[Strict]",
        column: str,
        *,
        depends_on: str | list[str],
    ) -> list[str] | None: ...

    def try_add(
        self, column: str, *, depends_on: str | list[str] | None = None
    ) -> list[str] | None:
        """
        Add a column to the tracker (write operation), optionally with dependencies.

        In 'strict' mode:
            - If any dependency does not exist, returns a Diagnostic describing the missing dependency.
            - Otherwise, adds the column and its dependencies.

        In 'relaxed' mode:
            - Any missing dependencies are automatically created as independent columns.
            - Always returns None.

        Args:
            column: The column name to add or write.
            depends_on: Optional dependency column name(s). Can be a single string, a list of strings, or None.

        Returns:
            None on success (or always in relaxed mode).
            Name of the columns that are required to exist in strict mode
        """
        columns = self.columns  # Local reference for faster lookups

        # Case 1: Adding a column without dependencies (e.g., df['X'] = 1)
        if depends_on is None:
            if column not in columns:
                columns[column] = set()
            return None

        # Normalize depends_on to a list for uniform processing
        dependencies = [depends_on] if isinstance(depends_on, str) else depends_on

        # Case 2: Adding a column with dependencies (e.g., df['F'] = df['A'] + df['B'])
        # In strict mode, collect all missing dependencies first
        if self._is_strict:
            missing = [dep for dep in dependencies if dep not in columns]
            if missing:
                return missing
            # All dependencies exist, add the column
            if column not in columns:
                columns[column] = set()
            columns[column].update(dependencies)
            return None

        # Relaxed mode: auto-create any missing dependencies
        for dep in dependencies:
            if dep not in columns:
                columns[dep] = set()

        # All dependencies are valid (or were created), add them
        if column not in columns:
            columns[column] = set()
        columns[column].update(dependencies)
        return None

    @classmethod
    def new_with_columns(cls, id_: str, columns: list[str]) -> "Tracker[Strict]":
        """
        Create a new strict-mode tracker with a predefined schema.

        Args:
            id_: Tracker identifier
            columns: List of column names to initialize

        Returns:
            A new FrameTracker in strict mode with columns initialized
        """
        tracker = Tracker(id_, mode="strict")
        for column in columns:
            tracker.columns[column] = set()
        return tracker

    def get_core(self) -> list[str]:
        """
        Get all independent columns (columns with no dependencies).

        Returns:
            List of column names that don't depend on other columns
        """
        columns = self.columns
        return [col for col in columns if not columns[col]]
