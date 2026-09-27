from ..diagnostic import IllegalAccess
from .models import DF, ColumnLambda, DFFuncResult, Result, idx_or_key


def _column_labels(value: Result) -> set[str] | None:
    """Resolve a column label argument (`"A"` or `["A", "B"]`) to known names."""
    match value:
        case str():
            return {value}
        case list():
            return {label for label in value if isinstance(label, str)}
        case _:
            return None


@DF.register("assign")
def df_assign(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # Keywords are assigned in order; a callable sees the columns created by
    # the keywords before it, so validate lambda reads against that state.
    returned = set(columns)
    missing: list[str] = []
    for name, value in keywords.items():
        if isinstance(value, ColumnLambda):
            missing.extend(
                col
                for col in value.columns
                if col not in returned and col not in missing
            )
        returned.add(name)

    error = IllegalAccess(missing=missing) if missing else None
    return columns, returned, error


@DF.register("insert")
def df_insert(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    column = idx_or_key(args, keywords, idx=1, key="column")
    if isinstance(column, str):
        columns.add(column)
    return columns, None, None


@DF.register("rename")
def df_rename(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    col_mapping = idx_or_key(args, keywords, key="columns")

    # Mapper+axis form: df.rename({"a": "b"}, axis=1) / axis="columns"
    if not isinstance(col_mapping, dict):
        axis = idx_or_key(args, keywords, idx=1, key="axis")
        if axis == 1 or axis == "columns":
            col_mapping = idx_or_key(args, keywords, idx=0, key="mapper")

    inplace = idx_or_key(args, keywords, key="inplace")

    if not isinstance(col_mapping, dict):
        # Can't determine rename statically — leave columns untouched
        if inplace is True:
            return columns, None, None
        return columns, columns, None

    new_columns = set()
    for col in columns:
        if col in col_mapping and isinstance(col_mapping[col], str):
            new_columns.add(col_mapping[col])
        else:
            new_columns.add(col)

    if inplace is True:
        return new_columns, None, None
    return columns, new_columns, None


@DF.register("drop")
def df_drop(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    labels = _column_labels(idx_or_key(args, keywords, key="columns"))

    # Labels+axis form: df.drop(["a"], axis=1) / axis="columns"
    if labels is None:
        axis = idx_or_key(args, keywords, key="axis")
        if axis == 1 or axis == "columns":
            labels = _column_labels(idx_or_key(args, keywords, idx=0, key="labels"))

    inplace = idx_or_key(args, keywords, key="inplace")

    # Unresolvable labels (or index-only drop) leave columns untouched
    new_columns = columns - labels if labels else columns

    # Missing labels raise KeyError unless errors="ignore"
    error = None
    errors = idx_or_key(args, keywords, key="errors")
    if labels and errors != "ignore" and (missing := labels - columns):
        error = IllegalAccess(missing=sorted(missing))

    if inplace is True:
        return new_columns, None, error
    return columns, new_columns, error


@DF.register("pop")
def df_pop(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # Removes the column in place and returns it as a Series
    column = idx_or_key(args, keywords, idx=0, key="item")
    if not isinstance(column, str):
        return columns, None, None
    if column not in columns:
        return columns, None, IllegalAccess(missing=[column])
    columns.remove(column)
    return columns, None, None


# Methods that return a frame with exactly the same columns (row selection,
# reordering, value transforms). Row-wise dropna(axis=1) may remove columns,
# but keeping the full set can't produce false positives.
@DF.register("abs")
@DF.register("astype")
@DF.register("bfill")
@DF.register("clip")
@DF.register("copy")
@DF.register("drop_duplicates")
@DF.register("dropna")
@DF.register("ffill")
@DF.register("fillna")
@DF.register("head")
@DF.register("nlargest")
@DF.register("nsmallest")
@DF.register("query")
@DF.register("replace")
@DF.register("round")
@DF.register("sample")
@DF.register("sort_index")
@DF.register("sort_values")
@DF.register("tail")
def df_same_columns(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    if idx_or_key(args, keywords, key="inplace") is True:
        return columns, None, None
    return columns, columns, None
