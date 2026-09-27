from ..diagnostic import IllegalAccess
from .models import DF, ColumnLambda, DFFuncResult, Result, Unknown, idx_or_key


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
# reordering, value transforms).
@DF.register("abs")
@DF.register("bfill")
@DF.register("clip")
@DF.register("copy")
@DF.register("ffill")
@DF.register("fillna")
@DF.register("head")
@DF.register("query")
@DF.register("replace")
@DF.register("round")
@DF.register("sample")
@DF.register("sort_index")
@DF.register("tail")
def df_same_columns(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    return _same_columns(columns, args, keywords)


def _same_columns(
    columns: set[str],
    args: list[Result],
    keywords: dict[str, Result],
    required: Result = Unknown,
) -> DFFuncResult:
    """Keep all columns; report `required` labels that don't exist (KeyError)."""
    error = None
    labels = _column_labels(required)
    if labels and (missing := labels - columns):
        error = IllegalAccess(missing=sorted(missing))

    if idx_or_key(args, keywords, key="inplace") is True:
        return columns, None, error
    return columns, columns, error


# The column-preserving methods below raise KeyError for unknown labels in
# their column arguments (unlike e.g. fillna/replace/round with a dict).


@DF.register("sort_values")
def df_sort_values(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    axis = idx_or_key(args, keywords, key="axis")
    if axis == 1 or axis == "columns":
        # Sorting columns by row labels
        return _same_columns(columns, args, keywords)
    by = idx_or_key(args, keywords, idx=0, key="by")
    return _same_columns(columns, args, keywords, required=by)


@DF.register("drop_duplicates")
def df_drop_duplicates(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    subset = idx_or_key(args, keywords, idx=0, key="subset")
    return _same_columns(columns, args, keywords, required=subset)


@DF.register("dropna")
def df_dropna(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # Rows are dropped by default; dropna(axis=1) may remove columns, but
    # keeping the full set can't produce false positives.
    subset = idx_or_key(args, keywords, key="subset")
    return _same_columns(columns, args, keywords, required=subset)


@DF.register("nlargest")
@DF.register("nsmallest")
def df_nlargest_nsmallest(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    required = idx_or_key(args, keywords, idx=1, key="columns")
    return _same_columns(columns, args, keywords, required=required)


@DF.register("astype")
def df_astype(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    dtype = idx_or_key(args, keywords, idx=0, key="dtype")
    if not isinstance(dtype, dict):
        return _same_columns(columns, args, keywords)
    required = [k for k in dtype if isinstance(k, str)]
    return _same_columns(columns, args, keywords, required=required)


_DEFAULT_SUFFIXES = ("_x", "_y")

# Positional parameters after `right` in DataFrame.merge / pd.merge
MERGE_PARAMS = (
    "how",
    "on",
    "left_on",
    "right_on",
    "left_index",
    "right_index",
    "sort",
    "suffixes",
    "copy",
    "indicator",
)


def bind_positional(
    args: list[Result], keywords: dict[str, Result], names: tuple[str, ...]
) -> dict[str, Result]:
    """Merge positional `args` into `keywords` using the parameter `names`."""
    bound = dict(zip(names, args, strict=False))
    bound.update(keywords)
    return bound


def merge_columns(
    left: set[str],
    left_label: str | None,
    right: Result,
    keywords: dict[str, Result],
) -> tuple[set[str] | None, IllegalAccess | None]:
    """
    Compute the columns of `left.merge(right, ...)` (and `pd.merge`).

    Join keys appear once; other columns present on both sides get the
    `suffixes` (default `_x`/`_y`, `None` = unchanged). Keys missing from
    either side raise KeyError, so they are returned as an error.

    Args:
        left: Columns of the left frame.
        left_label: Label of the left frame for diagnostics, or None when it
            is the frame the method is called on.
        right: The right frame argument.
        keywords: The remaining arguments by name (see `bind_positional`).

    Returns:
        The merged columns (None when they can't be determined statically)
        and an error for missing keys, if any.
    """
    if not isinstance(right, DF):
        return None, None
    right_columns = right.columns

    left_index = keywords.get("left_index") is True
    right_index = keywords.get("right_index") is True
    missing_left: set[str] = set()
    missing_right: set[str] = set()

    if left_index != right_index:
        # One-sided index merges have irregular output columns
        return None, None
    if keywords.get("how") == "cross" or left_index:
        shared: set[str] = set()
    elif keywords.get("on") is not None:
        on = _column_labels(keywords["on"])
        if on is None:
            return None, None
        missing_left, missing_right = on - left, on - right_columns
        shared = on
    elif keywords.get("left_on") is not None or keywords.get("right_on") is not None:
        left_on = _column_labels(keywords.get("left_on") or [])
        right_on = _column_labels(keywords.get("right_on") or [])
        if left_on is None or right_on is None:
            return None, None
        missing_left, missing_right = left_on - left, right_on - right_columns
        shared = left_on & right_on
    else:
        # Default: join on the columns both frames have
        shared = left & right_columns
        if not shared:
            return None, None

    error = None
    if missing_left:
        error = IllegalAccess(
            missing=sorted(missing_left),
            frame=left_label,
            available=sorted(left) if left_label is not None else None,
        )
    elif missing_right:
        error = IllegalAccess(
            missing=sorted(missing_right),
            frame=right.label,
            available=sorted(right_columns),
        )

    suffixes = keywords.get("suffixes", list(_DEFAULT_SUFFIXES))
    if not (
        isinstance(suffixes, list)
        and len(suffixes) == 2
        and all(s is None or isinstance(s, str) for s in suffixes)
    ):
        return None, error
    left_suffix, right_suffix = suffixes

    overlap = (left & right_columns) - shared
    merged = (left - overlap) | (right_columns - overlap)
    for column in overlap:
        merged.add(column + left_suffix if left_suffix else column)
        merged.add(column + right_suffix if right_suffix else column)

    indicator = keywords.get("indicator")
    if indicator is True:
        merged.add("_merge")
    elif isinstance(indicator, str):
        merged.add(indicator)
    return merged, error


@DF.register("merge")
def df_merge(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    right = idx_or_key(args, keywords, idx=0, key="right")
    params = bind_positional(args[1:], keywords, MERGE_PARAMS)
    merged, error = merge_columns(columns, None, right, params)
    return columns, merged, error
