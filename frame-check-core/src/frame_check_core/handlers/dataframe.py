import re

from ..diagnostic import IllegalAccess
from .expressions import missing_reads, parse_expression
from .models import (
    DF,
    UNPACKED,
    ColumnLambda,
    DFFuncResult,
    Result,
    Unknown,
    idx_or_key,
)


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
    if UNPACKED in keywords:
        # assign(**mapping) with a mapping we can't resolve: columns unknown
        return columns, None, None

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
    inplace = idx_or_key(args, keywords, key="inplace")

    if "columns" in keywords:
        col_mapping = keywords["columns"]
    else:
        # Mapper+axis form: df.rename({"a": "b"}, axis=1) / axis="columns"
        axis = idx_or_key(args, keywords, idx=1, key="axis")
        if not (axis == 1 or axis == "columns"):
            # Only the index is renamed
            return columns, None if inplace is True else columns, None
        col_mapping = idx_or_key(args, keywords, idx=0, key="mapper")

    if not isinstance(col_mapping, dict) or Unknown in col_mapping:
        # A callable (str.upper), an unresolved mapping or an unresolved key
        # renames columns in ways we can't know: the result is unknown
        if inplace is True:
            return None, None, None
        return columns, None, None

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


def _expression(args: list[Result], keywords: dict[str, Result]) -> Result:
    return idx_or_key(args, keywords, idx=0, key="expr")


def _expression_scope_is_plain(keywords: dict[str, Result]) -> bool:
    """Whether names in the expression resolve to the frame's columns only.

    `resolvers=` adds extra names and `target=` redirects assignments (bare
    names never see local or global variables, so `local_dict`/`global_dict`
    don't matter).
    """
    return (
        UNPACKED not in keywords
        and "resolvers" not in keywords
        and "target" not in keywords
        and keywords.get("parser", "pandas") == "pandas"
    )


@DF.register("query")
def df_query(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # Keeps the columns; names in the expression must be columns or index levels
    # (`@var` refers to a local variable). Anything we can't analyze is ignored.
    expr = _expression(args, keywords)
    error = None
    if isinstance(expr, str) and _expression_scope_is_plain(keywords):
        lines = parse_expression(expr, multiline=False)
        if (
            lines is not None
            and lines[0].target is None
            and (missing := missing_reads(lines[0], columns))
        ):
            error = IllegalAccess(missing=missing)
    updated, returned, _ = _same_columns(columns, args, keywords)
    return updated, returned, error


@DF.register("eval")
def df_eval(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # Assignments ("C = A * 2") add columns, to a copy or in place; a plain
    # expression evaluates to a Series (or scalar) instead of a frame.
    inplace = Unknown if UNPACKED in keywords else keywords.get("inplace", False)
    # What an assignment we can't resolve does to the original frame
    unknown = None if inplace is True or inplace is Unknown else columns

    expr = _expression(args, keywords)
    lines = None
    if isinstance(expr, str) and _expression_scope_is_plain(keywords):
        lines = parse_expression(expr, multiline=True)
    if lines is None:
        return unknown, None, None

    assigns = [line.target is not None for line in lines]
    if len(lines) > 1 and not all(assigns):
        return columns, None, None  # pandas raises; nothing to track

    new_columns = set(columns)
    missing: list[str] = []
    for line in lines:
        missing.extend(
            name for name in missing_reads(line, new_columns) if name not in missing
        )
        if line.target_quoted:
            # pandas stores backtick-quoted targets under a mangled name
            return unknown, None, None
        if line.target is not None:
            new_columns.add(line.target)
    error = IllegalAccess(missing=missing) if missing else None

    if not assigns[0]:
        return columns, None, error  # a Series, not a frame
    if inplace is True:
        return new_columns, None, error
    if inplace is Unknown:
        return None, None, error
    return columns, new_columns, error


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
    if not isinstance(right, DF) or UNPACKED in keywords:
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


_JOIN_PARAMS = ("on", "how", "lsuffix", "rsuffix", "sort", "validate")


@DF.register("join")
def df_join(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    other = idx_or_key(args, keywords, idx=0, key="other")
    params = bind_positional(args[1:], keywords, _JOIN_PARAMS)
    if UNPACKED in params:
        return columns, None, None

    # `on` names a column of the calling frame; unknown labels raise KeyError
    error = None
    if params.get("on") is not None:
        on = _column_labels(params["on"])
        if on is None:
            return columns, None, None
        if missing := on - columns:
            error = IllegalAccess(missing=sorted(missing))

    match other:
        case DF():
            others = [other.columns]
        case list() if other and all(isinstance(o, DF) for o in other):
            others = [o.columns for o in other if isinstance(o, DF)]
        case _:
            # Series or unresolved frames: columns unknown
            return columns, None, error

    lsuffix = params.get("lsuffix", "")
    rsuffix = params.get("rsuffix", "")
    if not (isinstance(lsuffix, str) and isinstance(rsuffix, str)):
        return columns, None, error

    if len(others) > 1:
        # Joining a list of frames: any overlapping columns raise ValueError
        seen = set(columns)
        for other_columns in others:
            if seen & other_columns:
                return columns, None, error
            seen |= other_columns
        return columns, seen, error

    right = others[0]
    overlap = columns & right
    if overlap and not (lsuffix or rsuffix):
        # "columns overlap but no suffix specified" (ValueError)
        return columns, None, error
    joined = (columns - overlap) | (right - overlap)
    for column in overlap:
        joined.add(column + lsuffix)
        joined.add(column + rsuffix)
    return columns, joined, error


_RESET_INDEX_PARAMS = (
    "level",
    "drop",
    "inplace",
    "col_level",
    "col_fill",
    "allow_duplicates",
    "names",
)


@DF.register("reset_index")
def df_reset_index(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    params = bind_positional(args, keywords, _RESET_INDEX_PARAMS)
    inplace = params.get("inplace") is True

    if params.get("drop") is True:
        new_columns = columns
    elif params.get("level") is not None:
        # Moves some index levels, whose names aren't tracked
        return (None, None, None) if inplace else (columns, None, None)
    else:
        names = params.get("names")
        if names is None:
            # The index isn't tracked; a default index becomes "index" (or
            # "level_0" when taken). A named index's column is typically
            # still in the tracked columns, since set_index isn't tracked.
            new_columns = columns | {"level_0" if "index" in columns else "index"}
        elif isinstance(names, str):
            new_columns = columns | {names}
        else:
            return (None, None, None) if inplace else (columns, None, None)

    if inplace:
        return new_columns, None, None
    return columns, new_columns, None


@DF.register("reindex")
def df_reindex(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # The labels define the columns exactly; unknown ones become NaN columns
    if "columns" in keywords:
        labels = keywords["columns"]
    else:
        axis = keywords.get("axis")
        if not (axis == 1 or axis == "columns"):
            # Only the index is reindexed
            return columns, columns, None
        labels = idx_or_key(args, keywords, idx=0, key="labels")

    names = _column_labels(labels)
    if names is None or (isinstance(labels, list) and len(names) != len(labels)):
        return columns, None, None
    return columns, names, None


@DF.register("filter")
def df_filter(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    params = bind_positional(args, keywords, ("items", "like", "regex", "axis"))
    axis = params.get("axis")
    if axis in (0, "index", "rows"):
        # Filters row labels
        return columns, columns, None
    if axis not in (None, 1, "columns"):
        return columns, None, None

    # Unknown labels in items= are skipped silently; like/regex are computed
    # on the known columns (pandas uses `in` and re.search)
    if (items := params.get("items")) is not None:
        names = _column_labels(items)
        return columns, (columns & names if names is not None else None), None
    if isinstance(like := params.get("like"), str):
        return columns, {c for c in columns if like in c}, None
    if isinstance(regex := params.get("regex"), str):
        try:
            pattern = re.compile(regex)
        except re.error:
            return columns, None, None
        return columns, {c for c in columns if pattern.search(c)}, None
    return columns, None, None


@DF.register("groupby")
def df_groupby(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # Returns a GroupBy, not a frame. A string key must be a column
    # (KeyError). A list isn't checked: one as long as the frame is used as
    # group labels, and the row count isn't known. Series, functions and
    # level= aren't checked either.
    by = idx_or_key(args, keywords, idx=0, key="by")
    error = None
    if isinstance(by, str) and keywords.get("level") is None and by not in columns:
        error = IllegalAccess(missing=[by])
    return columns, None, error


# Positional parameters of DataFrame.melt / pd.melt (after the frame)
MELT_PARAMS = (
    "id_vars",
    "value_vars",
    "var_name",
    "value_name",
    "col_level",
    "ignore_index",
)


def _label_list(value: Result) -> tuple[list[str], bool]:
    """
    Resolve a label-list argument (`None`, `"A"` or `["A", "B"]`).

    Returns the string labels and whether every label was resolved. A label
    that isn't a string (an int, a variable we can't resolve, a tuple key)
    makes the list incomplete.
    """
    match value:
        case None:
            return [], True
        case str():
            return [value], True
        case list():
            labels = [v for v in value if isinstance(v, str)]
            return labels, len(labels) == len(value)
        case _:
            return [], False


def melt_columns(
    columns: set[str],
    label: str | None,
    keywords: dict[str, Result],
) -> tuple[set[str] | None, IllegalAccess | None]:
    """
    Compute the columns of `df.melt(...)` (and `pd.melt`).

    The result is the `id_vars`, then the variable and value columns
    (`var_name` / `value_name`, default "variable" / "value"). Labels of
    `id_vars` and `value_vars` missing from the frame raise KeyError.

    Args:
        columns: Columns of the melted frame.
        label: Label of the frame for diagnostics, or None when it is the
            frame the method is called on.
        keywords: The arguments by name (see `bind_positional`).

    Returns:
        The melted columns (None when they can't be determined statically)
        and an error for missing labels, if any.
    """
    if UNPACKED in keywords:
        return None, None
    if keywords.get("col_level") is not None:
        # Melts one level of MultiIndex columns
        return None, None

    ids, ids_known = _label_list(keywords.get("id_vars"))
    values, _ = _label_list(keywords.get("value_vars"))

    missing = list(dict.fromkeys(c for c in (*ids, *values) if c not in columns))
    error = None
    if missing:
        error = IllegalAccess(
            missing=missing,
            frame=label,
            available=sorted(columns) if label is not None else None,
        )

    # The default variable name is the name of the columns axis ("variable"
    # when unnamed). The checker drops the result when the axis may be named
    # or a MultiIndex (see Checker._columns_may_be_named).
    var_name = keywords.get("var_name", "variable")
    if var_name is None:
        var_name = "variable"
    value_name = keywords.get("value_name", "value")
    if not (ids_known and isinstance(var_name, str) and isinstance(value_name, str)):
        # A label list or a None value_name (NaN column) isn't a plain label
        return None, error
    return {*ids, var_name, value_name}, error


@DF.register("melt")
def df_melt(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    params = bind_positional(args, keywords, MELT_PARAMS)
    melted, error = melt_columns(columns, None, params)
    return columns, melted, error


def reshape_error(
    columns: set[str], label: str | None, labels: list[str]
) -> IllegalAccess | None:
    """Report the `labels` missing from `columns` (a KeyError at runtime)."""
    missing = list(dict.fromkeys(c for c in labels if c not in columns))
    if not missing:
        return None
    return IllegalAccess(
        missing=missing,
        frame=label,
        available=sorted(columns) if label is not None else None,
    )


def pivot_error(
    columns: set[str], label: str | None, keywords: dict[str, Result]
) -> IllegalAccess | None:
    """
    Validate the labels of `df.pivot(index=, columns=, values=)`.

    A string or a list of strings in any of them must be a column: unlike
    pivot_table, an index level name isn't accepted.
    """
    if UNPACKED in keywords:
        return None
    labels = [
        name
        for key in ("index", "columns", "values")
        for name in _label_list(keywords.get(key))[0]
    ]
    return reshape_error(columns, label, labels)


# Positional parameters of DataFrame.pivot_table / pd.pivot_table (after the frame)
PIVOT_TABLE_PARAMS = ("values", "index", "columns")


def pivot_table_error(
    columns: set[str], label: str | None, keywords: dict[str, Result]
) -> IllegalAccess | None:
    """
    Validate the labels of `df.pivot_table(values=, index=, columns=)`.

    `values` and `index` take a string or a list of strings that must be
    columns. A string `columns` must be one too, but a list is not checked:
    pandas doesn't raise for an unknown label in a list of several. Callers
    must also drop the error when the frame may have a named index, since
    `index` and `columns` accept index level names.
    """
    if UNPACKED in keywords:
        return None
    labels = [
        *_label_list(keywords.get("index"))[0],
        *(c for c in [keywords.get("columns")] if isinstance(c, str)),
        *_label_list(keywords.get("values"))[0],
    ]
    return reshape_error(columns, label, labels)


# The columns of these results come from the data (the values of the pivoted
# columns), so the result is unknown. Only the labels are validated.


@DF.register("pivot")
def df_pivot(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    return columns, None, pivot_error(columns, None, keywords)


@DF.register("pivot_table")
def df_pivot_table(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    params = bind_positional(args, keywords, PIVOT_TABLE_PARAMS)
    return columns, None, pivot_table_error(columns, None, params)


# The result is a Series, or a frame whose columns are the values of an index
# level (or the old index), so it is unknown.
@DF.register("stack")
@DF.register("unstack")
@DF.register("transpose")
def df_unknown_columns(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    return columns, None, None
