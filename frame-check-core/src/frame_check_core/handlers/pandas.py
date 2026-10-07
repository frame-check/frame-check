from .dataframe import MERGE_PARAMS, bind_positional, merge_columns
from .models import DF, PD, UNPACKED, PDFuncResult, Result, idx_or_key


@PD.register("DataFrame")
def pd_dataframe(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    data = idx_or_key(args, keywords, idx=0, key="data")
    match data:
        case DF():
            # Copy constructor: pd.DataFrame(other_df)
            return set(data.columns), None
        case dict():
            return {k for k in data if isinstance(k, str)}, None
        case list():
            columns: set[str] = set()
            for item in data:
                if not isinstance(item, dict):
                    return None, None
                columns |= {k for k in item if isinstance(k, str)}
            return columns, None
        case _:
            return None, None


# Note: both read_csv and read_excel have the same usecols handling
# hence we can re-use the same function
@PD.register("read_csv")
@PD.register("read_excel")
def pd_read_csv_and_excel(
    args: list[Result], keywords: dict[str, Result]
) -> PDFuncResult:
    usecols = idx_or_key(args, keywords, key="usecols")
    match usecols:
        case str():
            return {usecols}, None
        case list():
            # Flatten and resolve variables (if a variable is a list, expand it)
            cols = set()
            for col in usecols:
                if isinstance(col, str):
                    cols.add(col)
                elif isinstance(col, list):
                    cols.update(x for x in col if isinstance(x, str))
            return cols, None
        case _:
            return None, None


@PD.register("read_json")
def pd_read_json(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    """Handle pd.read_json() - cannot determine columns statically."""
    # read_json doesn't have usecols parameter
    # Columns depend on the JSON data which we can't know statically
    return None, None


@PD.register("read_parquet")
def pd_read_parquet(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    """Handle pd.read_parquet() - uses 'columns' parameter."""
    columns = idx_or_key(args, keywords, key="columns")
    match columns:
        case list():
            cols = {col for col in columns if isinstance(col, str)}
            return cols, None
        case _:
            # No columns specified - all columns loaded but we don't know which
            return None, None


@PD.register("merge")
def pd_merge(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    """Handle pd.merge(left, right, ...) like left.merge(right, ...)."""
    left = idx_or_key(args, keywords, idx=0, key="left")
    if not isinstance(left, DF):
        return None, None
    right = idx_or_key(args, keywords, idx=1, key="right")
    params = bind_positional(args[2:], keywords, MERGE_PARAMS)
    return merge_columns(left.columns, left.label, right, params)


@PD.register("concat")
def pd_concat(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    """Handle pd.concat([df1, df2, ...]) for tracked frames."""
    objs = idx_or_key(args, keywords, idx=0, key="objs")
    if UNPACKED in keywords:
        return None, None
    if not (isinstance(objs, list) and objs and all(isinstance(o, DF) for o in objs)):
        # Series, dicts or unresolved frames: columns unknown
        return None, None
    frames = [o.columns for o in objs if isinstance(o, DF)]

    axis = keywords.get("axis", 0)
    if axis in (0, "index"):
        if keywords.get("join", "outer") == "inner":
            return set.intersection(*frames), None
        return set.union(*frames), None
    if axis in (1, "columns"):
        if "keys" in keywords or keywords.get("ignore_index") is True:
            # MultiIndex or renumbered columns
            return None, None
        return set.union(*frames), None
    return None, None
