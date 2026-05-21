from .models import DF, DFFuncResult, Result, idx_or_key


@DF.register("assign")
def df_assign(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    returned = columns | set(keywords.keys())
    return columns, returned, None


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
