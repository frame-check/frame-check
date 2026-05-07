from .models import PD, PDFuncResult, Result, idx_or_key


@PD.register("DataFrame")
def pd_dataframe(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    data = idx_or_key(args, keywords, idx=0, key="data")
    match data:
        case dict():
            return {k for k in data.keys() if isinstance(k, str)}, None
        case list():
            columns: set[str] = set()
            for item in data:
                if not isinstance(item, dict):
                    return None, None
                columns |= {k for k in item.keys() if isinstance(k, str)}
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
