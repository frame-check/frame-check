# Adding a Pandas Function

This guide shows how to add support for a new pandas function like `pd.read_excel()`, `pd.concat()`, or `pd.merge()`.

## Overview

Pandas functions are registered in the `PD` class registry using the `@PD.register()` decorator. Each handler receives parsed arguments and returns the columns that would exist on the resulting DataFrame.

**Location**: `frame-check-core/src/frame_check_core/handlers/pandas.py`

## The Pattern

```python
from .models import PD, PDFuncResult, Result, idx_or_key


@PD.register("function_name")
def pd_function_name(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    # Extract relevant arguments
    # Determine columns from arguments
    # Return (columns_set, error) or (None, None) if unknown
    ...
```

### Return Type: `PDFuncResult`

```python
PDFuncResult = tuple[set[str] | None, IllegalAccess | None]
```

- First element: Set of column names, or `None` if columns can't be determined
- Second element: Error if the call is invalid, or `None`

## Example: Existing `pd.DataFrame()`

```python
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
```

## Example: Existing `pd.read_csv()`

```python
@PD.register("read_csv")
def pd_read_csv(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    usecols = idx_or_key(args, keywords, key="usecols")
    match usecols:
        case str():
            return {usecols}, None
        case list():
            cols = set()
            for col in usecols:
                if isinstance(col, str):
                    cols.add(col)
                elif isinstance(col, list):
                    cols.update(x for x in col if isinstance(x, str))
            return cols, None
        case _:
            return None, None
```

## Step-by-Step: Adding `pd.read_excel()`

### Step 1: Understand the pandas API

```python
pd.read_excel(
    io,                    # File path or buffer
    sheet_name=0,          # Sheet to read
    usecols=None,          # Columns to read (our focus!)
    names=None,            # Column names to use
    ...
)
```

### Step 2: Add the handler

```python
@PD.register("read_excel")
def pd_read_excel(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    # Check if explicit column names are provided
    names = idx_or_key(args, keywords, key="names")
    if isinstance(names, list):
        return {n for n in names if isinstance(n, str)}, None
    
    # Check usecols parameter
    usecols = idx_or_key(args, keywords, key="usecols")
    match usecols:
        case str():
            return {usecols}, None
        case list():
            return {c for c in usecols if isinstance(c, str)}, None
        case _:
            # Can't determine columns statically
            return None, None
```

### Step 3: Add tests

Add tests to `frame-check-core/tests/test_checker.py`:

```python
def test_read_excel_with_usecols():
    code = """
import pandas as pd
df = pd.read_excel("data.xlsx", usecols=["A", "B", "C"])
df["A"]  # Valid
df["X"]  # Invalid
"""
    fc = Checker.check(code)
    assert "df" in fc.dfs
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B", "C"}
    assert len(fc.diagnostics) == 1
    assert "X" in fc.diagnostics[0].message


def test_read_excel_with_names():
    code = """
import pandas as pd
df = pd.read_excel("data.xlsx", names=["col1", "col2"])
df["col1"]  # Valid
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"col1", "col2"}
    assert len(fc.diagnostics) == 0


def test_read_excel_unknown_columns():
    code = """
import pandas as pd
df = pd.read_excel("data.xlsx")  # No usecols, can't determine columns
"""
    fc = Checker.check(code)
    # DataFrame should not be tracked since columns are unknown
    assert "df" not in fc.dfs
```

### Step 4: Run tests

```sh
cd frame-check-core
uv run pytest tests/test_read_excel.py -v
```

## More Examples

### Adding `pd.read_json()`

```python
@PD.register("read_json")
def pd_read_json(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    # read_json doesn't have usecols, columns come from JSON structure
    # We can't determine them statically in most cases
    return None, None
```

### Adding `pd.concat()`

```python
@PD.register("concat")
def pd_concat(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    objs = idx_or_key(args, keywords, idx=0, key="objs")
    
    # This would require tracking multiple DataFrames
    # For now, return None (future enhancement)
    return None, None
```

### Adding `pd.read_parquet()` with columns parameter

```python
@PD.register("read_parquet")
def pd_read_parquet(args: list[Result], keywords: dict[str, Result]) -> PDFuncResult:
    columns = idx_or_key(args, keywords, key="columns")
    match columns:
        case list():
            return {c for c in columns if isinstance(c, str)}, None
        case _:
            return None, None
```

## Utility Functions

### `idx_or_key(args, keywords, idx=None, key=None)`

Retrieves a value from positional args or keyword args:

```python
# For pd.DataFrame(data) or pd.DataFrame(data=data)
data = idx_or_key(args, keywords, idx=0, key="data")
```

### `Result` type

The `Result` type represents parsed argument values:

```python
Result = Union[str, dict, list, PD, PDMethod, DF, DFMethod, _Unknown]
```

- `str`: String literal
- `dict`: Dictionary literal (keys are column names for DataFrames)
- `list`: List literal
- `_Unknown`: Value couldn't be determined statically

## Tips

1. **Return `None` when uncertain**: If you can't determine columns statically, return `(None, None)`. This prevents false positives.

2. **Handle variables**: The `definitions` dict (passed to the handler wrapper) resolves variable references. If a user writes:
   ```python
   cols = ["A", "B"]
   df = pd.read_csv("file.csv", usecols=cols)
   ```
   The `usecols` will be resolved to `["A", "B"]`.

3. **Check multiple parameters**: Some functions have multiple ways to specify columns (e.g., `usecols`, `names`, `columns`). Check all relevant ones.

4. **Use pattern matching**: Python's `match` statement makes handling different argument types clean and readable.
