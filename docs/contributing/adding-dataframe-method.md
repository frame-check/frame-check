# Adding a DataFrame Method

This guide shows how to add support for a new DataFrame method like `df.drop()`, `df.rename()`, or `df.merge()`.

## Overview

DataFrame methods are registered in the `DF` class registry using the `@DF.register()` decorator. Each handler receives the current columns and parsed arguments, then returns the updated column state.

**Location**: `frame-check-core/src/frame_check_core/handlers/dataframe.py`

## The Pattern

```python
from .models import DF, DFFuncResult, Result, idx_or_key


@DF.register("method_name")
def df_method_name(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # columns = current columns on the DataFrame
    # Modify columns based on method behavior
    # Return (updated_columns, returned_columns, error)
    ...
```

### Return Type: `DFFuncResult`

```python
DFFuncResult = tuple[set[str], set[str] | None, IllegalAccess | None]
```

- First element: Updated columns on the original DataFrame (for in-place operations)
- Second element: Columns on the returned DataFrame, or `None` if method doesn't return a new DataFrame
- Third element: Error if the call is invalid, or `None` (see [Reporting errors](#reporting-errors))

### Understanding the Return Values

| Method Type | `updated` | `returned` | Example |
|------------|-----------|------------|---------|
| In-place mutation | modified columns | `None` | `df.insert()` |
| Returns new DataFrame | original columns | new columns | `df.assign()` |
| Returns non-DataFrame | original columns | `None` | `df.to_dict()` |

## Example: Existing `df.assign()`

```python
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
```

## Reporting errors

If the call raises `KeyError` at runtime for unknown columns, return
`IllegalAccess(missing=[...])` as the third element. The checker turns it into a
diagnostic with suggestions, for example
`df.drop(): column 'X' does not exist on DataFrame 'df'.`

```python
from ..diagnostic import IllegalAccess

if missing := labels - columns:
    return columns, columns - labels, IllegalAccess(missing=sorted(missing))
```

Only report what pandas really rejects. For example, `fillna({"X": 0})` silently
ignores unknown keys, so it must not be reported. When unsure, check the behavior
against pandas.

## Callables in arguments

A single-argument lambda is resolved to a `ColumnLambda` whose `columns` lists
the labels it reads from its argument (`lambda x: x["A"] + x["B"]` gives
`("A", "B")`). Use it to validate callables such as those passed to `assign()`.

## Where handlers run

You don't need to handle call sites yourself. The checker runs a registered
handler wherever the method is called on a tracked frame:

- as an assignment (`df2 = df.method(...)`) or a statement (`df.method(..., inplace=True)`)
- in method chains (`df.method(...).assign(...)`) and column selections (`df[["A"]].method(...)`)
- in any expression (`print(df.method(...))`)

## Example: Existing `df.insert()`

```python
@DF.register("insert")
def df_insert(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # insert() modifies in-place, adding a column at a specific position
    column = idx_or_key(args, keywords, idx=1, key="column")
    if isinstance(column, str):
        columns.add(column)
    return columns, None, None
```

## Step-by-Step: Adding `df.drop()`

### Step 1: Understand the pandas API

```python
df.drop(
    labels=None,           # Index or column labels to drop
    axis=0,                # 0 for rows, 1 for columns
    columns=None,          # Alternative to labels when axis=1
    inplace=False,         # Modify in place or return new DataFrame
    ...
)
```

### Step 2: Add the handler

```python
@DF.register("drop")
def df_drop(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    # Get columns to drop (can be specified multiple ways)
    cols_to_drop: set[str] = set()
    
    # Method 1: df.drop(columns=['A', 'B'])
    drop_cols = idx_or_key(args, keywords, key="columns")
    if isinstance(drop_cols, str):
        cols_to_drop.add(drop_cols)
    elif isinstance(drop_cols, list):
        cols_to_drop.update(c for c in drop_cols if isinstance(c, str))
    
    # Method 2: df.drop(['A', 'B'], axis=1)
    labels = idx_or_key(args, keywords, idx=0, key="labels")
    axis = idx_or_key(args, keywords, idx=1, key="axis")
    if axis == 1 or axis == "columns":
        if isinstance(labels, str):
            cols_to_drop.add(labels)
        elif isinstance(labels, list):
            cols_to_drop.update(c for c in labels if isinstance(c, str))
    
    # Calculate new columns
    new_columns = columns - cols_to_drop
    
    # Check inplace parameter
    inplace = idx_or_key(args, keywords, key="inplace")
    if inplace is True:
        return new_columns, None, None
    else:
        return columns, new_columns, None
```

### Step 3: Add tests

Add tests to `frame-check-core/tests/test_checker.py`:

```python
def test_drop_single_column():
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
df2 = df.drop(columns="A")
df2["B"]  # Valid
df2["A"]  # Invalid - was dropped
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df2"].columns.keys()) == {"B", "C"}
    assert len(fc.diagnostics) == 1
    assert "A" in fc.diagnostics[0].message


def test_drop_multiple_columns():
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
df2 = df.drop(columns=["A", "B"])
df2["C"]  # Valid
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df2"].columns.keys()) == {"C"}
    assert len(fc.diagnostics) == 0


def test_drop_with_axis():
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1], "B": [2]})
df2 = df.drop("A", axis=1)
df2["B"]  # Valid
df2["A"]  # Invalid
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df2"].columns.keys()) == {"B"}
    assert len(fc.diagnostics) == 1


def test_drop_original_unchanged():
    code = """
import pandas as pd
df = pd.DataFrame({"A": [1], "B": [2]})
df2 = df.drop(columns="A")
df["A"]  # Still valid on original
"""
    fc = Checker.check(code)
    assert set(fc.dfs["df"].columns.keys()) == {"A", "B"}
    assert len(fc.diagnostics) == 0
```

### Step 4: Run tests

```sh
cd frame-check-core
uv run pytest tests/test_checker.py -v
```

## More Examples

### Adding `df.rename()`

```python
@DF.register("rename")
def df_rename(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    col_mapping = idx_or_key(args, keywords, key="columns")
    
    if not isinstance(col_mapping, dict):
        # Can't determine rename statically
        return columns, columns.copy(), None
    
    new_columns = set()
    for col in columns:
        if col in col_mapping and isinstance(col_mapping[col], str):
            new_columns.add(col_mapping[col])
        else:
            new_columns.add(col)
    
    inplace = idx_or_key(args, keywords, key="inplace")
    if inplace is True:
        return new_columns, None, None
    else:
        return columns, new_columns, None
```

### Methods that keep all columns

Methods such as `copy()`, `head()` or `sort_values()` return a frame with the same
columns. Add them to the decorators on `df_same_columns`, or use
`_same_columns(..., required=...)` when an argument names columns that must
exist (like `sort_values(by=...)`).

### Adding `df.reset_index()`

```python
@DF.register("reset_index")
def df_reset_index(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    drop = idx_or_key(args, keywords, key="drop")
    
    if drop is True:
        # Index is discarded, columns unchanged
        new_columns = columns.copy()
    else:
        # Index becomes a column named 'index' (or the index name)
        new_columns = columns | {"index"}
    
    inplace = idx_or_key(args, keywords, key="inplace")
    if inplace is True:
        return new_columns, None, None
    else:
        return columns, new_columns, None
```

### Adding `df.set_index()`

```python
@DF.register("set_index")
def df_set_index(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    keys = idx_or_key(args, keywords, idx=0, key="keys")
    drop = idx_or_key(args, keywords, key="drop")
    
    # Default is drop=True
    if drop is not False:
        cols_to_remove: set[str] = set()
        if isinstance(keys, str):
            cols_to_remove.add(keys)
        elif isinstance(keys, list):
            cols_to_remove.update(k for k in keys if isinstance(k, str))
        new_columns = columns - cols_to_remove
    else:
        new_columns = columns.copy()
    
    inplace = idx_or_key(args, keywords, key="inplace")
    if inplace is True:
        return new_columns, None, None
    else:
        return columns, new_columns, None
```

### Adding `df.filter()`

```python
@DF.register("filter")
def df_filter(
    columns: set[str], args: list[Result], keywords: dict[str, Result]
) -> DFFuncResult:
    items = idx_or_key(args, keywords, key="items")
    
    if isinstance(items, list):
        # Only keep specified columns that exist
        filtered = {c for c in items if isinstance(c, str) and c in columns}
        return columns, filtered, None
    
    # regex, like, axis parameters are harder to handle statically
    # Return all columns as a conservative estimate
    return columns, columns.copy(), None
```

## Handling Method Chaining

Currently, method chaining like `df.drop("A").assign(B=1)` requires additional work in the checker. If you're interested in implementing this, see the `test_assign_chain` test marked as `xfail`.

## Utility Functions

### `idx_or_key(args, keywords, idx=None, key=None)`

Retrieves a value from positional args or keyword args:

```python
# For df.drop("A") or df.drop(labels="A")
labels = idx_or_key(args, keywords, idx=0, key="labels")
```

### Working with `columns` parameter

The `columns` parameter passed to your handler is a **copy** of the current columns. You can modify it freely without affecting the original:

```python
def df_some_method(columns: set[str], args, keywords) -> DFFuncResult:
    # Safe to modify - it's already a copy
    columns.add("new_col")
    columns.discard("old_col")
    return columns, None, None
```

## Tips

1. **Handle `inplace` parameter**: Many pandas methods support `inplace=True`. When true, modify the `updated` columns; when false, keep `updated` as original and put changes in `returned`.

2. **Be conservative**: If you can't determine the result statically, return the columns unchanged rather than guessing wrong.

3. **Consider edge cases**: What happens with empty column lists? Invalid column names? Handle these gracefully.

4. **Check pandas docs**: The exact behavior of methods can be subtle. Verify against the official pandas documentation.

5. **Test both assignment patterns**:
   ```python
   # Pattern 1: Assign result to new variable
   df2 = df.drop(columns="A")
   
   # Pattern 2: Reassign to same variable
   df = df.drop(columns="A")

   # Pattern 3: Chained and in-place calls
   df2 = df.assign(B=1).drop(columns="A")
   df.drop(columns="A", inplace=True)
   ```

## Common Patterns

### Methods that add columns
```python
return columns, columns | new_cols, None
```

### Methods that remove columns
```python
return columns, columns - removed_cols, None
```

### Methods that replace columns
```python
return columns, new_column_set, None
```

### In-place methods
```python
if inplace:
    return modified_columns, None, None
else:
    return columns, modified_columns, None
```
