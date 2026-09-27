# Frame Check Core

The core static analysis engine for frame-check that tracks pandas DataFrame schemas through Python AST.

## Architecture

### Core Classes

- **`FrameChecker`**: Main AST visitor that tracks DataFrame definitions and column accesses
- **`FrameInstance`**: Represents a DataFrame with its columns and metadata
- **`FrameHistory`**: Tracks DataFrame definitions across different lines for proper scoping

### Structural Pattern Matching

The Frame Check Core leverages Python's structural pattern matching to efficiently analyze and track DataFrame schemas. This approach allows for precise and readable code when handling complex AST nodes.

#### Consider using Pattern Matching

**Pattern Matching**: Simplifies the process of dissecting and interpreting AST nodes by using pattern matching to directly access relevant attributes:

```python
# Example of pattern matching in FrameChecker
match node:
    case ast.Call(
        func=ast.Attribute(value=ast.Name(), attr=attr), args=args, keywords=keywords
    ):
        if method := PD.get_method(attr):
            created, error = method(args, keywords, self.definitions)
            # Further processing...
```

**Enhanced Readability**: Reduces boilerplate code and improves readability by clearly defining patterns for AST node structures.

**Efficient Error Handling**: By matching specific patterns, errors can be caught and handled gracefully, improving the robustness of the static analysis engine.

#### Why Use Structural Pattern Matching?

1. **Simplifies Code**: Reduces the complexity of AST node traversal and manipulation.
2. **Improves Maintainability**: Patterns provide a clear structure, making the code easier to understand and modify.
3. **Boosts Performance**: Directly accesses node attributes without multiple conditional checks.
4. **Enhances Debugging**: Clearly defined patterns make it easier to trace and debug issues within the AST analysis.

## Important Notes for Static Analysis

1. **Dynamic column names**: `df[variable] = value` where `variable` contains the column name
2. **Computed column names**: `df[f'col_{i}'] = value` with f-strings or format
3. **Dictionary expansion**: `df = df.assign(**{f'col_{i}': i for i in range(5)})`
4. **Method chaining**: Operations that return new DataFrames vs in-place modifications
5. **Index as columns**: Operations like `reset_index()` can promote index to columns
6. **Conditional existence**: Columns might exist only in certain code paths
7. **External data**: `pd.read_csv()`, `pd.read_excel()` etc. introduce columns from files
8. **SQL queries**: `pd.read_sql()` columns depend on query
9. **Copy vs view**: Some operations create views that share column structure
```
