# Contributing to frame-check

Welcome! This section covers how to extend and contribute to frame-check.

## Architecture Overview

The `frame-check-core` package is built around these key components:

```
frame-check-core/
├── checker.py          # Main AST visitor (entry point)
├── tracker.py          # Column dependency tracking
├── refs.py             # Type guards and ColumnRef dataclass
├── handlers/           # Operation handlers (what columns are CREATED/MODIFIED)
│   ├── models.py       # PD/DF registries for operation handlers
│   ├── pandas.py       # pd.* function handlers
│   └── dataframe.py    # df.* method handlers
├── extractors/         # Column extractors (what columns are READ)
│   ├── registry.py     # Extractor registry
│   ├── column.py       # df['col'] patterns
│   └── binop.py        # df['A'] + df['B'] patterns
├── diagnostic/         # Error message generation
└── config/             # Configuration management
```

### Handlers vs Extractors

These two modules serve complementary purposes:

| Module | Purpose | What they handle | Example |
|--------|---------|------------------|---------|
| **Handlers** | Track column state changes | Pandas API calls that create/modify/delete columns | `pd.DataFrame({'A': [1]})` creates 'A' |
| **Extractors** | Identify column references | Operations between columns and expressions | `df['A'] + df['B']` reads 'A' and 'B' |

**Handlers** (Pandas core methods) answer: "What columns exist after this operation?"
- Creating DataFrames: `pd.DataFrame()`, `pd.read_csv()`
- Adding columns: `df.assign(B=1)`, `df.insert()`
- Removing columns: `df.drop('A')`
- Resizing/reshaping frames: `df.melt()`, `df.pivot()`

**Extractors** (Operations between columns) answer: "What columns are being accessed here?"
- Column access: `df['A']`, `df[['A', 'B']]`
- Binary operations: `df['A'] + df['B']`, `df['A'] * 2`
- Comparisons: `df['A'] > df['B']`
- Assignments: `df['C'] = df['A'] + df['B']` (RHS is extracted)

**How they work together:**
The checker uses **handlers** to build a model of what columns should exist, then uses **extractors** to validate that accessed columns are actually present. For example, in `df['C'] = df['A'] + df['B']`, the extractor identifies that columns 'A' and 'B' are being read, while the handler knows that column 'C' is being created.

## Extension Points

frame-check is designed to be extensible. There are three main ways to add features:

| Extension Type | Registration | Use Case | Difficulty |
|---------------|-----------|----------|------------|
| [Pandas Function](./adding-pandas-function.md) | `@PD.register()` decorator | Add support for `pd.read_excel()`, `pd.concat()`, etc. | ⭐ Easy |
| [DataFrame Method](./adding-dataframe-method.md) | `@DF.register()` decorator | Add support for `df.drop()`, `df.rename()`, etc. | ⭐ Easy |
| [Extractor](./adding-extractor.md) | Add to `EXTRACTORS` list | Handle new column reference patterns | ⭐ Easy |

### Registry Patterns

#### Pandas Functions and DataFrame Methods

These use decorator-based registration:

```python
# Pandas functions
@PD.register("read_excel")
def pd_read_excel(args, keywords) -> PDFuncResult:
    ...

# DataFrame methods
@DF.register("drop")
def df_drop(columns, args, keywords) -> DFFuncResult:
    ...
```

This means:
- **No manual registration** - decorators handle it automatically
- **Automatic discovery** - just import the module
- **Easy testing** - registries can be cleared/modified in tests

#### Extractors

Extractors use explicit list-based registration in `registry.py`:

```python
# In registry.py
EXTRACTORS: list[ExtractorFunc] = [
    extract_column_ref,                # df['col'] - most common
    extract_column_refs_from_binop,    # df['A'] + df['B']
    extract_method_call,               # Add your extractor here
]
```

This means:
- **All in one place** - see all extractors and their order
- **Clear ordering** - earlier in the list = tried first
- **No sorting overhead** - list is already in order
- **Simple to add** - just import and add to the list

## Quick Start

1. **Clone the repository**
   ```sh
   git clone https://github.com/frame-check/frame-check.git
   cd frame-check
   ```

2. **Set up development environment**
   ```sh
   cd frame-check-core
   uv sync --group dev
   ```

3. **Run tests**
   ```sh
   uv run pytest
   ```

4. **Make your changes** following the guides above

5. **Add tests** for your new feature (see [Test Structure](#test-structure) below)

6. **Submit a PR** 🎉

## Test Structure

Tests are organized to mirror the source structure:

```
conftest.py                    # Pytest configuration and fixtures (at root for --support to work)
frame-check-core/tests/
├── test_checker.py          # Core checker tests
├── config/                  # Tests for config module
│   ├── test_config.py
│   └── test_paths.py
├── diagnostic/              # Tests for diagnostic module
│   ├── test_diagnostics.py
│   └── test_output.py
├── extractors/              # Tests for extractors module
│   ├── test_binop.py
│   ├── test_column.py
│   └── test_registry.py
├── features/                # Feature/API completeness tests
│   ├── test_column_assignment_methods.py  # CAM-* features
│   └── test_dataframe_creation_methods.py # DCMS-* features
└── util/                    # Tests for utility module
    └── test_similarity.py
```

### Where to Add Tests

| Test Type | Location | Example |
|-----------|----------|---------|
| Core checker functionality | `tests/test_checker.py` | Import detection, DataFrame tracking |
| Extractor unit tests | `tests/extractors/test_*.py` | AST pattern matching |
| Config tests | `tests/config/test_*.py` | Config loading, path handling |
| Diagnostic tests | `tests/diagnostic/test_*.py` | Error messages, formatting |
| Feature completeness | `tests/features/test_*.py` | Tests with `@pytest.mark.support` |

### Feature Tests

Tests in `tests/features/` track API completeness and are organized by categories from `scripts/features.toml`:

- `test_dataframe_creation_methods.py` - DCMS-* (DataFrame creation)
- `test_column_assignment_methods.py` - CAM-* (column assignment)

Use the `@pytest.mark.support(code="#DCMS-1")` marker to link tests to features.

Run `pytest --support` from the project root to update `scripts/features.toml` with test results and regenerate the supported features table in `README.md`.

## Design Principles

When contributing, keep these principles in mind:

- **Fail gracefully**: Return `None` when a pattern isn't recognized rather than crashing
- **Be conservative**: Only report errors when you're confident something is wrong
- **Compose existing tools**: Reuse extractors and utilities where possible
- **Use the registries**: Don't hardcode - use `@PD.register()`, `@DF.register()`, or add to the `EXTRACTORS` list
- **Test thoroughly**: Each feature should have corresponding tests
- **Document clearly**: Add docstrings and update relevant documentation

## What to Contribute

### High Impact, Easy to Add

- **Pandas functions**: `pd.read_excel`, `pd.read_json`, `pd.read_parquet`, `pd.concat`
- **DataFrame methods**: `df.drop`, `df.rename`, `df.copy`, `df.reset_index`
- **Extractors**: Method calls (`df['A'].fillna(df['B'])`), comparisons (`df['A'] > df['B']`)

### Medium Effort

- Method chaining support (`df.assign(A=1).drop('B')`)
- `from pandas import DataFrame` imports
- Groupby result column inference

### Advanced

- Control flow analysis (if/else branches)
- Function boundary analysis (parameters and returns)
- Polars support
