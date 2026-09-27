/// admonition | Warning
    type: warning


This project is currently under active development and is not considered polished. You are welcome to fork it, contribute to making it more stable, or raise issues.
///

# frame-check
**A static checker for pandas DataFrames**

## Why frame-check?

Working with pandas DataFrames can be error-prone when it comes to column access. How many times have you written code like this, unsure if the column actually exists?

```python
# Will this work? 🤔
result = df["customer_id"]
filtered = df[df["status"] == "active"]
```

**The current reality:**

- ✅ Code runs fine in development with your test data

- ❌ Crashes in production when a column is missing

- 😰 Hours spent debugging runtime `KeyError` exceptions

## The Problem

When accessing DataFrame columns, you typically have to choose between:

1. **Manual verification** - Tediously trace through your code to verify every column reference
2. **Runtime checks** - Add defensive programming with `if 'column' in df.columns:` everywhere
3. **Source check** - Verify columns existence in your file or database schema
4. **Cross your fingers** - Hope the columns exist and deal with crashes later

```python
# Defensive programming gets verbose quickly
if "customer_id" in df.columns and "status" in df.columns:
    result = df[df["status"] == "active"]["customer_id"]
else:
    raise ValueError("Missing required columns")
```

## The Solution

**frame-check** brings static analysis to pandas DataFrames - just like `mypy` does for Python types. It tracks DataFrame schemas through your code and catches column access errors *before* your code runs.

### See it in action:

```python
import pandas as pd

# frame-check knows this DataFrame has columns: Name, Age, City, Salary
df = pd.DataFrame(
    {
        "Name": ["Alice", "Bob"],
        "Age": [25, 30],
        "City": ["NYC", "LA"],
        "Salary": [50000, 60000],
    }
)

# ❌ This will be caught by frame-check
result = df["customer_id"]  # Column doesn't exist!
```


```sh
frame-check example.py
```

**Error output:**
```
example.py:12:10: Column 'customer_id' does not exist on DataFrame 'df'.
   |
12 | result = df["customer_id"]
   |          ^^^^^^^^^^^^^^^^^
   |
   = available: Age, City, Name, Salary
```

## Key Benefits

- 🚀 **Catch errors early** - Find column access issues during development, not production
- 🧠 **Smart tracking** - Understands DataFrame transformations like `groupby()`, `assign()`, and column assignments
- 🔧 **Editor integration** - Real-time error highlighting in your favorite editor via LSP
- 📝 **Clear diagnostics** - Helpful error messages that show exactly where DataFrames were defined
- ⚡ **Zero runtime overhead** - Pure static analysis, no impact on your running code

**frame-check** - Because DataFrame bugs shouldn't be a surprise! 🐼✨



## How to install

=== "git clone + uv (Manual dev install)"

    ```sh
    git clone https://github.com/frame-check/frame-check.git
    cd frame-check/frame-check-core
    uv tool install . -e
    ```

<!--=== "uv"

    ```sh
    uv tool install frame-check
    ```-->
