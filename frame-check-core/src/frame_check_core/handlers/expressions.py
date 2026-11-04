"""
Analysis of pandas expression strings, as passed to `df.eval()` and `df.query()`.

pandas expressions are Python-like with a few twists that are handled here:

- `@name` refers to a local variable, not a column.
- `` `my col` `` quotes a column label that isn't a valid identifier.
- Multi-line `eval` strings are evaluated line by line (each line is parsed on
  its own, blank lines are ignored).
- Bare names resolve to columns and index levels only. Variables in scope are
  *not* visible without `@` (verified against pandas 3).

The analysis is conservative: anything unexpected makes it give up (`None`)
so that no column is guessed or reported.
"""

import ast
import re
from dataclasses import dataclass

_LOCAL = "__fc_local__"
_QUOTED = "__fc_quoted_"

# Names that pandas resolves without a column of that name
# (`columns`, and `ilevel_N`/`clevel_N` for index/column levels)
_IMPLICIT_NAMES = frozenset({"index", "columns", "inf", "Inf"})
_LEVEL_NAME = re.compile(r"[ic]level_\d+")

# Constructs pandas rejects or that bind names we don't model: give up on them
_UNSUPPORTED = (
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.NamedExpr,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Starred,
)


@dataclass(frozen=True, slots=True)
class ExprLine:
    """One parsed line of an expression string.

    `target` is the name assigned by the line (None for a plain expression).
    `reads` are the names read in it, in source order, as column labels:
    quoted names appear without their backticks.
    `quoted` are the reads that were written in backticks.
    `target_quoted` is True when the assignment target was in backticks, which
    pandas stores under a mangled name.
    """

    target: str | None
    target_quoted: bool
    reads: tuple[str, ...]
    quoted: frozenset[str]


def _substitute(line: str) -> tuple[str, dict[str, str]] | None:
    """Replace `@name` and backtick-quoted labels with plain identifiers."""
    if _LOCAL in line or _QUOTED in line or '"""' in line or "'''" in line:
        return None
    out: list[str] = []
    quoted: dict[str, str] = {}
    i = 0
    quote: str | None = None
    while i < len(line):
        char = line[i]
        if quote is not None:
            out.append(char)
            if char == "\\" and i + 1 < len(line):
                out.append(line[i + 1])
                i += 1
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            out.append(char)
        elif char == "@":
            out.append(_LOCAL)
        elif char == "`":
            # A doubled backtick is a literal backtick in the label
            label: list[str] = []
            i += 1
            while i < len(line):
                if line[i] == "`":
                    if line[i + 1 : i + 2] != "`":
                        break
                    i += 1
                label.append(line[i])
                i += 1
            if i >= len(line):
                return None
            placeholder = f"{_QUOTED}{len(quoted)}__"
            quoted[placeholder] = "".join(label)
            out.append(placeholder)
        else:
            out.append(char)
        i += 1
    if quote is not None:
        return None
    return "".join(out), quoted


def _parse_line(line: str) -> ExprLine | None:
    substituted = _substitute(line)
    if substituted is None:
        return None
    source, quoted = substituted
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    if len(module.body) != 1:
        return None
    statement = module.body[0]

    target: str | None = None
    target_quoted = False
    match statement:
        case ast.Expr(value=value):
            pass
        case ast.Assign(targets=[ast.Name(id=name)], value=value):
            target_quoted = name in quoted
            target = quoted.get(name, name)
        case _:
            return None

    if any(isinstance(node, _UNSUPPORTED) for node in ast.walk(value)):
        return None

    callees = {
        id(node.func)
        for node in ast.walk(value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = sorted(
        (
            node
            for node in ast.walk(value)
            if isinstance(node, ast.Name)
            and id(node) not in callees
            and not node.id.startswith(_LOCAL)
        ),
        key=lambda node: node.col_offset,
    )
    reads: list[str] = []
    quoted_reads: set[str] = set()
    for node in names:
        label = quoted.get(node.id, node.id)
        if node.id in quoted:
            quoted_reads.add(label)
        if label not in reads:
            reads.append(label)
    return ExprLine(target, target_quoted, tuple(reads), frozenset(quoted_reads))


def parse_expression(expr: str, *, multiline: bool) -> list[ExprLine] | None:
    """
    Parse an `eval`/`query` string into its lines.

    Args:
        expr: The expression string.
        multiline: Whether several lines are allowed (`eval`) or not (`query`).

    Returns:
        The parsed lines, or None if the string can't be analyzed.
    """
    lines = [line.strip() for line in expr.splitlines() if line.strip()]
    if not lines or (len(lines) > 1 and not multiline):
        return None
    parsed = [_parse_line(line) for line in lines]
    if any(line is None for line in parsed):
        return None
    return [line for line in parsed if line is not None]


def _normalize(label: str) -> str:
    return re.sub(r"[\W_]+", "", label).lower()


def missing_reads(line: ExprLine, available: set[str]) -> list[str]:
    """The names read by `line` that pandas can't resolve to a column."""
    loose = {_normalize(column) for column in available}
    missing = []
    for name in line.reads:
        if name in available:
            continue
        if name in line.quoted:
            # pandas mangles non-identifier labels; don't guess on look-alikes
            if _normalize(name) in loose:
                continue
        elif name in _IMPLICIT_NAMES or _LEVEL_NAME.fullmatch(name):
            continue
        missing.append(name)
    return missing
