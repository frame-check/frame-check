import ast
import contextlib
import sys

from frame_check_core import Checker
from frame_check_core.diagnostic import Diagnostic
from lsprotocol import types
from pygls.cli import start_server
from pygls.lsp.server import LanguageServer

server = LanguageServer("frame-check-lsp", "v0.1")
fc = Checker()

# Store diagnostics with their suggestions for code actions
_diagnostic_suggestions: dict[str, list[tuple[types.Diagnostic, Diagnostic]]] = {}


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
async def frame_diagnostics(
    ls: LanguageServer, params: types.DidOpenTextDocumentParams
):
    global fc
    text_doc = ls.workspace.get_text_document(params.text_document.uri)
    contents = text_doc.source
    ls_diagnostics: list[types.Diagnostic] = []
    uri = text_doc.uri

    # Clear previous suggestions for this document
    _diagnostic_suggestions[uri] = []

    with contextlib.suppress(SyntaxError):
        tree = ast.parse(contents)
        fc = fc.check(tree)
        for diagnostic in fc.diagnostics:
            ls_diagnostic = types.Diagnostic(
                range=types.Range(
                    start=types.Position(*diagnostic.region.start.as_lsp_position()),
                    end=types.Position(*diagnostic.region.end.as_lsp_position()),
                ),
                message=diagnostic.message,
                source="Frame Checker",
                severity=types.DiagnosticSeverity.Error,
            )

            ls_diagnostics.append(ls_diagnostic)

            # Store diagnostic with suggestion for code actions
            if diagnostic.name_suggestion is not None:
                _diagnostic_suggestions[uri].append((ls_diagnostic, diagnostic))

            if diagnostic.hint is not None and diagnostic.definition_region is not None:
                # Hint at DataFrame creation site
                creation_hint_msg = "\n".join(diagnostic.hint)
                ls_diagnostics.append(
                    types.Diagnostic(
                        range=types.Range(
                            start=types.Position(
                                *diagnostic.definition_region.start.as_lsp_position()
                            ),
                            end=types.Position(
                                *diagnostic.definition_region.end.as_lsp_position()
                            ),
                        ),
                        message=creation_hint_msg,
                        source="Frame Checker",
                        severity=types.DiagnosticSeverity.Hint,
                    )
                )

        # Send diagnostics (moved outside the loop to always run, even with empty diagnostics)
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(uri=text_doc.uri, diagnostics=ls_diagnostics)
        )


@server.feature(
    types.TEXT_DOCUMENT_CODE_ACTION,
    types.CodeActionOptions(code_action_kinds=[types.CodeActionKind.QuickFix]),
)
def code_actions(
    ls: LanguageServer, params: types.CodeActionParams
) -> list[types.CodeAction]:
    """Provide code actions for column rename suggestions."""
    uri = params.text_document.uri
    actions: list[types.CodeAction] = []

    if uri not in _diagnostic_suggestions:
        return actions

    text_doc = ls.workspace.get_text_document(uri)
    contents = text_doc.source
    lines = contents.splitlines(keepends=True)

    for ls_diagnostic, core_diagnostic in _diagnostic_suggestions[uri]:
        # Check if the diagnostic range intersects with the requested range
        diag_range = ls_diagnostic.range
        req_range = params.range

        # Simple intersection check
        if (
            diag_range.end.line < req_range.start.line
            or diag_range.start.line > req_range.end.line
        ):
            continue

        suggestion = core_diagnostic.name_suggestion
        if suggestion is None:
            continue

        # Get the text at the diagnostic range to find the column name to replace
        start_line = diag_range.start.line
        start_char = diag_range.start.character
        end_line = diag_range.end.line
        end_char = diag_range.end.character

        # Extract the text in the diagnostic range
        if start_line == end_line and start_line < len(lines):
            original_text = lines[start_line][start_char:end_char]
        else:
            # Multi-line range - just use the first line for now
            if start_line < len(lines):
                original_text = lines[start_line][start_char:]
            else:
                continue

        # Find the quoted column name in the original text and create replacement
        # The diagnostic region covers the subscript like df['col'], we need to replace just the column name
        new_text = original_text
        for quote in ["'", '"']:
            if quote in original_text:
                # Find the column name within quotes
                parts = original_text.split(quote)
                if len(parts) >= 3:
                    # Replace the column name (parts[1]) with suggestion
                    parts[1] = suggestion
                    new_text = quote.join(parts)
                    break

        if new_text == original_text:
            continue

        # Create the text edit
        text_edit = types.TextEdit(
            range=diag_range,
            new_text=new_text,
        )

        # Create the workspace edit
        workspace_edit = types.WorkspaceEdit(
            changes={uri: [text_edit]},
        )

        # Create the code action
        action = types.CodeAction(
            title=f"frame-check: Rename column to '{suggestion}'",
            kind=types.CodeActionKind.QuickFix,
            diagnostics=[ls_diagnostic],
            edit=workspace_edit,
        )
        actions.append(action)

    return actions


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: types.HoverParams) -> types.Hover | None:
    """Show DataFrame columns on hover."""
    text_doc = ls.workspace.get_text_document(params.text_document.uri)
    contents = text_doc.source
    lines = contents.splitlines()

    line_num = params.position.line
    char_pos = params.position.character

    if line_num >= len(lines):
        return None

    line = lines[line_num]

    # Find the word at the cursor position
    # Match Python identifiers
    word_start = char_pos
    word_end = char_pos

    # Expand left to find start of word
    while word_start > 0 and (
        line[word_start - 1].isalnum() or line[word_start - 1] == "_"
    ):
        word_start -= 1

    # Expand right to find end of word
    while word_end < len(line) and (line[word_end].isalnum() or line[word_end] == "_"):
        word_end += 1

    if word_start == word_end:
        return None

    word = line[word_start:word_end]

    # Check if this word is a known DataFrame
    if word not in fc.dfs:
        return None

    tracker = fc.dfs[word]
    columns = list(tracker.columns.keys())

    if not columns:
        hover_text = f"**DataFrame `{word}`**\n\nNo columns detected."
    else:
        sorted_cols = sorted(columns)
        cols_formatted = ", ".join(f"`{c}`" for c in sorted_cols)
        hover_text = (
            f"**DataFrame `{word}`**\n\n**Columns ({len(columns)}):** {cols_formatted}"
        )

    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value=hover_text,
        ),
        range=types.Range(
            start=types.Position(line=line_num, character=word_start),
            end=types.Position(line=line_num, character=word_end),
        ),
    )


def main():
    # Remove --stdio argument if present, as pygls handles stdio by default
    args = [arg for arg in sys.argv[1:] if arg != "--stdio"]

    # Temporarily replace sys.argv to remove --stdio
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + args

    try:
        start_server(server)
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()
