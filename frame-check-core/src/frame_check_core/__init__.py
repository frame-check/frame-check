"""
frame-check: A static column checker for dataframes!
"""

import argparse
import sys
from pathlib import Path

from .checker import Checker
from .config import Config, collect_python_files
from .formatting import format_diagnostic_rich


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for frame-check CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="frame-check",
        description="A static checker for dataframes!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "files",
        type=str,
        nargs="*",
        help="Python files (file.py), directories (dir/) or glob patterns (dir/**/*.py) to check. Directories will be searched recursively by default.",
    )

    parser.add_argument(
        "--ignore",
        type=str,
        nargs="+",
        help="Files (file.py), directories (dir/) or glob patterns (dir/**/*.py) to ignore during checking.",
        default=[],
    )
    parser.add_argument(
        "--non-recursive",
        "-n",
        action="store_true",
        help="Do not recursively check directories for Python files.",
        default=False,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to a configuration file (if not specified, configuration will be read frame-check.toml or pyproject.toml, if present).",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser


def main(argv: list[str] | None = None, override_config: Config | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.files:
        parser.print_help(sys.stderr)
        return 0

    if override_config is None:
        config = Config()

        if args.config:
            try:
                config = Config.load_from(args.config)
            except Exception as e:  # noqa: BLE001 - report any config error to the user
                print(
                    f"Error loading configuration from {args.config}:\n{e}",
                    file=sys.stderr,
                )
                return 1
        elif (frame_check_settings := Path.cwd() / "frame-check.toml").exists():
            config = Config.load_from(frame_check_settings)
        elif (pyproject_settings := Path.cwd() / "pyproject.toml").exists():
            config = Config.load_from(pyproject_settings)
        config.update(
            exclude=args.ignore,
            recursive=not args.non_recursive,
        )
    else:
        config = override_config

    python_files = collect_python_files(
        args.files,
        exclusion_patterns=config.exclude,
        recursive=config.recursive,
    )

    if not python_files:
        print("No Python files found to check.", file=sys.stderr)
        return 0

    # Track if any file has diagnostics
    has_errors = False

    # Process each file
    for file_path in python_files:
        try:
            source_code = file_path.read_text()
            checker = Checker.check(file_path)
            if checker.diagnostics:
                has_errors = True
                for diag in checker.diagnostics:
                    print(
                        format_diagnostic_rich(diag, file_path, source_code=source_code)
                    )

        except SyntaxError as e:
            print(f"Syntax error in {file_path}:\n{e}", file=sys.stderr)
            has_errors = True
        except Exception as e:  # noqa: BLE001 - keep checking the remaining files
            print(f"Error checking {file_path}:\n{e}", file=sys.stderr)
            has_errors = True

    return 1 if has_errors else 0
