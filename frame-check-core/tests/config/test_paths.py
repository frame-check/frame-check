import os
import tempfile
from pathlib import Path

import pytest
from frame_check_core.config import Config
from frame_check_core.config.paths import (
    any_match,
    collect_python_files,
    parse_filepath,
)


@pytest.mark.parametrize(
    "exclude, target, should_exclude",
    [
        ({".tmp/"}, ".tmp/test/file.py", True),
        ({"file.py"}, "other_file.py", False),
        ({"file.py"}, "file.py", True),
        ({"dir/file.py"}, "dir/file.py", True),
        ({"dir/file.py"}, "dir/other_file.py", False),
        ({"dir/file.py"}, "other_dir/file.py", False),
        ({"dir/"}, "dir/file.py", True),
        ({"dir/"}, "dir/subdir/file.py", True),
        ({"dir/"}, "other_dir/file.py", False),
        ({"*.py"}, "file.py", True),
        ({"src/*.py"}, "src/file.py", True),
        ({"src/*.py"}, "src/subdir/file.py", False),
        ({"**/*.py"}, "src/subdir/file.py", True),
        ({"src/*/file.py"}, "src/subdir/file.py", True),
        ({"src/*/file.py"}, "src/another/other.py", False),
        ({"*.txt"}, "file.py", False),
        # Multiple exclude patterns
        ({"*.txt", "*.py"}, "file.py", True),
        ({"*.txt", "*.py"}, "file.txt", True),
        ({"*.txt", "*.py"}, "file.md", False),
        ({"src/", "tests/"}, "src/file.py", True),
        ({"src/", "tests/"}, "tests/file.py", True),
        ({"src/", "tests/"}, "docs/file.py", False),
        ({"src/*.py", "tests/*.py"}, "src/file.py", True),
        ({"src/*.py", "tests/*.py"}, "tests/file.py", True),
        ({"src/*.py", "tests/*.py"}, "src/subdir/file.py", False),
        # More complex combinations
        ({"s**.py"}, "src/file.py", True),
        ({"s**c/*.py"}, "src/file.py", True),
        ({"s**.py"}, "docs/file.py", False),
        ({"*.py", "src/**/*.txt"}, "src/subdir/file.txt", True),
        (
            {"*.py", "src/**/*.txt"},
            "src/file.py",
            False,
        ),  # This was originally marked true, but should be false, because *.py is not recursive.
        ({"*.py", "src/**/*.txt"}, "src/subdir/file.md", False),
        ({"dir/", "**/*.py", "*.md"}, "other/file.py", True),
        # other wildcards
        ({"file?.py"}, "file1.py", True),
        ({"file?.py"}, "fileA.py", True),
        ({"file?.py"}, "file10.py", False),
        ({"file[ab].py"}, "filea.py", True),
        ({"file[ab].py"}, "fileb.py", True),
        ({"file[ab].py"}, "filec.py", False),
    ],
)
def test_any_match(exclude: set[str], target: str, should_exclude: bool):
    conf = Config()
    conf.update_exclude(exclude)
    target_path = Path(target)
    assert any_match(target_path.resolve(), conf.exclude) == should_exclude


def test_parse_filepath():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        base_path = Path(tmpdir).resolve()
        # Create test files and directories
        (base_path / "file1.py").touch()
        (base_path / "file2.txt").touch()
        (base_path / "subdir").mkdir()
        (base_path / "subdir" / "file3.py").touch()
        (base_path / "subdir" / "file4.md").touch()
        os.chdir(base_path)

        # Test single file
        files = list(
            parse_filepath((base_path / "file1.py").as_posix(), recursive=True)
        )
        assert files == [base_path / "file1.py"]

        # Test directory
        files = list(parse_filepath((base_path / "subdir/").as_posix(), recursive=True))
        assert set(files) == {
            base_path / "subdir" / "file3.py",
            base_path / "subdir" / "file4.md",
            base_path / "subdir",
        }

        # Test glob pattern
        files = list(parse_filepath((base_path / "*.py").as_posix(), recursive=True))
        assert set(files) == {base_path / "file1.py"}

        # Test recursive glob pattern
        files = list(parse_filepath((base_path / "**/*.py").as_posix(), recursive=True))
        assert set(files) == {
            base_path / "file1.py",
            base_path / "subdir" / "file3.py",
        }

        # Test '.'
        files = list(parse_filepath(".", recursive=True))
        assert set(files) == {
            base_path / "file1.py",
            base_path / "file2.txt",
            base_path / "subdir" / "file3.py",
            base_path / "subdir" / "file4.md",
            base_path / "subdir",
            base_path,
        }


def test_collect_python_files():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        base_path = Path(tmpdir).resolve()
        os.chdir(base_path)
        # Create test files and directories
        (base_path / "file1.py").touch()
        (base_path / "file2.txt").touch()
        (base_path / "subdir").mkdir()
        (base_path / "subdir" / "file3.py").touch()
        (base_path / "subdir" / "file4.md").touch()
        (base_path / "ignore_dir").mkdir()
        (base_path / "ignore_dir" / "file5.py").touch()

        # Test collecting Python files without exclusions
        files = collect_python_files(
            [base_path.as_posix()],
            exclusion_patterns=[],
            recursive=True,
        )
        assert set(files) == {
            base_path / "file1.py",
            base_path / "subdir" / "file3.py",
            base_path / "ignore_dir" / "file5.py",
        }

        # Test collecting Python files with exclusions
        files = collect_python_files(
            [base_path.as_posix()],
            exclusion_patterns=[f"{base_path.as_posix()}/ignore_dir/**"],
            recursive=True,
        )
        assert set(files) == {
            base_path / "file1.py",
            base_path / "subdir" / "file3.py",
        }

        # Test collecting Python files with non-recursive option
        files = collect_python_files(
            [base_path.as_posix()],
            exclusion_patterns=[],
            recursive=False,
        )
        assert set(files) == {base_path / "file1.py"}

        files = collect_python_files(
            [(base_path / "s**r").as_posix() + "/"],
            exclusion_patterns=[],
            recursive=True,
        )
        assert set(files) == {base_path / "subdir" / "file3.py"}
