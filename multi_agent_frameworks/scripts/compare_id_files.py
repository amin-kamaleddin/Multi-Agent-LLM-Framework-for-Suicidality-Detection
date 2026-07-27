"""Compare record IDs stored one per line in two or more text files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence


_VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class IdFileError(ValueError):
    """Raised when an input path or ID file does not satisfy the contract."""


def _sort_key(record_id: str) -> tuple[int, int | str, str]:
    if record_id.isdecimal():
        return (0, int(record_id), record_id)
    return (1, record_id.casefold(), record_id)


def load_id_file(path: Path) -> set[str]:
    """Load and validate a UTF-8 ``.txt`` file containing one ID per row."""

    if path.suffix.lower() != ".txt":
        raise IdFileError(f"Expected a .txt file: {path}")
    if not path.exists():
        raise IdFileError(f"File does not exist: {path}")
    if not path.is_file():
        raise IdFileError(f"Path is not a file: {path}")

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise IdFileError(f"File is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise IdFileError(f"Could not read {path}: {exc}") from exc

    if not lines:
        raise IdFileError(f"File is empty: {path}")

    ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise IdFileError(f"Blank row at {path}:{line_number}")
        if line != line.strip():
            raise IdFileError(
                f"Leading or trailing whitespace at {path}:{line_number}"
            )
        if not _VALID_ID.fullmatch(line):
            raise IdFileError(
                f"Invalid record ID at {path}:{line_number}: {line!r}. "
                "Use one ID per row with only letters, digits, '.', '_', ':', or '-'."
            )
        if line in ids:
            raise IdFileError(f"Duplicate record ID {line!r} at {path}:{line_number}")
        ids.add(line)

    return ids


def compare_id_files(paths: Sequence[str | Path]) -> dict[str, object]:
    """Return IDs that are not present in every supplied file."""

    if len(paths) < 2:
        raise IdFileError("Provide at least two .txt files.")

    resolved_paths = [Path(path).expanduser().resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise IdFileError("Each input file path must be unique.")

    file_ids = [(path, load_id_file(path)) for path in resolved_paths]
    all_ids = set().union(*(ids for _, ids in file_ids))
    common_ids = set.intersection(*(ids for _, ids in file_ids))
    non_matching_ids = all_ids - common_ids

    return {
        "file_count": len(file_ids),
        "union_count": len(all_ids),
        "intersection_count": len(common_ids),
        "non_matching_count": len(non_matching_ids),
        "non_matching_ids": sorted(non_matching_ids, key=_sort_key),
        "files": [
            {
                "path": str(path),
                "id_count": len(ids),
                "missing_ids": sorted(all_ids - ids, key=_sort_key),
            }
            for path, ids in file_ids
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find IDs that occur in at least one input file but not in every input "
            "file. Each UTF-8 .txt file must contain exactly one record ID per row."
        )
    )
    parser.add_argument("files", nargs="+", help="Two or more ID .txt files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = compare_id_files(args.files)
    except IdFileError as exc:
        parser.error(str(exc))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
