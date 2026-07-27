from pathlib import Path

import pytest

from scripts.compare_id_files import IdFileError, compare_id_files, load_id_file


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_compare_reports_union_minus_intersection_and_per_file_missing_ids(
    tmp_path: Path,
) -> None:
    first = _write(tmp_path / "first.txt", "1\n2\n10\nshared-id\n")
    second = _write(tmp_path / "second.txt", "2\n3\n10\nshared-id\n")
    third = _write(tmp_path / "third.txt", "2\n4\n10\nshared-id\n")

    report = compare_id_files([first, second, third])

    assert report["intersection_count"] == 3
    assert report["non_matching_ids"] == ["1", "3", "4"]
    assert report["files"] == [
        {"path": str(first.resolve()), "id_count": 4, "missing_ids": ["3", "4"]},
        {"path": str(second.resolve()), "id_count": 4, "missing_ids": ["1", "4"]},
        {"path": str(third.resolve()), "id_count": 4, "missing_ids": ["1", "3"]},
    ]


def test_identical_files_have_no_non_matching_ids(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.txt", "001\nabc-2\n")
    second = _write(tmp_path / "second.txt", "abc-2\n001\n")

    report = compare_id_files([first, second])

    assert report["non_matching_count"] == 0
    assert report["non_matching_ids"] == []


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "empty"),
        ("1\n\n2\n", "Blank row"),
        ("1\n1\n", "Duplicate record ID"),
        ("1\n two\n", "whitespace"),
        ("1\n2,3\n", "Invalid record ID"),
    ],
)
def test_load_id_file_rejects_invalid_content(
    tmp_path: Path, content: str, message: str
) -> None:
    path = _write(tmp_path / "ids.txt", content)

    with pytest.raises(IdFileError, match=message):
        load_id_file(path)


def test_compare_requires_two_unique_txt_files(tmp_path: Path) -> None:
    path = _write(tmp_path / "ids.txt", "1\n")

    with pytest.raises(IdFileError, match="at least two"):
        compare_id_files([path])
    with pytest.raises(IdFileError, match="must be unique"):
        compare_id_files([path, path])


def test_load_id_file_rejects_wrong_extension_and_missing_path(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "ids.csv", "1\n")

    with pytest.raises(IdFileError, match=r"Expected a \.txt file"):
        load_id_file(csv_path)
    with pytest.raises(IdFileError, match="does not exist"):
        load_id_file(tmp_path / "missing.txt")
