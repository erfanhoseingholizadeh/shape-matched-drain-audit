# Tests of the number checker on a toy table and a toy recorded output.
# Run: python3 -m pytest tests/test_check_number_sources.py
import csv
import json

from paper_values import check_number_sources as cn

HEADER = ["id", "value_as_printed", "kind", "produced_by", "recorded_output", "key_in_output", "tolerance"]


def write_table(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def test_matching_mismatching_and_missing_rows(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "out.json").write_text(json.dumps({"a": {"b": [0.1234, 7]}, "c": 2.0}))
    write_table(tmp_path / "docs" / "number_sources.csv", [
        ["ok", "0.123", "verified", "s.py", "out.json", "a.b.0", "0.0005"],
        ["count", "7", "verified", "s.py", "out.json", "a.b.1", "0.5"],
        ["bad", "0.2", "verified", "s.py", "out.json", "c", "0.0005"],
        ["gone", "1", "verified", "s.py", "nope.json", "c", ""],
        ["const", "0.02", "handoff", "recorded design constant", "", "", ""],
    ])
    assert cn.main(["--root", str(tmp_path)]) == 1
    rows = list(csv.DictReader(open(tmp_path / "docs" / "number_sources.csv")))
    assert cn.check_row(rows[0], tmp_path) is None and cn.check_row(rows[1], tmp_path) is None
    assert "bad" in cn.check_row(rows[2], tmp_path) and "not found" in cn.check_row(rows[3], tmp_path)


def test_all_good_table_exits_zero(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "out.json").write_text(json.dumps({"x": "0.5"}))
    write_table(tmp_path / "docs" / "number_sources.csv", [["i", "0.50", "verified", "s.py", "out.json", "x", "0.001"]])
    assert cn.main(["--root", str(tmp_path)]) == 0
