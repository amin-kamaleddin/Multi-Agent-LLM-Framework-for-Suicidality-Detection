import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class AuditResult:
    run_dir: Path
    rows_csv: int
    sample_id_missing: int
    sample_id_unique: int
    sample_id_duplicates: list[tuple[str, int]]
    sample_id_min: int | None
    sample_id_max: int | None
    sample_id_gaps: list[int]
    duplicate_text_groups: int
    duplicate_text_total_rows_in_groups: int
    duplicate_text_disagree_ground_truth_groups: int
    duplicate_text_disagree_prediction_groups: int
    duplicate_text_examples: list[dict[str, Any]]
    duplicate_row_groups: int
    duplicate_row_total_rows_in_groups: int
    ground_truth_dist: list[tuple[str, int]]
    prediction_dist: list[tuple[str, int]]
    prediction_missing: int
    correct_missing: int
    incorrect_count: int
    pred_confidence_dist: list[tuple[str, int]]
    class_scores_empty: int
    error_nonempty: int
    failed_cases_lines: int | None
    meta_samples: int | None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(s)
    except Exception:
        return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _read_meta_samples(meta_path: Path) -> int | None:
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Try a few common locations without assuming schema
    for key in ("n_samples", "samples", "num_samples"):
        if key in data and isinstance(data[key], int):
            return data[key]

    # Some Meta.json may nest fields
    for container_key in ("config", "run", "meta"):
        container = data.get(container_key)
        if isinstance(container, dict):
            for key in ("n_samples", "samples", "num_samples"):
                val = container.get(key)
                if isinstance(val, int):
                    return val

    return None


def audit_run(run_dir: Path) -> AuditResult:
    csv_path = run_dir / "all_cases.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows: list[dict[str, str]] = []
    sample_ids: list[str | None] = []
    pred_conf: list[str] = []
    class_scores_empty = 0
    error_nonempty = 0
    prediction_missing = 0
    correct_missing = 0
    incorrect_count = 0
    gt_vals: list[str] = []
    pred_vals: list[str] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            sid = row.get("sample_id")
            sample_ids.append(sid)

            gt = (row.get("ground_truth") or "").strip()
            pr = (row.get("prediction") or "").strip()
            gt_vals.append(gt)
            pred_vals.append(pr)

            if pr == "":
                prediction_missing += 1
            if (row.get("correct") or "").strip() == "":
                correct_missing += 1
            else:
                if (row.get("correct") or "").strip() == "0":
                    incorrect_count += 1

            conf = row.get("prediction_confidence")
            if conf is not None:
                pred_conf.append(conf.strip())

            if (row.get("class_scores") or "").strip() == "":
                class_scores_empty += 1

            if (row.get("error") or "").strip() != "":
                error_nonempty += 1

    missing = sum(1 for sid in sample_ids if sid is None or str(sid).strip() == "")

    ctr = Counter(str(sid).strip() if sid is not None else None for sid in sample_ids)
    dup_ids = [(k, v) for k, v in ctr.items() if k not in (None, "") and v > 1]
    dup_ids.sort(key=lambda kv: (-kv[1], kv[0]))

    numeric_ids = [_safe_int(sid) for sid in sample_ids]
    numeric_ids_clean = [x for x in numeric_ids if x is not None]
    min_id = min(numeric_ids_clean) if numeric_ids_clean else None
    max_id = max(numeric_ids_clean) if numeric_ids_clean else None

    gaps: list[int] = []
    if min_id is not None and max_id is not None:
        present = set(numeric_ids_clean)
        for i in range(min_id, max_id + 1):
            if i not in present:
                gaps.append(i)

    # Duplicate TEXT (hash full field; normalize whitespace minimally)
    text_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    text_hash_to_gts: dict[str, set[str]] = defaultdict(set)
    text_hash_to_preds: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        text = (row.get("text") or "")
        # Keep full text, but normalize CRLF/leading/trailing whitespace to avoid false negatives
        norm = text.replace("\r\n", "\n").strip()
        h = _sha256_text(norm)
        text_hash_to_ids[h].append(str(row.get("sample_id")))
        text_hash_to_gts[h].add((row.get("ground_truth") or "").strip())
        text_hash_to_preds[h].add((row.get("prediction") or "").strip())

    dup_text_groups = 0
    dup_text_total = 0
    dup_text_disagree_gt = 0
    dup_text_disagree_pred = 0
    dup_text_examples: list[dict[str, Any]] = []
    for ids in text_hash_to_ids.values():
        if len(ids) > 1:
            dup_text_groups += 1
            dup_text_total += len(ids)

    # Compute disagreement + capture a few examples
    for h, ids in text_hash_to_ids.items():
        if len(ids) <= 1:
            continue
        gts = sorted(text_hash_to_gts[h])
        preds = sorted(text_hash_to_preds[h])
        if len(gts) > 1:
            dup_text_disagree_gt += 1
        if len(preds) > 1:
            dup_text_disagree_pred += 1
        if len(dup_text_examples) < 10:
            dup_text_examples.append(
                {
                    "n": len(ids),
                    "sample_ids": ids[:20],
                    "ground_truths": gts,
                    "predictions": preds,
                }
            )

    # Duplicate full rows ignoring timing/token fields (focus on logical content)
    row_key_to_ids: dict[str, list[str]] = defaultdict(list)
    key_fields = ("ground_truth", "prediction", "correct", "prediction_confidence", "class_scores", "error", "text")
    for row in rows:
        payload = {k: (row.get(k) or "") for k in key_fields}
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        h = hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()
        row_key_to_ids[h].append(str(row.get("sample_id")))

    dup_row_groups = 0
    dup_row_total = 0
    for ids in row_key_to_ids.values():
        if len(ids) > 1:
            dup_row_groups += 1
            dup_row_total += len(ids)

    pred_conf_ctr = Counter(pred_conf)
    pred_conf_dist = sorted(pred_conf_ctr.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    gt_dist = sorted(Counter(gt_vals).items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    pr_dist = sorted(Counter(pred_vals).items(), key=lambda kv: (-kv[1], kv[0]))[:20]

    failed_cases_path = run_dir / "failed_cases.txt"
    failed_lines: int | None = None
    if failed_cases_path.exists():
        try:
            failed_lines = sum(1 for _ in failed_cases_path.open("r", encoding="utf-8", errors="replace"))
        except Exception:
            failed_lines = None

    meta_samples = _read_meta_samples(run_dir / "Meta.json")

    unique_ids = len([k for k in ctr.keys() if k not in (None, "")])

    return AuditResult(
        run_dir=run_dir,
        rows_csv=len(rows),
        sample_id_missing=missing,
        sample_id_unique=unique_ids,
        sample_id_duplicates=dup_ids,
        sample_id_min=min_id,
        sample_id_max=max_id,
        sample_id_gaps=gaps,
        duplicate_text_groups=dup_text_groups,
        duplicate_text_total_rows_in_groups=dup_text_total,
        duplicate_text_disagree_ground_truth_groups=dup_text_disagree_gt,
        duplicate_text_disagree_prediction_groups=dup_text_disagree_pred,
        duplicate_text_examples=dup_text_examples,
        duplicate_row_groups=dup_row_groups,
        duplicate_row_total_rows_in_groups=dup_row_total,
        ground_truth_dist=gt_dist,
        prediction_dist=pr_dist,
        prediction_missing=prediction_missing,
        correct_missing=correct_missing,
        incorrect_count=incorrect_count,
        pred_confidence_dist=pred_conf_dist,
        class_scores_empty=class_scores_empty,
        error_nonempty=error_nonempty,
        failed_cases_lines=failed_lines,
        meta_samples=meta_samples,
    )


def _print_result(res: AuditResult) -> None:
    print(f"run_dir={res.run_dir}")
    print(f"rows_csv={res.rows_csv}")
    if res.meta_samples is not None:
        print(f"meta_samples={res.meta_samples}")
    print(f"sample_id: unique={res.sample_id_unique} missing={res.sample_id_missing} duplicates={len(res.sample_id_duplicates)}")
    if res.sample_id_min is not None and res.sample_id_max is not None:
        print(f"sample_id_range={res.sample_id_min}..{res.sample_id_max} gaps={len(res.sample_id_gaps)}")
    if res.sample_id_duplicates:
        print("dup_sample_id_examples=" + json.dumps(res.sample_id_duplicates[:20]))

    print(f"duplicate_text: groups={res.duplicate_text_groups} total_rows_in_groups={res.duplicate_text_total_rows_in_groups}")
    print(
        "duplicate_text_disagreement: ground_truth_groups={}".format(
            res.duplicate_text_disagree_ground_truth_groups
        )
        + " prediction_groups={}".format(res.duplicate_text_disagree_prediction_groups)
    )
    if res.duplicate_text_examples:
        print("duplicate_text_examples=" + json.dumps(res.duplicate_text_examples, ensure_ascii=False))
    print(f"duplicate_rows(logical): groups={res.duplicate_row_groups} total_rows_in_groups={res.duplicate_row_total_rows_in_groups}")

    print(f"ground_truth top={json.dumps(res.ground_truth_dist, ensure_ascii=False)}")
    print(f"prediction top={json.dumps(res.prediction_dist, ensure_ascii=False)}")
    print(f"prediction_missing_rows={res.prediction_missing}")
    print(f"correct_missing_rows={res.correct_missing}")
    print(f"incorrect_rows(correct==0)={res.incorrect_count}")

    print(f"prediction_confidence top={json.dumps(res.pred_confidence_dist)}")
    print(f"class_scores_empty_rows={res.class_scores_empty}")
    print(f"error_nonempty_rows={res.error_nonempty}")

    if res.failed_cases_lines is not None:
        print(f"failed_cases_txt_lines={res.failed_cases_lines}")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(list(argv) if argv is not None else None)

    res = audit_run(args.run_dir)
    _print_result(res)

    # Non-zero exit for obvious integrity issues
    problems: list[str] = []
    if res.sample_id_missing != 0:
        problems.append("missing_sample_id")
    if res.sample_id_duplicates:
        problems.append("duplicate_sample_id")
    if res.sample_id_gaps:
        problems.append("gapped_sample_id")
    if res.rows_csv != res.sample_id_unique and res.sample_id_missing == 0:
        # If missing ids exist, this check isn't meaningful.
        problems.append("rows_not_equal_unique_ids")

    if problems:
        print("problems=" + ",".join(problems))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
