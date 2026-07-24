"""Crash-safe per-run progress tracking for concurrent evaluations."""

from __future__ import annotations

from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import pandas as pd


MANIFEST_SCHEMA_VERSION = "1.0"
PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dataframe_sha256(df: pd.DataFrame) -> str:
    """Return an order-sensitive fingerprint of the evaluated dataframe."""
    payload = df.to_json(orient="records", date_format="iso", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_sha256(payload: Any) -> str:
    """Return a stable fingerprint for a JSON-compatible value."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically replace the human-readable run manifest."""
    manifest_path = Path(path)
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


class ProgressStore:
    """SQLite-backed source of truth for one evaluation run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                sample_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed')),
                attempt_count INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                agent_execution_duration_s REAL NOT NULL DEFAULT 0,
                failed_llm_attempts INTEGER NOT NULL DEFAULT 0,
                failed_record INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                first_call_started_at_s REAL,
                last_call_finished_at_s REAL,
                real_execution_duration_s REAL NOT NULL DEFAULT 0
            )
            """
        )
        self.connection.commit()

    def initialize(self, sample_ids: Iterable[str]) -> None:
        expected_ids = [str(sample_id) for sample_id in sample_ids]
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("Evaluation sample IDs must be unique")

        existing_ids = {
            row[0]
            for row in self.connection.execute("SELECT sample_id FROM cases")
        }
        if existing_ids:
            if existing_ids != set(expected_ids):
                missing = sorted(set(expected_ids) - existing_ids)
                unexpected = sorted(existing_ids - set(expected_ids))
                raise ValueError(
                    "Progress database IDs do not match the evaluation dataset: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            return

        timestamp = utc_now()
        self.connection.executemany(
            "INSERT INTO cases(sample_id, status, updated_at) VALUES (?, 'pending', ?)",
            [(sample_id, timestamp) for sample_id in expected_ids],
        )
        self.connection.commit()

    def reset_running(self) -> int:
        cursor = self.connection.execute(
            "UPDATE cases SET status='pending', updated_at=? WHERE status='running'",
            (utc_now(),),
        )
        self.connection.commit()
        return int(cursor.rowcount)

    def statuses(self) -> dict[str, str]:
        return {
            sample_id: status
            for sample_id, status in self.connection.execute(
                "SELECT sample_id, status FROM cases"
            )
        }

    def claim(self, sample_id: str) -> None:
        cursor = self.connection.execute(
            """
            UPDATE cases
            SET status='running', attempt_count=attempt_count + 1, updated_at=?
            WHERE sample_id=? AND status='pending'
            """,
            (utc_now(), str(sample_id)),
        )
        self.connection.commit()
        if cursor.rowcount != 1:
            status_row = self.connection.execute(
                "SELECT status FROM cases WHERE sample_id=?",
                (str(sample_id),),
            ).fetchone()
            status = status_row[0] if status_row else "missing"
            raise RuntimeError(
                f"Cannot claim sample_id={sample_id!r}; current status={status!r}"
            )

    def release(self, sample_id: str) -> None:
        cursor = self.connection.execute(
            """
            UPDATE cases
            SET status='pending', updated_at=?
            WHERE sample_id=? AND status='running'
            """,
            (utc_now(), str(sample_id)),
        )
        self.connection.commit()
        if cursor.rowcount != 1:
            raise RuntimeError(f"Cannot release sample_id={sample_id!r}")

    def has_execution_sessions(self) -> bool:
        row = self.connection.execute(
            "SELECT EXISTS(SELECT 1 FROM execution_sessions)"
        ).fetchone()
        return bool(row[0])

    def add_legacy_execution_session(self, duration_s: float) -> None:
        """Seed an old run's trace-derived duration before its first resume."""
        if self.has_execution_sessions():
            return
        self.connection.execute(
            """
            INSERT INTO execution_sessions(started_at, real_execution_duration_s)
            VALUES (?, ?)
            """,
            (utc_now(), max(0.0, float(duration_s))),
        )
        self.connection.commit()

    def start_execution_session(self) -> int:
        """Create one active evaluator invocation for concurrency-aware timing."""
        cursor = self.connection.execute(
            "INSERT INTO execution_sessions(started_at) VALUES (?)",
            (utc_now(),),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def complete(
        self,
        sample_id: str,
        *,
        payload: dict[str, Any],
        agent_execution_duration_s: float,
        execution_session_id: int,
        call_started_at_s: float,
        call_finished_at_s: float,
        failed_llm_attempts: int,
        failed_record: bool,
    ) -> None:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE cases
                SET status='completed',
                    result_json=?,
                    agent_execution_duration_s=?,
                    failed_llm_attempts=?,
                    failed_record=?,
                    updated_at=?
                WHERE sample_id=? AND status='running'
                """,
                (
                    json.dumps(payload, ensure_ascii=False),
                    float(agent_execution_duration_s),
                    int(failed_llm_attempts),
                    int(failed_record),
                    utc_now(),
                    str(sample_id),
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Cannot commit duplicate or unscheduled result for sample_id={sample_id!r}"
                )

            session_row = self.connection.execute(
                """
                SELECT first_call_started_at_s, last_call_finished_at_s
                FROM execution_sessions
                WHERE session_id=?
                """,
                (int(execution_session_id),),
            ).fetchone()
            if session_row is None:
                raise RuntimeError(
                    f"Missing execution session_id={execution_session_id!r}"
                )
            first_started = (
                float(call_started_at_s)
                if session_row[0] is None
                else min(float(session_row[0]), float(call_started_at_s))
            )
            last_finished = (
                float(call_finished_at_s)
                if session_row[1] is None
                else max(float(session_row[1]), float(call_finished_at_s))
            )
            self.connection.execute(
                """
                UPDATE execution_sessions
                SET first_call_started_at_s=?,
                    last_call_finished_at_s=?,
                    real_execution_duration_s=?
                WHERE session_id=?
                """,
                (
                    first_started,
                    last_finished,
                    max(0.0, last_finished - first_started),
                    int(execution_session_id),
                ),
            )

    def completed_payloads(self) -> dict[str, dict[str, Any]]:
        return {
            sample_id: json.loads(result_json)
            for sample_id, result_json in self.connection.execute(
                """
                SELECT sample_id, result_json
                FROM cases
                WHERE status='completed' AND result_json IS NOT NULL
                """
            )
        }

    def summary(self) -> dict[str, Any]:
        status_counts = {
            status: int(count)
            for status, count in self.connection.execute(
                "SELECT status, COUNT(*) FROM cases GROUP BY status"
            )
        }
        totals = self.connection.execute(
            """
            SELECT
                COALESCE(SUM(agent_execution_duration_s), 0),
                COALESCE(SUM(failed_llm_attempts), 0),
                COALESCE(SUM(failed_record), 0)
            FROM cases
            WHERE status='completed'
            """
        ).fetchone()
        real_execution_duration_s = self.connection.execute(
            """
            SELECT COALESCE(SUM(real_execution_duration_s), 0)
            FROM execution_sessions
            """
        ).fetchone()[0]
        return {
            "pending_records": status_counts.get(PENDING, 0),
            "running_records": status_counts.get(RUNNING, 0),
            "completed_records": status_counts.get(COMPLETED, 0),
            "agent_execution_duration_s": float(totals[0]),
            "real_execution_duration_s": float(real_execution_duration_s),
            "failed_llm_attempts": int(totals[1]),
            "failed_records": int(totals[2]),
        }

    def coverage(
        self,
        expected_ids: Iterable[str],
        *,
        output_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        expected = [str(sample_id) for sample_id in expected_ids]
        actual = (
            [str(sample_id) for sample_id in output_ids]
            if output_ids is not None
            else [
                row[0]
                for row in self.connection.execute(
                    """
                    SELECT sample_id
                    FROM cases
                    WHERE status='completed' AND result_json IS NOT NULL
                    """
                )
            ]
        )
        expected_counts = Counter(expected)
        actual_counts = Counter(actual)
        missing_ids = [sample_id for sample_id in expected if sample_id not in actual_counts]
        unexpected_ids = sorted(set(actual_counts) - set(expected_counts))
        duplicate_ids = sorted(
            sample_id
            for sample_id, count in actual_counts.items()
            if count != 1
        )
        return {
            "expected_records": len(expected),
            "output_records": len(actual),
            "unique_output_ids": len(actual_counts),
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "duplicate_ids": duplicate_ids,
            "complete": expected_counts == actual_counts,
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ProgressStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
