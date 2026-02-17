"""SQLite-based storage for evaluation results."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from llm_eval_suite.exceptions import StorageError
from llm_eval_suite.models import CriterionScore, EvalResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

CREATE_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL,
    rubric_name TEXT NOT NULL,
    overall_score REAL NOT NULL,
    criterion_scores TEXT NOT NULL,
    judge_model TEXT DEFAULT '',
    timestamp TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEX_SAMPLE = """
CREATE INDEX IF NOT EXISTS idx_sample_id ON eval_results(sample_id);
"""

CREATE_INDEX_RUBRIC = """
CREATE INDEX IF NOT EXISTS idx_rubric_name ON eval_results(rubric_name);
"""


class EvalStorage:
    """SQLite-backed storage for evaluation results.

    Provides persistent storage with query capabilities for tracking
    evaluation results over time.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize storage with SQLite database.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create database tables if they don't exist."""
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            cursor = self._conn.cursor()
            cursor.execute(CREATE_RESULTS_TABLE)
            cursor.execute(CREATE_INDEX_SAMPLE)
            cursor.execute(CREATE_INDEX_RUBRIC)
            self._conn.commit()
            logger.info(f"Initialized storage at {self.db_path}")
        except sqlite3.Error as e:
            raise StorageError(f"Failed to initialize database: {e}") from e

    def _get_conn(self) -> sqlite3.Connection:
        """Get the database connection.

        Returns:
            Active SQLite connection

        Raises:
            StorageError: If connection is closed
        """
        if self._conn is None:
            raise StorageError("Database connection is closed")
        return self._conn

    def save_result(self, result: EvalResult) -> int:
        """Save an evaluation result.

        Args:
            result: EvalResult to persist

        Returns:
            Row ID of the inserted record

        Raises:
            StorageError: If insert fails
        """
        conn = self._get_conn()
        criterion_scores_json = json.dumps([
            {
                "criterion_name": cs.criterion_name,
                "score": cs.score,
                "reasoning": cs.reasoning,
            }
            for cs in result.criterion_scores
        ])

        try:
            cursor = conn.execute(
                """INSERT INTO eval_results
                   (sample_id, rubric_name, overall_score, criterion_scores,
                    judge_model, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.sample_id,
                    result.rubric_name,
                    result.overall_score,
                    criterion_scores_json,
                    result.judge_model,
                    result.timestamp,
                    json.dumps(result.metadata),
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid or 0
            logger.debug(f"Saved result for sample {result.sample_id}, row_id={row_id}")
            return row_id
        except sqlite3.Error as e:
            raise StorageError(f"Failed to save result: {e}") from e

    def get_results_by_rubric(self, rubric_name: str) -> List[EvalResult]:
        """Get all results for a specific rubric.

        Args:
            rubric_name: Name of the rubric

        Returns:
            List of EvalResult objects
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM eval_results WHERE rubric_name = ? ORDER BY created_at DESC",
            (rubric_name,),
        )
        return [self._row_to_result(row) for row in cursor.fetchall()]

    def get_results_by_sample(self, sample_id: str) -> List[EvalResult]:
        """Get all results for a specific sample.

        Args:
            sample_id: Sample identifier

        Returns:
            List of EvalResult objects
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM eval_results WHERE sample_id = ? ORDER BY created_at DESC",
            (sample_id,),
        )
        return [self._row_to_result(row) for row in cursor.fetchall()]

    def get_all_results(self) -> List[EvalResult]:
        """Get all stored evaluation results.

        Returns:
            List of all EvalResult objects
        """
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM eval_results ORDER BY created_at DESC")
        return [self._row_to_result(row) for row in cursor.fetchall()]

    def count_results(self) -> int:
        """Count total stored results.

        Returns:
            Number of stored results
        """
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM eval_results")
        return cursor.fetchone()[0]

    def _row_to_result(self, row: sqlite3.Row) -> EvalResult:
        """Convert a database row to an EvalResult.

        Args:
            row: SQLite Row object

        Returns:
            EvalResult reconstructed from stored data
        """
        scores_data = json.loads(row["criterion_scores"])
        criterion_scores = [
            CriterionScore(
                criterion_name=s["criterion_name"],
                score=s["score"],
                reasoning=s.get("reasoning", ""),
            )
            for s in scores_data
        ]

        return EvalResult(
            sample_id=row["sample_id"],
            rubric_name=row["rubric_name"],
            criterion_scores=criterion_scores,
            overall_score=row["overall_score"],
            judge_model=row["judge_model"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Storage connection closed")
