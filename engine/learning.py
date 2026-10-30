"""Private, local-only request measurements and feedback-guided profiles."""

import os
import sqlite3
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


PROFILE_CAPS = {
    "full": 1.0,
    "balanced": 0.75,
    "efficient": 0.5,
}

_PROFILE_ORDER = ("full", "balanced", "efficient")


def _default_database_path() -> Path:
    configured_dir = os.environ.get("VOLTINFER_DATA_DIR")
    if configured_dir:
        data_dir = Path(configured_dir)
    elif sys.platform == "win32":
        data_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VoltInfer"
    elif sys.platform == "darwin":
        data_dir = Path.home() / "Library" / "Application Support" / "VoltInfer"
    else:
        data_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "voltinfer"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "learning.sqlite3"


class LearningStore:
    """Stores aggregate inference measurements and explicit user ratings, never prompts."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = Path(database_path) if database_path else _default_database_path()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    route TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    status INTEGER NOT NULL,
                    pressure_band TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    requested_max_tokens INTEGER,
                    applied_max_tokens INTEGER,
                    completion_tokens INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    request_id TEXT PRIMARY KEY REFERENCES requests(request_id),
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database_path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def pressure_band(pressure: float) -> str:
        if pressure >= 0.65:
            return "high"
        if pressure >= 0.35:
            return "medium"
        return "low"

    def choose_profile(
        self, model: str, pressure_band: str, requested_profile: Optional[str] = None
    ) -> Tuple[str, str, int]:
        if requested_profile is not None:
            if not isinstance(requested_profile, str) or requested_profile not in PROFILE_CAPS:
                raise ValueError("profile must be one of: full, balanced, efficient")
            return requested_profile, "requested", 0

        defaults = {"low": "full", "medium": "balanced", "high": "efficient"}
        fallback = defaults[pressure_band]
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT r.profile, COUNT(*) AS sample_count, AVG(f.rating) AS average_rating
                FROM requests AS r
                JOIN feedback AS f ON f.request_id = r.request_id
                WHERE r.model = ? AND r.pressure_band = ?
                  AND r.profile IN ('full', 'balanced', 'efficient')
                GROUP BY r.profile
                """,
                (model, pressure_band),
            ).fetchall()

        learned = {
            row["profile"]: (int(row["sample_count"]), float(row["average_rating"]))
            for row in rows
            if int(row["sample_count"]) >= 2
        }
        if len(learned) < 2:
            return fallback, "hardware_policy", sum(count for count, _ in learned.values())

        selected = max(
            _PROFILE_ORDER,
            key=lambda profile: (
                learned.get(profile, (0, 0.0))[1] if profile in learned else -1.0,
                -_PROFILE_ORDER.index(profile),
            ),
        )
        return selected, "local_feedback", sum(count for count, _ in learned.values())

    def record_request(
        self,
        request_id: str,
        model: str,
        route: str,
        latency_ms: float,
        status: int,
        pressure_band: str,
        profile: str,
        requested_max_tokens: Optional[int],
        applied_max_tokens: Optional[int],
        completion_tokens: Optional[int],
    ) -> None:
        safe_model = (model or "unknown")[:200]
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO requests
                (request_id, model, route, latency_ms, status, pressure_band, profile,
                 requested_max_tokens, applied_max_tokens, completion_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    safe_model,
                    route.split("?", 1)[0][:200],
                    max(0.0, latency_ms),
                    status,
                    pressure_band,
                    profile,
                    requested_max_tokens,
                    applied_max_tokens,
                    completion_tokens,
                ),
            )

    def add_feedback(self, request_id: str, rating: int) -> None:
        if not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 5:
            raise ValueError("rating must be an integer from 1 to 5")
        with self._lock, self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if exists is None:
                raise LookupError("request_id was not found")
            connection.execute(
                """
                INSERT INTO feedback (request_id, rating) VALUES (?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    rating = excluded.rating, created_at = CURRENT_TIMESTAMP
                """,
                (request_id, rating),
            )

    def get_status(self) -> Dict[str, Any]:
        with self._lock, self._connection() as connection:
            metrics = connection.execute(
                """
                SELECT COUNT(*) AS request_count, AVG(latency_ms) AS average_latency_ms,
                       SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) AS error_count
                FROM requests
                """
            ).fetchone()
            feedback = connection.execute(
                "SELECT COUNT(*) AS feedback_count, AVG(rating) AS average_rating FROM feedback"
            ).fetchone()
            profiles = connection.execute(
                """
                SELECT r.profile, COUNT(*) AS sample_count, AVG(f.rating) AS average_rating
                FROM requests AS r JOIN feedback AS f ON f.request_id = r.request_id
                WHERE r.profile IN ('full', 'balanced', 'efficient')
                GROUP BY r.profile
                """
            ).fetchall()
        return {
            "enabled": True,
            "request_count": int(metrics["request_count"] or 0),
            "average_latency_ms": round(metrics["average_latency_ms"], 2)
            if metrics["average_latency_ms"] is not None
            else None,
            "error_count": int(metrics["error_count"] or 0),
            "feedback_count": int(feedback["feedback_count"] or 0),
            "average_rating": round(feedback["average_rating"], 2)
            if feedback["average_rating"] is not None
            else None,
            "profile_feedback": {
                row["profile"]: {
                    "samples": int(row["sample_count"]),
                    "average_rating": round(float(row["average_rating"]), 2),
                }
                for row in profiles
            },
            "prompt_storage": False,
            "credential_storage": False,
        }
