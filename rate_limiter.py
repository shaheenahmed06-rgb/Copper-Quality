"""
rate_limiter.py
---------------
SQLite-based rate limiter. Tracks usage per session per day.
No external dependencies — uses Python's built-in sqlite3.
"""

import sqlite3
from datetime import date
from pathlib import Path


class RateLimiter:
    def __init__(self, db_path: str = "rate_limit.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the usage table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    session_id TEXT NOT NULL,
                    date       TEXT NOT NULL,
                    count      INTEGER DEFAULT 0,
                    PRIMARY KEY (session_id, date)
                )
            """)

    def get_usage(self, session_id: str) -> int:
        """Return how many analyses this session has run today."""
        today = date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT count FROM usage WHERE session_id = ? AND date = ?",
                (session_id, today),
            ).fetchone()
        return row[0] if row else 0

    def get_remaining(self, session_id: str, limit: int) -> int:
        """Return analyses remaining for today."""
        return max(0, limit - self.get_usage(session_id))

    def check_and_increment(self, session_id: str, limit: int) -> bool:
        """
        Attempt to consume one analysis slot.
        Returns True if allowed (and increments count), False if limit reached.
        """
        today = date.today().isoformat()
        if self.get_usage(session_id) >= limit:
            return False
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO usage (session_id, date, count) VALUES (?, ?, 1)
                ON CONFLICT(session_id, date) DO UPDATE SET count = count + 1
                """,
                (session_id, today),
            )
        return True
