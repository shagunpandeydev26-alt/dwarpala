"""
Optional SQLite audit log for the Dwarpala API.

Enabled only when settings.audit_log == "sqlite" (env AUDIT_LOG=sqlite). Each
verification/match/liveness result is appended as one row. NO image bytes are
ever stored — only request_id, timestamp, endpoint, verdict, scores, latency.

Schema (table ``audit``):
    id            INTEGER PRIMARY KEY AUTOINCREMENT
    request_id    TEXT      -- uuid4 of the request
    ts            TEXT      -- ISO-8601 UTC timestamp
    endpoint      TEXT      -- "verify" | "match" | "liveness"
    verdict       TEXT      -- e.g. ACCEPT/REJECT/MANUAL_REVIEW/LIVE/SPOOF/MATCH
    match_score   REAL      -- nullable
    liveness_score REAL     -- nullable
    latency_ms    REAL
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dwarpala.utils.logger import get_logger

logger = get_logger("api.audit")


class AuditLog:
    """Thread-safe, append-only SQLite audit sink (no image data)."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        # check_same_thread=False: we guard all access with our own lock.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                ts TEXT,
                endpoint TEXT,
                verdict TEXT,
                match_score REAL,
                liveness_score REAL,
                latency_ms REAL
            )
            """)
        self._conn.commit()
        logger.info(f"Audit log enabled (sqlite): {self.db_path}")

    def record(
        self,
        request_id: str,
        endpoint: str,
        verdict: str,
        latency_ms: float,
        match_score: Optional[float] = None,
        liveness_score: Optional[float] = None,
    ) -> None:
        """Append one audit row. Never raises into the request path."""
        try:
            ts = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._conn.execute(
                    "INSERT INTO audit (request_id, ts, endpoint, verdict, "
                    "match_score, liveness_score, latency_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (request_id, ts, endpoint, verdict, match_score, liveness_score, latency_ms),
                )
                self._conn.commit()
        except Exception as e:  # pragma: no cover - audit must never break a request
            logger.warning(f"Audit write failed (request_id={request_id}): {e}")

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:  # pragma: no cover
            pass


def maybe_create_audit(settings) -> Optional[AuditLog]:
    """Return an AuditLog if audit is enabled in settings, else None."""
    if not getattr(settings, "audit_enabled", False):
        return None
    Path(settings.audit_db).parent.mkdir(parents=True, exist_ok=True)
    return AuditLog(settings.audit_db)
