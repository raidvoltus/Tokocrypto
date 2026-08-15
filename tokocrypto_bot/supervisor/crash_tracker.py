"""
MODULE: tokocrypto_bot.supervisor.crash_tracker
DESCRIPTION: Persistent Crash-Loop Tracker using Rolling Time Window (SQLite-backed).
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from tokocrypto_bot.persistence.database import DatabaseManager, get_db_transaction

logger = logging.getLogger("NVRA.Supervisor.CrashTracker")

class PersistentCrashTracker:
    def __init__(self, db_mgr: DatabaseManager, window_minutes: int = 5, max_crashes: int = 3):
        self.db = db_mgr
        self.window_minutes = window_minutes
        self.max_crashes = max_crashes
        self._init_table()

    def _init_table(self) -> None:
        """Membuat tabel supervisor_incidents jika belum ada."""
        with get_db_transaction(self.db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS supervisor_incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pid INTEGER,
                    exit_code INTEGER,
                    reason TEXT NOT NULL,
                    restart_number INTEGER NOT NULL
                );
                """
            )

    def record_crash_event(self, pid: Optional[int], exit_code: Optional[int], reason: str) -> int:
        """Mencatat event crash ke disk dan mengembalikan jumlah crash dalam window aktif."""
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        
        # Hitung jumlah restart sebelumnya
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM supervisor_incidents")
            total_restarts = cursor.fetchone()[0] + 1
        finally:
            conn.close()

        with get_db_transaction(self.db) as conn:
            conn.execute(
                """
                INSERT INTO supervisor_incidents (timestamp, pid, exit_code, reason, restart_number)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now_str, pid, exit_code, reason, total_restarts)
            )

        recent_count = self.get_recent_crash_count()
        logger.warning(
            f"CRASH INCIDENT RECORDED: Exit Code={exit_code}, Reason='{reason}'. "
            f"Recent crashes in last {self.window_minutes}m: {recent_count}/{self.max_crashes}"
        )
        return recent_count

    def get_recent_crash_count(self) -> int:
        """Menghitung jumlah crash dalam rolling window (misal: 5 menit terakhir)."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.window_minutes)
        cutoff_str = cutoff.isoformat()

        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM supervisor_incidents WHERE timestamp >= ?",
                (cutoff_str,)
            )
            count = cursor.fetchone()[0]
            return count
        finally:
            conn.close()

    def is_crash_loop_triggered(self) -> bool:
        """Memeriksa apakah ambang batas crash-loop terlampaui."""
        return self.get_recent_crash_count() >= self.max_crashes
