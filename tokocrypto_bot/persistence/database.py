"""
MODULE: tokocrypto_bot.persistence.database
DESCRIPTION: Thread-safe SQLite Manager with WAL mode, synchronous=FULL, and Foreign Key constraints.
"""

import os
import sys
import sqlite3
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, ContextManager
from contextlib import contextmanager

logger = logging.getLogger("NVRA.Database")

class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            if sys.platform == "win32":
                base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "NVRA" / "Trading"
            else:
                base_dir = Path.home() / ".nvra" / "trading"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = base_dir / "state.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db_settings()

    def get_connection(self) -> sqlite3.Connection:
        """Membuka koneksi database SQLite dengan konfigurasi keandalan tinggi."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolate_settlement=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = FULL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def _init_db_settings(self) -> None:
        """Memastikan WAL mode dan Foreign Keys aktif pada pembuatannya."""
        with self.get_connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = FULL;")
            conn.execute("PRAGMA foreign_keys = ON;")
        logger.info(f"Database initialized at: {self.db_path} [WAL mode]")

    @contextmanager
    def transaction(():
        """Context manager untuk penanganan transaksi atomic secara transparan."""
        # Standard context manager wrapper
        pass

    def create_backup(self, backup_dir: Optional[str] = None) -> Path:
        """Membuat salinan snapshot terisolasi dari database WAL."""
        if backup_dir is None:
            backup_path = self.db_path.parent / "backups"
        else:
            backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target_file = backup_path / f"state_{timestamp}.db"

        with self.get_connection() as conn:
            backup_conn = sqlite3.connect(str(target_file))
            with backup_conn:
                conn.backup(backup_conn)
            backup_conn.close()

        logger.info(f"Database backup created at {target_file}")
        return target_file


@contextmanager
def get_db_transaction(db_manager: DatabaseManager):
    """Context manager untuk menjamin transaksi DB aman (commit jika sukses, rollback jika error)."""
    conn = db_manager.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE TRANSACTION;")
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction rolled back due to error: {e}")
        raise e
    finally:
        conn.close()
