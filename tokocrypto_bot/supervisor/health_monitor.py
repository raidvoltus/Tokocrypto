"""
MODULE: tokocrypto_bot.supervisor.health_monitor
DESCRIPTION: 3-Layer Health Evaluation (L1 Process, L2 Heartbeat, L3 Application State).
"""

import json
import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from tokocrypto_bot.persistence.database import DatabaseManager

logger = logging.getLogger("NVRA.Supervisor.HealthMonitor")


class WorkerHealthStatus(str, Enum):
    HEALTHY_TRADING = "HEALTHY_TRADING"
    HEALTHY_SAFE_MODE = "HEALTHY_SAFE_MODE"  # Heartbeat sehat tapi sengaja menolak order
    STARTING_GRACE_PERIOD = "STARTING_GRACE_PERIOD"
    UNHEALTHY_PROCESS_DEAD = "UNHEALTHY_PROCESS_DEAD"
    UNHEALTHY_STALE_HEARTBEAT = "UNHEALTHY_STALE_HEARTBEAT"
    UNHEALTHY_INVALID_STATE = "UNHEALTHY_INVALID_STATE"


@dataclass(frozen=True)
class HealthEvaluationResult:
    status: WorkerHealthStatus
    is_alive: bool
    app_state: str
    heartbeat_age_seconds: float
    message: str


class HealthMonitor:
    def __init__(
        self,
        db_mgr: DatabaseManager,
        heartbeat_timeout_sec: float = 30.0,
        startup_grace_sec: float = 60.0
    ):
        self.db = db_mgr
        self.heartbeat_timeout = heartbeat_timeout_sec
        self.startup_grace = startup_grace_sec

    def evaluate_worker_health(
        self,
        worker_process_alive: bool,
        worker_start_time: Optional[datetime] = None
    ) -> HealthEvaluationResult:
        now = datetime.now(timezone.utc)

        # L1 Check: Process Alive
        if not worker_process_alive:
            return HealthEvaluationResult(
                status=WorkerHealthStatus.UNHEALTHY_PROCESS_DEAD,
                is_alive=False,
                app_state="UNKNOWN",
                heartbeat_age_seconds=-1.0,
                message="L1 Failure: Worker process is not alive (OS Process Dead)."
            )

        # Startup Grace Period Handling
        if worker_start_time:
            elapsed = (now - worker_start_time).total_seconds()
            if elapsed < self.startup_grace:
                return HealthEvaluationResult(
                    status=WorkerHealthStatus.STARTING_GRACE_PERIOD,
                    is_alive=True,
                    app_state="STARTING",
                    heartbeat_age_seconds=elapsed,
                    message=f"L2 Grace Period: Worker in startup grace window ({elapsed:.1f}s / {self.startup_grace}s)."
                )

        # L2 & L3 Check: Load Heartbeat Contract & Application State from DB
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value, updated_at FROM bot_state WHERE key = 'heartbeat'")
            hb_row = cursor.fetchone()

            cursor.execute("SELECT value FROM bot_state WHERE key = 'application_state'")
            app_row = cursor.fetchone()
            app_state = app_row[0] if app_row else "UNKNOWN"

            if not hb_row:
                return HealthEvaluationResult(
                    status=WorkerHealthStatus.UNHEALTHY_STALE_HEARTBEAT,
                    is_alive=True,
                    app_state=app_state,
                    heartbeat_age_seconds=9999.0,
                    message="L2 Failure: No heartbeat entry found in database."
                )

            hb_data = json.loads(hb_row["value"])
            last_hb_str = hb_data.get("last_heartbeat", hb_row["updated_at"])
            last_hb_dt = datetime.fromisoformat(last_hb_str)
            if last_hb_dt.tzinfo is None:
                last_hb_dt = last_hb_dt.replace(tzinfo=timezone.utc)

            age_sec = (now - last_hb_dt).total_seconds()

            # Stale Heartbeat Check
            if age_sec > self.heartbeat_timeout:
                return HealthEvaluationResult(
                    status=WorkerHealthStatus.UNHEALTHY_STALE_HEARTBEAT,
                    is_alive=True,
                    app_state=app_state,
                    heartbeat_age_seconds=age_sec,
                    message=f"L2 Failure: Heartbeat stale ({age_sec:.1f}s > threshold {self.heartbeat_timeout}s)."
                )

            # L3 Application State Rules
            # RULE: SAFE_MODE dengan Heartbeat Sehat = HEALTHY (NO RESTART!)
            if app_state == "SAFE_MODE":
                return HealthEvaluationResult(
                    status=WorkerHealthStatus.HEALTHY_SAFE_MODE,
                    is_alive=True,
                    app_state=app_state,
                    heartbeat_age_seconds=age_sec,
                    message="L3 Evaluation: Worker in SAFE_MODE with active heartbeat. Process is healthy (No restart required)."
                )

            return HealthEvaluationResult(
                status=WorkerHealthStatus.HEALTHY_TRADING,
                is_alive=True,
                app_state=app_state,
                heartbeat_age_seconds=age_sec,
                message=f"L3 Evaluation: Worker HEALTHY in state [{app_state}]."
            )

        except Exception as e:
            logger.error(f"Error checking heartbeat/state from database: {e}")
            return HealthEvaluationResult(
                status=WorkerHealthStatus.UNHEALTHY_INVALID_STATE,
                is_alive=True,
                app_state="ERROR",
                heartbeat_age_seconds=-1.0,
                message=f"L3 Failure: Exception evaluating DB state ({e})"
            )
        finally:
            conn.close()
