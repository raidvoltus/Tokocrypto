"""
MODULE: tokocrypto_bot.supervisor.supervisor
DESCRIPTION: Decoupled Supervisor Daemon and Process Controller for NVRA Trading Worker.
"""

import sys
import time
import subprocess
import logging
from enum import Enum
from datetime import datetime, timezone
from typing import Optional, List

from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.supervisor.crash_tracker import PersistentCrashTracker
from tokocrypto_bot.supervisor.health_monitor import HealthMonitor, WorkerHealthStatus
from tokocrypto_bot.supervisor.restart_policy import RestartPolicy, WorkerExitCategory

logger = logging.getLogger("NVRA.Supervisor")


class SupervisorState(str, Enum):
    INITIALIZING = "INITIALIZING"
    STARTING_WORKER = "STARTING_WORKER"
    MONITORING = "MONITORING"
    WORKER_UNHEALTHY = "WORKER_UNHEALTHY"
    RESTARTING = "RESTARTING"
    CRASH_LOOP = "CRASH_LOOP"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class NVRASupervisor:
    def __init__(
        self,
        db_mgr: DatabaseManager,
        worker_cmd: List[str],
        heartbeat_timeout_sec: float = 30.0,
        startup_grace_sec: float = 60.0,
        window_minutes: int = 5,
        max_crashes: int = 3
    ):
        self.db_mgr = db_mgr
        self.worker_cmd = worker_cmd
        self.crash_tracker = PersistentCrashTracker(db_mgr, window_minutes, max_crashes)
        self.health_monitor = HealthMonitor(db_mgr, heartbeat_timeout_sec, startup_grace_sec)

        self._state = SupervisorState.INITIALIZING
        self._worker_process: Optional[subprocess.Popen] = None
        self._worker_start_time: Optional[datetime] = None
        self._manual_stop_requested = False

    @property
    def state(self) -> SupervisorState:
        return self._state

    def start_worker(self) -> bool:
        """Memulai Trading Worker via Subprocess."""
        self._state = SupervisorState.STARTING_WORKER
        logger.info(f"Spawning Trading Worker process: {self.worker_cmd}")

        try:
            self._worker_process = subprocess.Popen(
                self.worker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self._worker_start_time = datetime.now(timezone.utc)
            logger.info(f"Trading Worker spawned successfully (PID: {self._worker_process.pid})")
            self._state = SupervisorState.MONITORING
            return True
        except Exception as e:
            logger.critical(f"Failed to spawn Trading Worker process: {e}")
            self.crash_tracker.record_crash_event(None, -1, f"Spawn failure: {e}")
            if self.crash_tracker.is_crash_loop_triggered():
                self._state = SupervisorState.CRASH_LOOP
            return False

    def monitor_tick(self) -> SupervisorState:
        """Satu iterasi pemonitoran kesehatan worker (Tick loop)."""
        if self._state in (SupervisorState.STOPPED, SupervisorState.CRASH_LOOP):
            return self._state

        is_alive = self._worker_process is not None and self._worker_process.poll() is None
        health = self.health_monitor.evaluate_worker_health(is_alive, self._worker_start_time)

        # 1. Healthy / Safe Mode / Grace Period -> Continue Monitoring
        if health.status in (
            WorkerHealthStatus.HEALTHY_TRADING,
            WorkerHealthStatus.HEALTHY_SAFE_MODE,
            WorkerHealthStatus.STARTING_GRACE_PERIOD
        ):
            self._state = SupervisorState.MONITORING
            return self._state

        # 2. Unhealthy -> Handle Restart / Crash
        logger.warning(f"WORKER UNHEALTHY DETECTED: {health.message}")
        self._state = SupervisorState.WORKER_UNHEALTHY

        # Kill process if alive but heartbeat stale
        exit_code = None
        if is_alive and health.status == WorkerHealthStatus.UNHEALTHY_STALE_HEARTBEAT:
            logger.warning(f"Terminating unresponsive worker (PID: {self._worker_process.pid}) due to stale heartbeat...")
            self._worker_process.terminate()
            try:
                self._worker_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._worker_process.kill()
            exit_code = -9
        elif not is_alive:
            exit_code = self._worker_process.poll() if self._worker_process else -1

        exit_cat = RestartPolicy.classify_exit_code(exit_code)
        crash_count = self.crash_tracker.record_crash_event(
            pid=self._worker_process.pid if self._worker_process else None,
            exit_code=exit_code,
            reason=health.message
        )

        crash_loop = self.crash_tracker.is_crash_loop_triggered()
        decision = RestartPolicy.evaluate_restart(exit_cat, crash_loop, self._manual_stop_requested)

        if decision.should_restart:
            logger.info(f"Restart Decision: RESTART WORKER ({decision.reason})")
            self._state = SupervisorState.RESTARTING
            time.sleep(2.0)  # Brief backoff before restart
            self.start_worker()
        else:
            if crash_loop:
                logger.critical(f"SUPERVISOR ENTERING CRASH_LOOP STATE! {decision.reason}")
                self._state = SupervisorState.CRASH_LOOP
            else:
                logger.info(f"SUPERVISOR STOPPING WORKER MONITORING: {decision.reason}")
                self._state = SupervisorState.STOPPED

        return self._state

    def stop_supervisor(self) -> None:
        """Graceful shutdown untuk Supervisor dan Worker-nya."""
        logger.info("Stopping NVRA Supervisor...")
        self._manual_stop_requested = True
        self._state = SupervisorState.STOPPING

        if self._worker_process and self._worker_process.poll() is None:
            logger.info(f"Sending SIGTERM to Trading Worker (PID: {self._worker_process.pid})...")
            self._worker_process.terminate()
            try:
                self._worker_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                logger.warning("Worker did not terminate within timeout. Forcing SIGKILL...")
                self._worker_process.kill()

        self._state = SupervisorState.STOPPED
        logger.info("Supervisor stopped cleanly.")
