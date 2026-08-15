"""
MODULE: tokocrypto_bot.supervisor.restart_policy
DESCRIPTION: Exit Code Classifier and Restart Policy Matrix for Supervisor.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class WorkerExitCategory(str, Enum):
    NORMAL_SHUTDOWN = "NORMAL_SHUTDOWN"       # Code 0: Shutdown normal / SIGTERM
    USER_STOPPED = "USER_STOPPED"             # Code 0 / Custom: User mematikan secara sengaja
    SAFE_MODE_EXIT = "SAFE_MODE_EXIT"         # Worker keluar karena masuk SAFE_MODE
    UNHANDLED_EXCEPTION = "UNHANDLED_EXCEPTION"# Code 1 atau crash tidak terduga
    WATCHDOG_KILLED = "WATCHDOG_KILLED"       # Dibunuh oleh supervisor karena stale heartbeat

@dataclass(frozen=True)
class RestartDecision:
    should_restart: bool
    reason: str

class RestartPolicy:
    @staticmethod
    def classify_exit_code(exit_code: Optional[int]) -> WorkerExitCategory:
        if exit_code == 0:
            return WorkerExitCategory.NORMAL_SHUTDOWN
        elif exit_code == 100:
            return WorkerExitCategory.SAFE_MODE_EXIT
        elif exit_code == 130 or exit_code == 143:  # SIGINT / SIGTERM
            return WorkerExitCategory.USER_STOPPED
        else:
            return WorkerExitCategory.UNHANDLED_EXCEPTION

    @staticmethod
    def evaluate_restart(
        exit_category: WorkerExitCategory,
        crash_loop_triggered: bool,
        is_manual_stop: bool = False
    ) -> RestartDecision:
        if is_manual_stop or exit_category == WorkerExitCategory.USER_STOPPED:
            return RestartDecision(False, "User requested manual stop. Auto-restart disabled.")

        if exit_category == WorkerExitCategory.NORMAL_SHUTDOWN:
            return RestartDecision(False, "Worker exited normally (Code 0).")

        if crash_loop_triggered:
            return RestartDecision(
                False,
                "CRASH-LOOP PROTECTION TRIGGERED: Maximum crash limit reached within rolling window. Auto-restart HALTED."
            )

        if exit_category == WorkerExitCategory.UNHANDLED_EXCEPTION or exit_category == WorkerExitCategory.WATCHDOG_KILLED:
            return RestartDecision(True, f"Worker crashed unexpectedly ({exit_category.value}). Initiating restart sequence via P0-D.")

        return RestartDecision(False, f"No restart action defined for exit category '{exit_category.value}'.")
