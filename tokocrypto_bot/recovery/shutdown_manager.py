"""
MODULE: tokocrypto_bot.recovery.shutdown_manager
DESCRIPTION: Graceful Shutdown Manager for Signal Handling and Clean Exit.
"""

import sys
import signal
import logging
from typing import Callable, List

from tokocrypto_bot.persistence.lifecycle_state import LifecycleManager, ApplicationState
from tokocrypto_bot.recovery.single_instance import SingleInstanceLock

logger = logging.getLogger("NVRA.ShutdownManager")

class ShutdownManager:
    def __init__(self, lifecycle_mgr: LifecycleManager, instance_lock: SingleInstanceLock):
        self.lifecycle_mgr = lifecycle_mgr
        self.instance_lock = instance_lock
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._is_shutting_down = False

        self._register_signals()

    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        self._cleanup_callbacks.append(callback)

    def _register_signals(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if sys.platform == "win32":
            # Direct Windows Console CTRL Event Handling if applicable
            pass

    def _signal_handler(self, signum, frame) -> None:
        if self._is_shutting_down:
            logger.warning("Forced shutdown requested. Exiting immediately.")
            sys.exit(1)

        sig_name = signal.Signals(signum).name
        logger.warning(f"Shutdown signal received: [{sig_name}]. Initiating Graceful Shutdown...")
        self.execute_graceful_shutdown(reason=f"Signal {sig_name}")

    def execute_graceful_shutdown(self, reason: str = "Clean Shutdown") -> None:
        if self._is_shutting_down:
            return
        self._is_shutting_down = True

        logger.info("Executing Graceful Shutdown Sequence:")
        logger.info("1. Stopping New Orders (State set to STOPPING)...")
        self.lifecycle_mgr.set_state(ApplicationState.STOPPING, reason=reason)

        logger.info("2. Executing Cleanup Callbacks...")
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error during cleanup callback: {e}")

        logger.info("3. Marking Clean Shutdown in Database...")
        self.lifecycle_mgr.set_state(ApplicationState.STOPPED, reason="Clean Shutdown Complete")

        logger.info("4. Releasing Single Instance Lock...")
        self.instance_lock.release()

        logger.info("Graceful Shutdown Complete. Terminating process.")
        sys.exit(0)
