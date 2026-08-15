"""
MODULE: tokocrypto_bot.application
DESCRIPTION: Main Application Controller & Worker Loop Entry Point for NVRA Engine.
"""

import sys
import time
import logging
from typing import Optional

from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.lifecycle_state import ApplicationState
from tokocrypto_bot.recovery.startup_recovery import StartupRecoveryOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("NVRA.Application")


class DummyExchangeAdapter:
    """Mock Exchange Adapter default jika direct connection belum terhubung."""
    def fetch_open_orders(self, symbol=None): return []
    def fetch_order_by_client_id(self, symbol, client_order_id): return None
    def fetch_recent_trades(self, symbol, limit=50): return []
    def fetch_account_balances(self): return {"USDT": {"free": 1000.0, "locked": 0.0}}


class ApplicationController:
    def __init__(self, db_path: Optional[str] = None):
        self.db_mgr = DatabaseManager(db_path=db_path)
        self.exchange_adapter = DummyExchangeAdapter()
        self.orchestrator = StartupRecoveryOrchestrator(self.db_mgr, self.exchange_adapter)

    def run(self):
        logger.info("Starting NVRA Application Controller Worker...")
        
        # Run P0-D Startup Recovery Gate
        boot_state = self.orchestrator.run_startup_recovery_gate()
        
        if boot_state != ApplicationState.READY:
            logger.warning(f"Boot finished with state [{boot_state.value}]. Trading loop HALTED.")

        # Heartbeat & Worker Main Loop
        try:
            while True:
                self.orchestrator.lifecycle_mgr.write_heartbeat({"status": "RUNNING_IDLE"})
                time.sleep(5.0)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received via KeyboardInterrupt.")
            self.orchestrator.shutdown_mgr.execute_graceful_shutdown("KeyboardInterrupt")


if __name__ == "__main__":
    app = ApplicationController()
    app.run()
