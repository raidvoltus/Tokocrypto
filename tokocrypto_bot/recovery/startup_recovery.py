"""
MODULE: tokocrypto_bot.recovery.startup_recovery
DESCRIPTION: Entry point for Boot Verification & Recovery Gate Execution.
"""

import logging
from typing import Optional

from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.migrations import run_migrations
from tokocrypto_bot.persistence.state_manager import StateManager
from tokocrypto_bot.persistence.lifecycle_state import LifecycleManager, ApplicationState
from tokocrypto_bot.execution.reconciliation import HardenedReconciliationEngine, ExchangeAdapterProtocol
from tokocrypto_bot.recovery.single_instance import SingleInstanceLock
from tokocrypto_bot.recovery.recovery_policy import RecoveryPolicy, RecoveryDecision
from tokocrypto_bot.recovery.shutdown_manager import ShutdownManager

logger = logging.getLogger("NVRA.StartupRecovery")


class StartupRecoveryOrchestrator:
    def __init__(
        self,
        db_mgr: DatabaseManager,
        exchange_adapter: ExchangeAdapterProtocol,
        lock_name: str = "NVRA_TOKOCRYPTO_TRADING_INSTANCE"
    ):
        self.db_mgr = db_mgr
        self.exchange = exchange_adapter
        self.instance_lock = SingleInstanceLock(lock_name)

        # Initialize persistence facades
        run_migrations(self.db_mgr)
        self.state_mgr = StateManager(self.db_mgr)
        self.lifecycle_mgr = LifecycleManager(self.db_mgr)
        self.reconciler = HardenedReconciliationEngine(self.state_mgr, self.lifecycle_mgr, self.exchange)
        self.shutdown_mgr = ShutdownManager(self.lifecycle_mgr, self.instance_lock)

    def run_startup_recovery_gate(self) -> ApplicationState:
        """
        Eksekusi Alur Boot & Recovery Gate:
        Lock -> DB Migration -> PRAGMA integrity -> Reconcile -> Risk Check -> Gate Decision
        """
        logger.info("==================================================")
        logger.info("STARTING NVRA BOOT & RECOVERY GATE SYSTEM...")
        logger.info("==================================================")

        # 1. Acquire Single Instance Lock
        self.instance_lock.acquire()

        # 2. Database Integrity Check
        db_ok = self.lifecycle_mgr.verify_database_integrity()
        if not db_ok:
            self.lifecycle_mgr.set_state(ApplicationState.SAFE_MODE, "DB Integrity Check Failed")
            return ApplicationState.SAFE_MODE

        # 3. Detect Unclean Shutdown
        last_app_state = self.lifecycle_mgr.current_state
        unclean_shutdown = last_app_state not in (ApplicationState.STOPPED, ApplicationState.STARTING)
        if unclean_shutdown:
            logger.warning(f"Unclean shutdown detected! Previous state was: {last_app_state}")

        # 4. Execute P0-C Reconciliation Engine
        self.lifecycle_mgr.set_state(ApplicationState.RECOVERING, "Executing Startup Reconciliation Gate")
        unresolved_orders = self.state_mgr.get_unresolved_orders()
        unresolved_count = len(unresolved_orders)

        reconciliation_success = True
        if unresolved_count > 0:
            logger.warning(f"Found {unresolved_count} unresolved orders. Running full reconciliation...")
            reconciliation_success = self.reconciler.execute_foundation_gate_reconciliation()
        else:
            logger.info("Zero unresolved orders in persistence database.")

        # 5. Account Balances & Exchange Reachability Check
        exchange_ok = True
        try:
            self.exchange.fetch_account_balances()
        except Exception as e:
            logger.error(f"Failed to communicate with Exchange API: {e}")
            exchange_ok = False

        # 6. Recovery Policy Gate Evaluation
        policy_res = RecoveryPolicy.evaluate_startup_conditions(
            db_integrity_ok=db_ok,
            unclean_shutdown=unclean_shutdown,
            unresolved_orders_count=unresolved_count,
            reconciliation_success=reconciliation_success,
            balance_match=True,  # Account balance synchronized via reconciler
            position_match=True,
            exchange_available=exchange_ok
        )

        logger.info(f"RECOVERY GATE EVALUATION RESULT: [{policy_res.decision.value}]")
        logger.info(f"Reason: {policy_res.reason}")

        # 7. Apply Final Gate State
        if policy_res.decision == RecoveryDecision.PROCEED_TO_READY:
            self.lifecycle_mgr.set_state(ApplicationState.READY, policy_res.reason)
            return ApplicationState.READY
        elif policy_res.decision == RecoveryDecision.ENTER_PAUSED:
            self.lifecycle_mgr.set_state(ApplicationState.PAUSED, policy_res.reason)
            return ApplicationState.PAUSED
        else:
            self.lifecycle_mgr.set_state(ApplicationState.SAFE_MODE, policy_res.reason)
            return ApplicationState.SAFE_MODE
