"""
MODULE: tokocrypto_bot.recovery.live_gate
DESCRIPTION: P2-E Hard Live Gate verification engine for NVRA Trading System.
"""

import logging
from dataclasses import dataclass
from typing import List

from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.state_manager import StateManager
from tokocrypto_bot.persistence.lifecycle_state import ApplicationState
from tokocrypto_bot.security.credential_manager import SecureCredentialStore

logger = logging.getLogger("NVRA.LiveGate")


@dataclass(frozen=True)
class LiveGateVerificationResult:
    live_allowed: bool
    failed_checks: List[str]
    passed_checks: List[str]
    summary: str


class HardLiveGate:
    def __init__(self, db_mgr: DatabaseManager):
        self.db = db_mgr
        self.state_mgr = StateManager(db_mgr)
        self.cred_store = SecureCredentialStore()

    def verify_all_live_conditions(
        self,
        is_p0_passed: bool,
        is_p1_paper_passed: bool,
        is_model_valid: bool,
        is_strategy_healthy: bool,
        is_kill_switch_active: bool = False
    ) -> LiveGateVerificationResult:
        passed = []
        failed = []

        # 1. P0 Reliability Check
        if is_p0_passed: passed.append("P0_RELIABILITY_PASS")
        else: failed.append("P0_RELIABILITY_FAIL")

        # 2. P1 Paper/Shadow Check
        if is_p1_paper_passed: passed.append("P1_PAPER_SHADOW_PASS")
        else: failed.append("P1_PAPER_SHADOW_FAIL")

        # 3. DB Integrity Check
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            res = cursor.fetchone()[0]
            if res.lower() == "ok": passed.append("DB_INTEGRITY_PASS")
            else: failed.append("DB_INTEGRITY_FAIL")
        except Exception:
            failed.append("DB_INTEGRITY_EXCEPTION")
        finally:
            conn.close()

        # 4. Unresolved Orders Check
        unresolved = self.state_mgr.get_unresolved_orders()
        if len(unresolved) == 0: passed.append("ZERO_UNRESOLVED_ORDERS_PASS")
        else: failed.append(f"UNRESOLVED_ORDERS_EXIST_COUNT_{len(unresolved)}")

        # 5. Model Validation
        if is_model_valid: passed.append("MODEL_VALIDATION_PASS")
        else: failed.append("MODEL_VALIDATION_FAIL")

        # 6. Strategy Health
        if is_strategy_healthy: passed.append("STRATEGY_HEALTH_PASS")
        else: failed.append("STRATEGY_HEALTH_FAIL")

        # 7. Kill Switch Operational
        if not is_kill_switch_active: passed.append("KILL_SWITCH_IDLE_PASS")
        else: failed.append("KILL_SWITCH_ACTIVE_FAIL")

        # 8. DPAPI Credentials Verification
        key, secret = self.cred_store.load_api_credentials()
        if key and secret: passed.append("CREDENTIAL_DPAPI_PASS")
        else: failed.append("CREDENTIAL_MISSING_FAIL")

        live_allowed = len(failed) == 0
        summary = "LIVE TRADING UNLOCKED" if live_allowed else f"LIVE TRADING BLOCKED ({len(failed)} critical checks failed)"

        logger.info(f"HARD LIVE GATE EVALUATION: {summary}")
        return LiveGateVerificationResult(
            live_allowed=live_allowed,
            failed_checks=failed,
            passed_checks=passed,
            summary=summary
        )
