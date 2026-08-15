"""
MODULE: tokocrypto_bot.recovery.recovery_policy
DESCRIPTION: Pure Decision Matrix for NVRA Trading Engine Recovery Gate.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class RecoveryDecision(str, Enum):
    PROCEED_TO_READY = "PROCEED_TO_READY"
    ENTER_PAUSED = "ENTER_PAUSED"
    ENTER_SAFE_MODE = "ENTER_SAFE_MODE"
    ABORT_EXIT = "ABORT_EXIT"

@dataclass(frozen=True)
class PolicyEvaluationResult:
    decision: RecoveryDecision
    reason: str

class RecoveryPolicy:
    @staticmethod
    def evaluate_startup_conditions(
        db_integrity_ok: bool,
        unclean_shutdown: bool,
        unresolved_orders_count: int,
        reconciliation_success: bool,
        balance_match: bool,
        position_match: bool,
        exchange_available: bool
    ) -> PolicyEvaluationResult:
        """Evaluasi matriks kebijakan startup secara ketat."""

        if not db_integrity_ok:
            return PolicyEvaluationResult(
                RecoveryDecision.ENTER_SAFE_MODE,
                "Database integrity check failed (PRAGMA integrity_check != OK)."
            )

        if not exchange_available:
            return PolicyEvaluationResult(
                RecoveryDecision.ENTER_PAUSED,
                "Exchange API is currently unreachable. Pausing trading until connection restores."
            )

        if unresolved_orders_count > 0 and not reconciliation_success:
            return PolicyEvaluationResult(
                RecoveryDecision.ENTER_SAFE_MODE,
                "Unresolved orders exist and reconciliation failed to establish 100% certainty."
            )

        if not balance_match or not position_match:
            return PolicyEvaluationResult(
                RecoveryDecision.ENTER_SAFE_MODE,
                f"Financial state mismatch detected (Balance match: {balance_match}, Position match: {position_match})."
            )

        if unclean_shutdown and reconciliation_success:
            return PolicyEvaluationResult(
                RecoveryDecision.PROCEED_TO_READY,
                "Recovered from unclean shutdown successfully via Reconciliation Gate."
            )

        return PolicyEvaluationResult(
            RecoveryDecision.PROCEED_TO_READY,
            "All startup verification checks passed perfectly."
        )
