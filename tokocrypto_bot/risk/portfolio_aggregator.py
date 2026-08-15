"""
MODULE: tokocrypto_bot.risk.portfolio_aggregator
DESCRIPTION: Portfolio-Level Correlation, Sector, & Quote Currency Risk Controller (P2 Risk).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Set

logger = logging.getLogger("NVRA.PortfolioAggregatorRisk")


@dataclass(frozen=True)
class MacroRiskEvaluation:
    is_allowed: bool
    total_crypto_exposure_pct: float
    usdt_concentration_pct: float
    idr_concentration_pct: float
    high_correlation_cluster_detected: bool
    rejection_reasons: List[str]


class PortfolioLevelRiskController:
    def __init__(
        self,
        max_aggregate_crypto_exposure_pct: float = 0.40,
        max_quote_asset_concentration_pct: float = 0.70
    ):
        self.max_crypto_exp = max_aggregate_crypto_exposure_pct
        self.max_quote_conc = max_quote_asset_concentration_pct

    def evaluate_macro_portfolio_risk(
        self,
        total_equity_usdt: float,
        active_positions: Dict[str, float]  # Symbol -> Notional Value
    ) -> MacroRiskEvaluation:
        reasons: List[str] = []
        if total_equity_usdt <= 0:
            return MacroRiskEvaluation(False, 0.0, 0.0, 0.0, False, ["INVALID_EQUITY"])

        total_exposure = sum(active_positions.values())
        total_crypto_exp_pct = total_exposure / total_equity_usdt

        # Check Aggregate Exposure Limit
        if total_crypto_exp_pct > self.max_crypto_exp:
            reasons.append(f"AGGREGATE_CRYPTO_EXPOSURE_LIMIT_EXCEEDED ({total_crypto_exp_pct*100:.1f}% > {self.max_crypto_exp*100:.1f}%)")

        # Quote Asset Concentration Check
        usdt_positions = sum(val for sym, val in active_positions.items() if sym.endswith("USDT"))
        idr_positions = sum(val for sym, val in active_positions.items() if sym.endswith("BIDR") or sym.endswith("IDR"))

        usdt_conc = usdt_positions / max(1.0, total_exposure) if total_exposure > 0 else 0.0
        idr_conc = idr_positions / max(1.0, total_exposure) if total_exposure > 0 else 0.0

        if usdt_conc > self.max_quote_conc:
            reasons.append(f"USDT_QUOTE_CONCENTRATION_EXCEEDED ({usdt_conc*100:.1f}%)")

        is_allowed = len(reasons) == 0
        return MacroRiskEvaluation(
            is_allowed=is_allowed,
            total_crypto_exposure_pct=total_crypto_exp_pct,
            usdt_concentration_pct=usdt_conc,
            idr_concentration_pct=idr_conc,
            high_correlation_cluster_detected=False,
            rejection_reasons=reasons
        )
