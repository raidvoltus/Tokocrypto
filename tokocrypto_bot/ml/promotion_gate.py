"""
MODULE: tokocrypto_bot.ml.promotion_gate
DESCRIPTION: ML Model Promotion Gate (Champion vs Challenger OOS Validation) (P2-C).
"""

import os
import pickle
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from tokocrypto_bot.quant.performance_evaluator import PerformanceEvaluator, PerformanceReport

logger = logging.getLogger("NVRA.MLPromotionGate")


@dataclass(frozen=True)
class ModelPromotionResult:
    champion_version: str
    challenger_version: str
    is_promoted: bool
    champion_expectancy: float
    challenger_expectancy: float
    expectancy_lift_pct: float
    reason: str


class ModelPromotionGate:
    def __init__(self, min_expectancy_lift_pct: float = 0.10):
        self.min_lift = min_expectancy_lift_pct  # Wajib lebih baik 10% dibanding Champion

    def evaluate_promotion(
        self,
        champion_report: PerformanceReport,
        challenger_report: PerformanceReport,
        champion_version: str,
        challenger_version: str
    ) -> ModelPromotionResult:
        """Menilai apakah Challenger berhak dipromosikan menggantikan Champion."""

        # 1. Sample Size Check
        if challenger_report.total_trades < 30:
            return ModelPromotionResult(
                champion_version, challenger_version, False,
                champion_report.net_expectancy, challenger_report.net_expectancy, 0.0,
                "Challenger OOS trade sample < 30. Promotion REJECTED."
            )

        # 2. Net Expectancy & Friction Check
        if challenger_report.net_expectancy <= 0:
            return ModelPromotionResult(
                champion_version, challenger_version, False,
                champion_report.net_expectancy, challenger_report.net_expectancy, 0.0,
                "Challenger Net Expectancy <= 0 after frictions. Promotion REJECTED."
            )

        # 3. Comparative Performance Lift
        champ_exp = max(1e-8, champion_report.net_expectancy)
        lift = (challenger_report.net_expectancy - champ_exp) / champ_exp

        if lift >= self.min_lift and challenger_report.max_drawdown_pct <= champion_report.max_drawdown_pct:
            logger.info(f"MODEL PROMOTED! Challenger v{challenger_version} beat Champion v{champion_version} by {lift*100:.1f}% Net Expectancy lift.")
            return ModelPromotionResult(
                champion_version, challenger_version, True,
                champion_report.net_expectancy, challenger_report.net_expectancy, lift,
                f"Challenger PASSED all promotion gates with {lift*100:.1f}% expectancy lift."
            )

        return ModelPromotionResult(
            champion_version, challenger_version, False,
            champion_report.net_expectancy, challenger_report.net_expectancy, lift,
            f"Challenger lift ({lift*100:.1f}%) < required minimum threshold ({self.min_lift*100:.1f}%)."
        )
