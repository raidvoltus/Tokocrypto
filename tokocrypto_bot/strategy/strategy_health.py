"""
MODULE: tokocrypto_bot.strategy.strategy_health
DESCRIPTION: Per-Strategy Health Monitor & Automatic Degradation Queue (P2-B).
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

from tokocrypto_bot.quant.performance_evaluator import PerformanceReport

logger = logging.getLogger("NVRA.StrategyHealth")


class StrategyHealthState(str, Enum):
    HEALTHY = "HEALTHY"
    CAUTION = "CAUTION"
    DEGRADED = "DEGRADED"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class StrategyHealthStatus:
    strategy_name: str
    state: StrategyHealthState
    net_expectancy: float
    profit_factor: float
    drawdown_pct: float
    consecutive_degradations: int
    recommendation: str


class StrategyHealthMonitor:
    def __init__(self, min_expectancy_usdt: float = 0.5, max_allowed_drawdown: float = 0.08):
        self.min_expectancy = min_expectancy_usdt
        self.max_dd = max_allowed_drawdown

    def evaluate_strategy_health(self, strategy_name: str, report: PerformanceReport) -> StrategyHealthStatus:
        if report.total_trades < 10:
            return StrategyHealthStatus(
                strategy_name, StrategyHealthState.HEALTHY,
                report.net_expectancy, report.profit_factor, report.max_drawdown_pct, 0, "Insufficient trade sample for health check."
            )

        if report.net_expectancy < 0 or report.max_drawdown_pct >= self.max_dd:
            state = StrategyHealthState.DEGRADED
            rec = f"Strategy DEGRADED: Negative Net Expectancy (${report.net_expectancy:.2f}) or High Drawdown ({report.max_drawdown_pct*100:.1f}%)."
        elif report.profit_factor < 1.1 or report.net_expectancy < self.min_expectancy:
            state = StrategyHealthState.CAUTION
            rec = f"Strategy CAUTION: Low Profit Factor ({report.profit_factor:.2f}). Position sizing reduced."
        else:
            state = StrategyHealthState.HEALTHY
            rec = "Strategy HEALTHY."

        logger.info(f"STRATEGY HEALTH CHECK [{strategy_name}]: {state.value} - {rec}")
        return StrategyHealthStatus(
            strategy_name=strategy_name,
            state=state,
            net_expectancy=report.net_expectancy,
            profit_factor=report.profit_factor,
            drawdown_pct=report.max_drawdown_pct,
            consecutive_degradations=1 if state != StrategyHealthState.HEALTHY else 0,
            recommendation=rec
        )
