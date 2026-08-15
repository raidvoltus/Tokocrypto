"""
MODULE: tokocrypto_bot.quant.performance_evaluator
DESCRIPTION: Performance Metrics & Net Expectancy Evaluator for P2 Validation.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any

logger = logging.getLogger("NVRA.PerformanceEvaluator")


@dataclass(frozen=True)
class PerformanceReport:
    total_trades: int
    win_rate: float
    gross_edge: float
    net_expectancy: float
    profit_factor: float
    average_r: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_fee_cost_usdt: float
    total_slippage_usdt: float
    signal_to_fill_ratio: float


class PerformanceEvaluator:
    def __init__(self, default_fee_pct: float = 0.001, default_slippage_pct: float = 0.0005):
        self.fee_pct = default_fee_pct
        self.slippage_pct = default_slippage_pct

    def evaluate_trades(self, df_trades: pd.DataFrame) -> PerformanceReport:
        if df_trades is None or df_trades.empty:
            return PerformanceReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total_trades = len(df_trades)
        pnl = df_trades["pnl_usdt"].values
        returns = df_trades["return_pct"].values
        notionals = df_trades["notional_usdt"].values

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-8
        profit_factor = gross_profit / gross_loss

        # Frictions: Fees & Slippage Calculation
        fee_costs = np.sum(notionals * (self.fee_pct * 2.0))  # Roundtrip Fee
        slippage_costs = np.sum(notionals * self.slippage_pct)
        total_frictions = fee_costs + slippage_costs

        net_pnl = np.sum(pnl) - total_frictions
        net_expectancy = net_pnl / total_trades if total_trades > 0 else 0.0
        gross_edge = np.sum(pnl) / total_trades if total_trades > 0 else 0.0

        # Risk Ratios
        cum_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdowns = (peak - cum_returns) / np.maximum(1e-8, peak)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        std_return = np.std(returns) if len(returns) > 1 else 1e-8
        sharpe = (np.mean(returns) / std_return) * np.sqrt(252) if std_return > 0 else 0.0

        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-8
        sortino = (np.mean(returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

        return PerformanceReport(
            total_trades=total_trades,
            win_rate=win_rate,
            gross_edge=gross_edge,
            net_expectancy=net_expectancy,
            profit_factor=profit_factor,
            average_r=np.mean(pnl) / max(1e-8, abs(np.mean(losses))) if len(losses) > 0 else 0.0,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            total_fee_cost_usdt=fee_costs,
            total_slippage_usdt=slippage_costs,
            signal_to_fill_ratio=1.0
        )
