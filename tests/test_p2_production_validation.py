"""
MODULE: tests.test_p2_production_validation
DESCRIPTION: Test Suite for P2 Production Validation, Champion/Challenger ML & Hard Live Gate.
"""

import os
import pytest
import tempfile
import pandas as pd

from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.migrations import run_migrations
from tokocrypto_bot.quant.performance_evaluator import PerformanceEvaluator
from tokocrypto_bot.strategy.strategy_health import StrategyHealthMonitor, StrategyHealthState
from tokocrypto_bot.ml.promotion_gate import ModelPromotionGate
from tokocrypto_bot.risk.portfolio_aggregator import PortfolioLevelRiskController
from tokocrypto_bot.recovery.live_gate import HardLiveGate


def test_net_expectancy_deducts_fees_and_slippage():
    evaluator = PerformanceEvaluator(default_fee_pct=0.001, default_slippage_pct=0.0005)
    df_trades = pd.DataFrame([
        {"pnl_usdt": 10.0, "return_pct": 0.01, "notional_usdt": 1000.0},
        {"pnl_usdt": -5.0, "return_pct": -0.005, "notional_usdt": 1000.0}
    ])
    report = evaluator.evaluate_trades(df_trades)
    assert report.total_trades == 2
    assert report.net_expectancy < report.gross_edge  # Frictions harus memotong gross edge


def test_strategy_health_auto_degradation():
    monitor = StrategyHealthMonitor(min_expectancy_usdt=1.0, max_allowed_drawdown=0.05)
    evaluator = PerformanceEvaluator()
    df_losing = pd.DataFrame([{"pnl_usdt": -10.0, "return_pct": -0.02, "notional_usdt": 500.0}] * 15)
    report = evaluator.evaluate_trades(df_losing)
    status = monitor.evaluate_strategy_health("ScalpingStrategy", report)
    assert status.state == StrategyHealthState.DEGRADED


def test_hard_live_gate_blocks_if_credentials_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_mgr = DatabaseManager(db_path=os.path.join(tmpdir, "live_gate.db"))
        run_migrations(db_mgr)
        gate = HardLiveGate(db_mgr)

        res = gate.verify_all_live_conditions(
            is_p0_passed=True,
            is_p1_paper_passed=True,
            is_model_valid=True,
            is_strategy_healthy=True
        )
        assert res.live_allowed is False
        assert "CREDENTIAL_MISSING_FAIL" in res.failed_checks
