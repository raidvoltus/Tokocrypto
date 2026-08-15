"""
MODULE: tests.test_p1f_autonomous_worker
DESCRIPTION: Comprehensive Paper & Integration Test Suite for P1-F Autonomous Worker Loop.
"""

import os
import pytest
import tempfile
from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.migrations import run_migrations
from tokocrypto_bot.persistence.lifecycle_state import ApplicationState
from tokocrypto_bot.application import AutonomousTradingWorker, ExecutionMode, PaperExchangeAdapter
from tokocrypto_bot.strategy.decision import DecisionAction
from tokocrypto_bot.strategy.portfolio import PositionPlan, RiskDecision, RiskAction


@pytest.fixture
def worker_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "p1f_worker_test.db")
        worker = AutonomousTradingWorker(mode=ExecutionMode.PAPER, db_path=db_path)
        run_migrations(worker.db_mgr)
        yield worker


def test_paper_worker_cycle_execution(worker_env):
    worker = worker_env
    # Perform single autonomous cycle
    worker._run_single_autonomous_cycle()

    # Lifecycle state must return to READY after cycle
    assert worker.orchestrator.lifecycle_mgr.current_state == ApplicationState.READY
    # Heartbeat entry exists
    conn = worker.db_mgr.get_connection()
    hb = conn.execute("SELECT value FROM bot_state WHERE key='heartbeat'").fetchone()
    conn.close()
    assert hb is not None


def test_network_timeout_transitions_to_unknown_without_retry(worker_env):
    worker = worker_env

    # Mock Exception on POST
    def throw_timeout(*args, **kwargs):
        raise ConnectionError("Timeout communicating with Tokocrypto POST endpoint")
    
    worker.mode = ExecutionMode.LIVE
    worker.exchange.post_order_non_retry = throw_timeout

    mock_plan = PositionPlan(
        symbol="BTCUSDT", timestamp=1700000000000, action=DecisionAction.BUY,
        approved_notional_usdt=100.0, calculated_quantity=0.002, target_price=50000.0,
        stop_loss_price=48000.0, take_profit_price=54000.0,
        risk_decision=RiskDecision("BTCUSDT", RiskAction.ALLOW, 100.0, 100.0, 0.01, 0.01, 0.04, 4.0, 0.9, [], ["PASS"])
    )

    worker._execute_single_position_plan(mock_plan)

    # Order must be persisted as UNKNOWN in database
    unresolved = worker.state_mgr.get_unresolved_orders()
    assert len(unresolved) == 1
    assert unresolved[0]["status"] == "UNKNOWN"
