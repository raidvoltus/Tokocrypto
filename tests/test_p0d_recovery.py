"""
MODULE: tests.test_p0d_recovery
DESCRIPTION: Integration Test Suite for P0-D Startup Recovery & Mutex Locking.
"""

import os
import pytest
import tempfile
from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.lifecycle_state import ApplicationState
from tokocrypto_bot.recovery.startup_recovery import StartupRecoveryOrchestrator
from tokocrypto_bot.recovery.single_instance import InstanceAlreadyRunningException, SingleInstanceLock
from tokocrypto_bot.execution.reconciliation import OrderStateMachine


class MockExchangeAdapter:
    def fetch_open_orders(self, symbol=None): return []
    def fetch_order_by_client_id(self, symbol, client_order_id): return None
    def fetch_recent_trades(self, symbol, limit=50): return []
    def fetch_account_balances(self): return {"USDT": {"free": 1000.0, "locked": 0.0}}


@pytest.fixture
def recovery_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "recovery_test.db")
        db_mgr = DatabaseManager(db_path=db_path)
        mock_ex = MockExchangeAdapter()
        orchestrator = StartupRecoveryOrchestrator(db_mgr, mock_ex, lock_name="NVRA_TEST_LOCK")
        yield orchestrator, db_mgr, mock_ex


def test_single_instance_lock_blocks_second_instance():
    lock1 = SingleInstanceLock("NVRA_UNIQUE_TEST_MUTEX")
    lock2 = SingleInstanceLock("NVRA_UNIQUE_TEST_MUTEX")

    assert lock1.acquire() is True
    with pytest.raises(InstanceAlreadyRunningException):
        lock2.acquire()

    lock1.release()
    assert lock2.acquire() is True
    lock2.release()


def test_startup_recovery_gate_clean_boot(recovery_env):
    orchestrator, _, _ = recovery_env
    app_state = orchestrator.run_startup_recovery_gate()
    assert app_state == ApplicationState.READY
    orchestrator.instance_lock.release()


def test_startup_recovery_reconciles_unclean_shutdown(recovery_env):
    orchestrator, state_mgr, mock_ex = recovery_env
    
    # Simulate Unclean Shutdown with SUBMITTING Order
    cid = OrderStateMachine.generate_client_order_id("EXEC-UNSHUTDOWN", "SIG-01", "BTCUSDT", "BUY")
    orchestrator.state_mgr.create_order_intent(cid, "EXEC-UNSHUTDOWN", "SIG-01", "BTCUSDT", "BUY", "LIMIT", 60000.0, 0.5)
    orchestrator.state_mgr.transition_order_state(cid, "CREATED", "SUBMITTING", "POST_SENT")

    # Mock Exchange returning FILLED for this order
    mock_ex.fetch_order_by_client_id = lambda symbol, client_order_id: {
        "orderId": "998877", "clientOrderId": cid, "symbol": "BTCUSDT",
        "status": "FILLED", "price": "60000.0", "origQty": "0.5", "executedQty": "0.5", "cummulativeQuoteQty": "30000.0"
    }

    # Execute Startup Recovery Gate
    app_state = orchestrator.run_startup_recovery_gate()
    assert app_state == ApplicationState.READY

    order_in_db = orchestrator.state_mgr.get_order(cid)
    assert order_in_db["status"] == "FILLED"
    orchestrator.instance_lock.release()
