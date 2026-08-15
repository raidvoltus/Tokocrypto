"""
MODULE: tests.test_reconciliation
DESCRIPTION: Unit and Integration Tests for P0-C Reconciliation Engine.
"""

import os
import pytest
import tempfile
from typing import List, Dict, Any, Optional

from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.migrations import run_migrations
from tokocrypto_bot.persistence.state_manager import StateManager
from tokocrypto_bot.execution.order_state_machine import OrderStateMachine, OrderStatus
from tokocrypto_bot.execution.reconciliation import (
    ReconciliationEngine, SystemRecoveryStatus, ReconciliationDecision
)


class MockExchangeAdapter:
    """Mock Exchange Adapter untuk menguji respon API Tokocrypto tanpa koneksi jaringan."""

    def __init__(self):
        self.open_orders_response: List[Dict[str, Any]] = []
        self.order_detail_response: Dict[str, Dict[str, Any]] = {}
        self.recent_trades_response: List[Dict[str, Any]] = []

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.open_orders_response

    def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        return self.order_detail_response.get(client_order_id)

    def fetch_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self.recent_trades_response

    def fetch_account_balances(self) -> Dict[str, Dict[str, float]]:
        return {"USDT": {"free": 1000.0, "locked": 0.0}}


@pytest.fixture
def setup_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_reconciliation.db")
        db_mgr = DatabaseManager(db_path=db_path)
        run_migrations(db_mgr)
        state_mgr = StateManager(db_mgr)
        mock_exchange = MockExchangeAdapter()
        engine = ReconciliationEngine(state_mgr, mock_exchange)
        yield state_mgr, mock_exchange, engine


def test_reconcile_filled_order_via_order_detail(setup_env):
    state_mgr, mock_exchange, engine = setup_env
    cid = OrderStateMachine.generate_client_order_id("EXEC-001", "SIG-01", "BTCUSDT", "BUY")

    # 1. Setup Order Intent -> SUBMITTING -> UNKNOWN
    state_mgr.create_order_intent(cid, "EXEC-001", "SIG-01", "BTCUSDT", "BUY", "LIMIT", 60000.0, 0.1)
    state_mgr.transition_order_state(cid, "CREATED", "SUBMITTING", "POST_SENT")
    state_mgr.transition_order_state(cid, "SUBMITTING", "UNKNOWN", "NETWORK_TIMEOUT")

    # 2. Setup Mock Exchange returning FILLED
    mock_exchange.order_detail_response[cid] = {
        "orderId": "12345678",
        "clientOrderId": cid,
        "symbol": "BTCUSDT",
        "status": "FILLED",
        "price": "60000.0",
        "origQty": "0.1",
        "executedQty": "0.1",
        "cummulativeQuoteQty": "6000.0"
    }

    # 3. Run Reconciliation
    recovery_status = engine.reconcile_all_unresolved_orders("EXEC-001")

    assert recovery_status == SystemRecoveryStatus.RECOVERY_COMPLETE
    order_db = state_mgr.get_order(cid)
    assert order_db["status"] == "FILLED"
    assert order_db["exchange_order_id"] == "12345678"


def test_reconcile_not_found_triggers_safe_mode(setup_env):
    """Memverifikasi aturan kritis: NOT_FOUND TIDAK BISA otomatis jadi REJECTED -> Wajib masuk SAFE_MODE."""
    state_mgr, mock_exchange, engine = setup_env
    cid = OrderStateMachine.generate_client_order_id("EXEC-002", "SIG-02", "ETHUSDT", "BUY")

    state_mgr.create_order_intent(cid, "EXEC-002", "SIG-02", "ETHUSDT", "BUY", "LIMIT", 3000.0, 1.0)
    state_mgr.transition_order_state(cid, "CREATED", "SUBMITTING", "POST_SENT")
    state_mgr.transition_order_state(cid, "SUBMITTING", "UNKNOWN", "NETWORK_TIMEOUT")

    # Exchange tidak mengembalikan apa-apa (NOT_FOUND)
    recovery_status = engine.reconcile_all_unresolved_orders("EXEC-002")

    assert recovery_status == SystemRecoveryStatus.SAFE_MODE
    order_db = state_mgr.get_order(cid)
    assert order_db["status"] == "UNKNOWN", "Status order harus TETAP UNKNOWN demi keamanan"


def test_reconcile_api_error_preserves_unknown(setup_env):
    state_mgr, mock_exchange, engine = setup_env
    cid = OrderStateMachine.generate_client_order_id("EXEC-003", "SIG-03", "SOLUSDT", "BUY")

    state_mgr.create_order_intent(cid, "EXEC-003", "SIG-03", "SOLUSDT", "BUY", "LIMIT", 150.0, 2.0)
    state_mgr.transition_order_state(cid, "CREATED", "SUBMITTING", "POST_SENT")

    # Mock Exception
    def throw_error(*args, **kwargs):
        raise ConnectionError("502 Bad Gateway")
    mock_exchange.fetch_open_orders = throw_error

    recovery_status = engine.reconcile_all_unresolved_orders("EXEC-003")

    assert recovery_status == SystemRecoveryStatus.SAFE_MODE
    order_db = state_mgr.get_order(cid)
    assert order_db["status"] == "UNKNOWN"
