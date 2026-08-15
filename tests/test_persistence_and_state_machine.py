"""
MODULE: tests.test_persistence_and_state_machine
DESCRIPTION: Comprehensive unit & integration tests for P0-A (State Machine) and P0-B (SQLite WAL Persistence).
"""

import os
import pytest
import tempfile
from pathlib import Path

from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.migrations import run_migrations
from tokocrypto_bot.persistence.state_manager import StateManager
from tokocrypto_bot.execution.order_state_machine import (
    OrderStateMachine, OrderStatus, InvalidStateTransitionException
)

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_state.db")
        db_mgr = DatabaseManager(db_path=db_path)
        run_migrations(db_mgr)
        yield db_mgr

def test_deterministic_client_order_id_length():
    cid1 = OrderStateMachine.generate_client_order_id("EXEC-001", "SIG-999", "BTCUSDT", "BUY")
    cid2 = OrderStateMachine.generate_client_order_id("EXEC-001", "SIG-999", "BTCUSDT", "BUY")
    
    assert cid1 == cid2, "Client Order ID harus deterministik untuk input yang sama"
    assert len(cid1) <= 36, f"Client Order ID melebihi limit Tokocrypto 36 chars: {len(cid1)}"
    assert cid1.startswith("QBOT-")

def test_illegal_state_transition():
    with pytest.raises(InvalidStateTransitionException):
        OrderStateMachine.validate_transition(OrderStatus.FILLED, OrderStatus.SUBMITTING)

def test_atomic_order_intent_creation_and_recovery(temp_db):
    state_mgr = StateManager(temp_db)
    cid = OrderStateMachine.generate_client_order_id("EXEC-101", "SIG-01", "ETHUSDT", "BUY")

    # 1. Create Order Intent
    success = state_mgr.create_order_intent(
        client_order_id=cid,
        execution_id="EXEC-101",
        signal_id="SIG-01",
        symbol="ETHUSDT",
        side="BUY",
        order_type="LIMIT",
        price=3000.0,
        quantity=0.5
    )
    assert success is True

    # 2. Duplicate Check
    dup_success = state_mgr.create_order_intent(
        client_order_id=cid,
        execution_id="EXEC-101",
        signal_id="SIG-01",
        symbol="ETHUSDT",
        side="BUY",
        order_type="LIMIT",
        price=3000.0,
        quantity=0.5
    )
    assert dup_success is False, "Duplicate order intent harus ditolak"

    # 3. Transition to SUBMITTING
    OrderStateMachine.validate_transition(OrderStatus.CREATED, OrderStatus.SUBMITTING)
    state_mgr.transition_order_state(cid, OrderStatus.CREATED.value, OrderStatus.SUBMITTING.value, "HTTP_POST_INITIATED")

    # 4. Simulate Timeout -> UNKNOWN
    OrderStateMachine.validate_transition(OrderStatus.SUBMITTING, OrderStatus.UNKNOWN)
    state_mgr.transition_order_state(cid, OrderStatus.SUBMITTING.value, OrderStatus.UNKNOWN.value, "NETWORK_TIMEOUT")

    # Verify persistent state in DB
    unresolved = state_mgr.get_unresolved_orders()
    assert len(unresolved) == 1
    assert unresolved[0]["client_order_id"] == cid
    assert unresolved[0]["status"] == "UNKNOWN"

def test_wal_mode_and_backup(temp_db):
    conn = temp_db.get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    conn.close()
    assert mode.lower() == "wal", "Database harus beroperasi dalam mode WAL"

    backup_path = temp_db.create_backup()
    assert backup_path.exists(), "Snapshot backup DB harus berhasil dibuat"
