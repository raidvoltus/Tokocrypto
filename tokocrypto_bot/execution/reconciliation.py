"""
MODULE: tokocrypto_bot.execution.reconciliation
DESCRIPTION: Hardened Reconciliation Engine with Partial-Fill Precision & Fill Deduplication.
"""

import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Protocol

from tokocrypto_bot.execution.order_state_machine import OrderStateMachine, OrderStatus, OrderSide
from tokocrypto_bot.persistence.state_manager import StateManager
from tokocrypto_bot.persistence.database import get_db_transaction
from tokocrypto_bot.persistence.lifecycle_state import LifecycleManager, ApplicationState

logger = logging.getLogger("NVRA.ReconciliationEngine")


class ReconciliationDecision(str, Enum):
    FOUND_NEW = "FOUND_NEW"
    FOUND_PARTIALLY_FILLED = "FOUND_PARTIALLY_FILLED"
    FOUND_FILLED = "FOUND_FILLED"
    FOUND_CANCELED = "FOUND_CANCELED"
    FOUND_REJECTED = "FOUND_REJECTED"
    FOUND_EXPIRED = "FOUND_EXPIRED"
    NOT_FOUND = "NOT_FOUND"
    API_ERROR = "API_ERROR"


@dataclass(frozen=True)
class ReconciliationResult:
    client_order_id: str
    decision: ReconciliationDecision
    target_order_status: OrderStatus
    exchange_order_id: Optional[str] = None
    executed_qty: float = 0.0
    remaining_qty: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    fee_asset: Optional[str] = None
    reason: str = ""
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)


class ExchangeAdapterProtocol(Protocol):
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]: ...
    def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]: ...
    def fetch_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]: ...
    def fetch_account_balances(self) -> Dict[str, Dict[str, float]]: ...


class HardenedReconciliationEngine:
    def __init__(self, state_manager: StateManager, lifecycle_manager: LifecycleManager, exchange_adapter: ExchangeAdapterProtocol):
        self.state_mgr = state_manager
        self.lifecycle_mgr = lifecycle_manager
        self.exchange = exchange_adapter

    def execute_foundation_gate_reconciliation(() -> bool:
        """
        Gerbang Utama (Foundation Gate):
        Audit menyeluruh Unresolved Orders + Balances + Positions sebelum Trading diizinkan.
        """
        self.lifecycle_mgr.set_state(ApplicationState.RECONCILING, "Starting Foundation Gate Reconciliation")
        
        # 1. Audit Orders State
        unresolved = self.state_mgr.get_unresolved_orders()
        for order in unresolved:
            res = self.reconcile_single_order(order)
            success = self._apply_reconciliation_result(order, res)
            if not success:
                logger.critical(f"Order [{order['client_order_id']}] failed reconciliation gate!")
                self.lifecycle_mgr.set_state(ApplicationState.SAFE_MODE, "Unresolved Order found during Reconciliation Gate")
                return False

        # 2. Audit Balances & Positions Consistency
        if not self._reconcile_account_and_positions():
            self.lifecycle_mgr.set_state(ApplicationState.SAFE_MODE, "Balance/Position discrepancy detected")
            return False

        self.lifecycle_mgr.set_state(ApplicationState.READY, "Foundation Gate Reconciliation PASSED")
        return True

    def reconcile_single_order(self, order: Dict[str, Any]) -> ReconciliationResult:
        cid = order["client_order_id"]
        symbol = order["symbol"]
        curr_status = order["status"]

        if curr_status == OrderStatus.SUBMITTING.value:
            self.state_mgr.transition_order_state(cid, OrderStatus.SUBMITTING.value, OrderStatus.UNKNOWN.value, "UPCAST_SUBMITTING_ON_RECOVERY")
            curr_status = OrderStatus.UNKNOWN.value

        if curr_status == OrderStatus.UNKNOWN.value:
            self.state_mgr.transition_order_state(cid, OrderStatus.UNKNOWN.value, OrderStatus.RECONCILING.value, "RECONCILING_START")
            curr_status = OrderStatus.RECONCILING.value

        try:
            # Hierarki 1: Open Orders
            open_orders = self.exchange.fetch_open_orders(symbol=symbol)
            match_open = next((o for o in open_orders if o.get("clientOrderId") == cid), None)
            if match_open:
                return self._parse_exchange_response(cid, order, match_open, "OPEN_ORDERS_API")

            # Hierarki 2: Direct Lookup
            ex_order = self.exchange.fetch_order_by_client_id(symbol=symbol, client_order_id=cid)
            if ex_order:
                return self._parse_exchange_response(cid, order, ex_order, "ORDER_DETAIL_API")

            # Hierarki 3: Recent Trades
            trades = self.exchange.fetch_recent_trades(symbol=symbol, limit=50)
            matching_trades = [t for t in trades if t.get("clientOrderId") == cid]
            if matching_trades:
                return self._aggregate_trades(cid, order, matching_trades)

            # Rule: NOT_FOUND -> Preserves UNKNOWN (NOT REJECTED!)
            logger.warning(f"Order [{cid}] NOT FOUND in exchange. Retaining UNKNOWN state.")
            return ReconciliationResult(
                client_order_id=cid,
                decision=ReconciliationDecision.NOT_FOUND,
                target_order_status=OrderStatus.UNKNOWN,
                reason="Order not found in exchange. Preserving UNKNOWN status to prevent duplicate execution."
            )

        except Exception as e:
            logger.error(f"Exchange API error reconciling [{cid}]: {e}")
            return ReconciliationResult(
                client_order_id=cid,
                decision=ReconciliationDecision.API_ERROR,
                target_order_status=OrderStatus.UNKNOWN,
                reason=f"API Error: {str(e)}"
            )

    def _parse_exchange_response(self, cid: str, local_order: Dict[str, Any], ex_order: Dict[str, Any], source: str) -> ReconciliationResult:
        ex_status = ex_order.get("status", "").upper()
        ex_id = str(ex_order.get("orderId", ""))
        orig_qty = float(ex_order.get("origQty", local_order["quantity"]))
        executed_qty = float(ex_order.get("executedQty", 0.0))
        remaining_qty = max(0.0, orig_qty - executed_qty)
        price = float(ex_order.get("price", 0.0))
        cum_quote = float(ex_order.get("cummulativeQuoteQty", 0.0))
        avg_price = cum_quote / executed_qty if executed_qty > 0 else price

        # Precision Partial-Fill Evaluation
        if ex_status == "PARTIALLY_FILLED" or (0 < executed_qty < orig_qty and ex_status not in ("CANCELED", "EXPIRED")):
            decision = ReconciliationDecision.FOUND_PARTIALLY_FILLED
            target_status = OrderStatus.PARTIALLY_FILLED
        elif ex_status == "FILLED" or (executed_qty >= orig_qty and orig_qty > 0):
            decision = ReconciliationDecision.FOUND_FILLED
            target_status = OrderStatus.FILLED
        elif ex_status == "CANCELED":
            decision = ReconciliationDecision.FOUND_CANCELED
            target_status = OrderStatus.CANCELED
        elif ex_status == "REJECTED":
            decision = ReconciliationDecision.FOUND_REJECTED
            target_status = OrderStatus.REJECTED
        elif ex_status == "EXPIRED":
            decision = ReconciliationDecision.FOUND_EXPIRED
            target_status = OrderStatus.EXPIRED
        else:
            decision = ReconciliationDecision.FOUND_NEW
            target_status = OrderStatus.NEW

        return ReconciliationResult(
            client_order_id=cid,
            decision=decision,
            target_order_status=target_status,
            exchange_order_id=ex_id,
            executed_qty=executed_qty,
            remaining_qty=remaining_qty,
            avg_price=avg_price,
            reason=f"Matched via {source} as {ex_status}",
            raw_response=ex_order
        )

    def _apply_reconciliation_result(self, local_order: Dict[str, Any], result: ReconciliationResult) -> bool:
        cid = result.client_order_id
        if result.decision in (ReconciliationDecision.NOT_FOUND, ReconciliationDecision.API_ERROR):
            return False

        curr_status = local_order["status"]
        if curr_status in (OrderStatus.SUBMITTING.value, OrderStatus.UNKNOWN.value):
            curr_status = OrderStatus.RECONCILING.value

        try:
            OrderStateMachine.validate_transition(OrderStatus(curr_status), result.target_order_status)
            
            # Idempotent Fills Registration (Deduplicated by fill_id / exchange trade_id)
            if result.executed_qty > 0:
                self._record_fill_idempotent(local_order, result)

            return self.state_mgr.transition_order_state(
                client_order_id=cid,
                previous_status=curr_status,
                new_status=result.target_order_status.value,
                trigger=f"RECONCILED_{result.decision.value}",
                details={"executed_qty": result.executed_qty, "remaining_qty": result.remaining_qty, "avg_price": result.avg_price},
                exchange_order_id=result.exchange_order_id
            )
        except Exception as e:
            logger.error(f"Transition error for [{cid}]: {e}")
            return False

    def _record_fill_idempotent(self, local_order: Dict[str, Any], result: ReconciliationResult) -> None:
        """Deduplication fill menggunakan UNIQUE constraint / INSERT OR IGNORE."""
        fill_id = f"FILL-{result.exchange_order_id or result.client_order_id}-{result.executed_qty}"
        now_str = datetime.now(timezone.utc).isoformat()
        
        with get_db_transaction(self.state_mgr.db) as conn:
            conn.execute(
                """
                INSERT INTO fills (fill_id, client_order_id, exchange_order_id, symbol, side, price, quantity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fill_id) DO NOTHING;
                """,
                (fill_id, result.client_order_id, result.exchange_order_id, local_order["symbol"], local_order["side"], result.avg_price, result.executed_qty, now_str)
            )

    def _reconcile_account_and_positions(self) -> bool:
        """Verifikasi saldo dan posisi lokal terhadap Exchange."""
        try:
            balances = self.exchange.fetch_account_balances()
            now_str = datetime.now(timezone.utc).isoformat()
            
            with get_db_transaction(self.state_mgr.db) as conn:
                for asset, data in balances.items():
                    conn.execute(
                        """
                        INSERT INTO balances (asset, free, locked, updated_at) VALUES (?, ?, ?, ?)
                        ON CONFLICT(asset) DO UPDATE SET free=excluded.free, locked=excluded.locked, updated_at=excluded.updated_at
                        """,
                        (asset, data["free"], data["locked"], now_str)
                    )
            return True
        except Exception as e:
            logger.error(f"Account balance reconciliation failed: {e}")
            return False
