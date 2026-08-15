"""
MODULE: tokocrypto_bot.execution.reconciliation
DESCRIPTION: Full Exchange Reconciliation Engine acting as Recovery Authority for NVRA Trading Engine.
COMPATIBILITY: P0-A (OrderStateMachine) & P0-B (StateManager, SQLite WAL)
"""

import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Protocol

from tokocrypto_bot.execution.order_state_machine import OrderStateMachine, OrderStatus
from tokocrypto_bot.persistence.state_manager import StateManager
from tokocrypto_bot.persistence.database import get_db_transaction

logger = logging.getLogger("NVRA.ReconciliationEngine")


class SystemRecoveryStatus(str, Enum):
    RECOVERY_COMPLETE = "RECOVERY_COMPLETE"
    RECOVERY_INCOMPLETE = "RECOVERY_INCOMPLETE"
    SAFE_MODE = "SAFE_MODE"


class ReconciliationDecision(str, Enum):
    FOUND_NEW = "FOUND_NEW"
    FOUND_PARTIALLY_FILLED = "FOUND_PARTIALLY_FILLED"
    FOUND_FILLED = "FOUND_FILLED"
    FOUND_CANCELED = "FOUND_CANCELED"
    FOUND_REJECTED = "FOUND_REJECTED"
    FOUND_EXPIRED = "FOUND_EXPIRED"
    NOT_FOUND = "NOT_FOUND"
    STILL_UNKNOWN = "STILL_UNKNOWN"
    API_ERROR = "API_ERROR"


@dataclass(frozen=True)
class ReconciliationResult:
    client_order_id: str
    decision: ReconciliationDecision
    target_order_status: Optional[OrderStatus]
    exchange_order_id: Optional[str] = None
    executed_qty: float = 0.0
    remaining_qty: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    fee_asset: Optional[str] = None
    reason: str = ""
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)


class ExchangeAdapterProtocol(Protocol):
    """Protocol Interface untuk memisahkan call API Exchange dari Reconciliation Engine."""
    
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        ...

    def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        ...

    def fetch_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        ...

    def fetch_account_balances(self) -> Dict[str, Dict[str, float]]:
        ...


class ReconciliationEngine:
    """
    Otoritas Pemulihan State (Recovery Authority).
    Menjalankan audit silang hierarkis antara DB lokal dan Tokocrypto Exchange API.
    """

    def __init__(self, state_manager: StateManager, exchange_adapter: ExchangeAdapterProtocol):
        self.state_mgr = state_manager
        self.exchange = exchange_adapter

    def reconcile_all_unresolved_orders(self, execution_id: Optional[str] = None) -> SystemRecoveryStatus:
        """
        Entry point utama startup/network recovery.
        Memproses seluruh order yang berstatus CREATED, SUBMITTING, UNKNOWN, NEW, PARTIALLY_FILLED.
        """
        logger.info("Initiating Full Exchange Reconciliation cycle...")
        unresolved_orders = self.state_mgr.get_unresolved_orders()

        if execution_id:
            unresolved_orders = [o for o in unresolved_orders if o.get("execution_id") == execution_id]

        if not unresolved_orders:
            logger.info("No unresolved orders found. Reconciliation COMPLETE.")
            return SystemRecoveryStatus.RECOVERY_COMPLETE

        logger.warning(f"Found {len(unresolved_orders)} unresolved orders to reconcile.")
        has_failure = False

        for order in unresolved_orders:
            result = self.reconcile_single_order(order)
            success = self._apply_reconciliation_result(order, result)
            if not success:
                has_failure = True

        if has_failure:
            logger.critical("Reconciliation finished with unresolved state! Entering SAFE_MODE.")
            return SystemRecoveryStatus.SAFE_MODE

        logger.info("All unresolved orders successfully reconciled. Recovery COMPLETE.")
        return SystemRecoveryStatus.RECOVERY_COMPLETE

    def reconcile_single_order(self, order: Dict[str, Any]) -> ReconciliationResult:
        """
        Mengeksekusi Hierarki Verification Query 4 tingkat:
        1. Query openOrders
        2. Query order detail by clientOrderId / allOrders
        3. Query recent trades / fills
        4. Decision generation (Melarang keras NOT_FOUND auto-REJECTED)
        """
        client_order_id = order["client_order_id"]
        symbol = order["symbol"]
        current_status_str = order["status"]

        logger.info(f"Reconciling order [{client_order_id}] (Current local status: {current_status_str})")

        # Step 0: Jika status lokal SUBMITTING saat recovery, transisikan dulu ke RECONCILING via UNKNOWN
        if current_status_str == OrderStatus.SUBMITTING.value:
            self._transition_local_state(
                client_order_id, OrderStatus.SUBMITTING.value, OrderStatus.UNKNOWN.value,
                "UPCAST_SUBMITTING_TO_UNKNOWN_ON_RECOVERY"
            )
            current_status_str = OrderStatus.UNKNOWN.value

        if current_status_str == OrderStatus.UNKNOWN.value:
            self._transition_local_state(
                client_order_id, OrderStatus.UNKNOWN.value, OrderStatus.RECONCILING.value,
                "RECONCILIATION_AUDIT_INITIATED"
            )
            current_status_str = OrderStatus.RECONCILING.value

        try:
            # HIERARKI 1: Open Orders Check
            open_orders = self.exchange.fetch_open_orders(symbol=symbol)
            matching_open = next((o for o in open_orders if o.get("clientOrderId") == client_order_id), None)

            if matching_open:
                return self._parse_exchange_order_response(client_order_id, matching_open, "OPEN_ORDERS_API")

            # HIERARKI 2: Direct Order Lookup by clientOrderId / Order History
            ex_order = self.exchange.fetch_order_by_client_id(symbol=symbol, client_order_id=client_order_id)
            if ex_order:
                # Verifikasi tambahan via trades jika order FILLED / PARTIALLY_FILLED
                trades_info = self._enrich_fill_details_if_needed(symbol, ex_order)
                return self._parse_exchange_order_response(client_order_id, ex_order, "ORDER_DETAIL_API", trades_info)

            # HIERARKI 3: Trade History Scanning
            trades = self.exchange.fetch_recent_trades(symbol=symbol, limit=50)
            matching_trades = [t for t in trades if t.get("clientOrderId") == client_order_id]
            if matching_trades:
                return self._aggregate_trades_to_result(client_order_id, symbol, matching_trades)

            # HIERARKI 4: NOT FOUND HANDLING
            # CRITICAL RULE: NOT_FOUND tidak boleh langsung diterjemahkan menjadi REJECTED!
            logger.warning(
                f"Order [{client_order_id}] NOT FOUND across open orders, order history, and recent trades. "
                "Maintaining UNKNOWN state for security."
            )
            return ReconciliationResult(
                client_order_id=client_order_id,
                decision=ReconciliationDecision.NOT_FOUND,
                target_order_status=OrderStatus.UNKNOWN,
                reason="Order not found in exchange active/historical queries. Preserving UNKNOWN status to prevent duplicate execution."
            )

        except Exception as e:
            logger.error(f"API Error during reconciliation of [{client_order_id}]: {e}", exc_info=True)
            return ReconciliationResult(
                client_order_id=client_order_id,
                decision=ReconciliationDecision.API_ERROR,
                target_order_status=OrderStatus.UNKNOWN,
                reason=f"Exchange API request failed: {str(e)}"
            )

    def _apply_reconciliation_result(self, local_order: Dict[str, Any], result: ReconciliationResult) -> bool:
        """
        Mengeksekusi transisi state transaksional di StateManager dan mencatat audit trail di reconciliation_events.
        """
        client_order_id = result.client_order_id
        curr_status = local_order["status"]
        if curr_status == OrderStatus.SUBMITTING.value:
            curr_status = OrderStatus.UNKNOWN.value

        # Atomic Audit Trail Recording
        self._log_reconciliation_event(local_order.get("execution_id", "UNKNOWN"), result)

        if result.decision in (ReconciliationDecision.NOT_FOUND, ReconciliationDecision.API_ERROR, ReconciliationDecision.STILL_UNKNOWN):
            logger.error(f"Reconciliation unresolved for [{client_order_id}]. Status remains UNKNOWN.")
            return False

        target_status = result.target_order_status
        if not target_status:
            return False

        # Validate & Execute State Transition
        try:
            # Transisi dari RECONCILING ke Target Status (FILLED, NEW, CANCELED, dll.)
            if curr_status == OrderStatus.RECONCILING.value or curr_status == local_order["status"]:
                OrderStateMachine.validate_transition(OrderStatus(curr_status), target_status)

            details = {
                "decision": result.decision.value,
                "executed_qty": result.executed_qty,
                "remaining_qty": result.remaining_qty,
                "avg_price": result.avg_price,
                "fee": result.fee,
                "fee_asset": result.fee_asset,
                "reason": result.reason
            }

            success = self.state_mgr.transition_order_state(
                client_order_id=client_order_id,
                previous_status=curr_status,
                new_status=target_status.value,
                trigger=f"RECONCILIATION_DECISION_{result.decision.value}",
                details=details,
                exchange_order_id=result.exchange_order_id
            )

            if success and result.executed_qty > 0:
                self._record_fills_and_position(local_order, result)

            return success

        except Exception as e:
            logger.error(f"Failed to apply state transition for [{client_order_id}]: {e}")
            return False

    def _parse_exchange_order_response(
        self,
        client_order_id: str,
        ex_order: Dict[str, Any],
        source_api: str,
        trades_info: Optional[Dict[str, Any]] = None
    ) -> ReconciliationResult:
        """Penerjemah status mentah Exchange ke ReconciliationResult yang immutable."""
        ex_status = ex_order.get("status", "").upper()
        ex_id = str(ex_order.get("orderId", ""))
        executed_qty = float(ex_order.get("executedQty", 0.0))
        orig_qty = float(ex_order.get("origQty", 0.0))
        remaining_qty = max(0.0, orig_qty - executed_qty)
        price = float(ex_order.get("price", 0.0))

        if trades_info:
            executed_qty = trades_info.get("executed_qty", executed_qty)
            avg_price = trades_info.get("avg_price", price)
            fee = trades_info.get("fee", 0.0)
            fee_asset = trades_info.get("fee_asset")
        else:
            avg_price = float(ex_order.get("cummulativeQuoteQty", 0.0)) / executed_qty if executed_qty > 0 else price
            fee = 0.0
            fee_asset = None

        decision_map = {
            "NEW": (ReconciliationDecision.FOUND_NEW, OrderStatus.NEW),
            "PARTIALLY_FILLED": (ReconciliationDecision.FOUND_PARTIALLY_FILLED, OrderStatus.PARTIALLY_FILLED),
            "FILLED": (ReconciliationDecision.FOUND_FILLED, OrderStatus.FILLED),
            "CANCELED": (ReconciliationDecision.FOUND_CANCELED, OrderStatus.CANCELED),
            "REJECTED": (ReconciliationDecision.FOUND_REJECTED, OrderStatus.REJECTED),
            "EXPIRED": (ReconciliationDecision.FOUND_EXPIRED, OrderStatus.EXPIRED),
        }

        decision, target_status = decision_map.get(
            ex_status,
            (ReconciliationDecision.STILL_UNKNOWN, OrderStatus.UNKNOWN)
        )

        return ReconciliationResult(
            client_order_id=client_order_id,
            decision=decision,
            target_order_status=target_status,
            exchange_order_id=ex_id,
            executed_qty=executed_qty,
            remaining_qty=remaining_qty,
            avg_price=avg_price,
            fee=fee,
            fee_asset=fee_asset,
            reason=f"Matched via {source_api} with status {ex_status}",
            raw_response=ex_order
        )

    def _enrich_fill_details_if_needed(self, symbol: str, ex_order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Mengekstrak rincian executed trades & fee jika order sudah terisi sebagian/penuh."""
        client_order_id = ex_order.get("clientOrderId")
        if not client_order_id:
            return None

        trades = self.exchange.fetch_recent_trades(symbol=symbol, limit=50)
        matching = [t for t in trades if t.get("clientOrderId") == client_order_id or str(t.get("orderId")) == str(ex_order.get("orderId"))]

        if not matching:
            return None

        total_qty = sum(float(t.get("qty", 0.0)) for t in matching)
        total_quote = sum(float(t.get("qty", 0.0)) * float(t.get("price", 0.0)) for t in matching)
        total_fee = sum(float(t.get("commission", 0.0)) for t in matching)
        fee_asset = matching[0].get("commissionAsset") if matching else None

        avg_price = total_quote / total_qty if total_qty > 0 else float(ex_order.get("price", 0.0))

        return {
            "executed_qty": total_qty,
            "avg_price": avg_price,
            "fee": total_fee,
            "fee_asset": fee_asset
        }

    def _aggregate_trades_to_result(
        self,
        client_order_id: str,
        symbol: str,
        trades: List[Dict[str, Any]]
    ) -> ReconciliationResult:
        """Menggabungkan potongan trade history menjadi status order FILLED/PARTIALLY_FILLED terverifikasi."""
        total_qty = sum(float(t.get("qty", 0.0)) for t in trades)
        total_quote = sum(float(t.get("qty", 0.0)) * float(t.get("price", 0.0)) for t in trades)
        total_fee = sum(float(t.get("commission", 0.0)) for t in trades)
        fee_asset = trades[0].get("commissionAsset") if trades else None

        avg_price = total_quote / total_qty if total_qty > 0 else 0.0
        ex_id = str(trades[0].get("orderId")) if trades else None

        return ReconciliationResult(
            client_order_id=client_order_id,
            decision=ReconciliationDecision.FOUND_FILLED,
            target_order_status=OrderStatus.FILLED,
            exchange_order_id=ex_id,
            executed_qty=total_qty,
            remaining_qty=0.0,
            avg_price=avg_price,
            fee=total_fee,
            fee_asset=fee_asset,
            reason="Reconstructed fully from recent trade history fills",
            raw_response={"trades": trades}
        )

    def _transition_local_state(self, client_order_id: str, prev: str, target: str, trigger: str) -> None:
        """Helper transisi state internal."""
        OrderStateMachine.validate_transition(OrderStatus(prev), OrderStatus(target))
        self.state_mgr.transition_order_state(client_order_id, prev, target, trigger)

    def _log_reconciliation_event(self, execution_id: str, result: ReconciliationResult) -> None:
        """Mencatat event audit imutabel ke tabel reconciliation_events."""
        now_str = datetime.now(timezone.utc).isoformat()
        discrepancies = {
            "executed_qty": result.executed_qty,
            "remaining_qty": result.remaining_qty,
            "avg_price": result.avg_price,
            "raw_reason": result.reason
        }
        action_taken = {
            "decision": result.decision.value,
            "target_status": result.target_order_status.value if result.target_order_status else None,
            "exchange_order_id": result.exchange_order_id
        }

        with get_db_transaction(self.state_mgr.db) as conn:
            conn.execute(
                """
                INSERT INTO reconciliation_events (
                    reconciliation_id, trigger_source, status,
                    discrepancies_found_json, action_taken_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"REC-{result.client_order_id}",
                    "RECONCILIATION_ENGINE",
                    result.decision.value,
                    json.dumps(discrepancies),
                    json.dumps(action_taken),
                    now_str
                )
            )

    def _record_fills_and_position(self, local_order: Dict[str, Any], result: ReconciliationResult) -> None:
        """Memperbarui pencatatan fills dan saldo/posisi lokal pasca rekonsiliasi sukses."""
        now_str = datetime.now(timezone.utc).isoformat()
        with get_db_transaction(self.state_mgr.db) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO fills (
                    client_order_id, exchange_order_id, symbol, side,
                    price, quantity, fee, fee_asset, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.client_order_id,
                    result.exchange_order_id,
                    local_order["symbol"],
                    local_order["side"],
                    result.avg_price,
                    result.executed_qty,
                    result.fee,
                    result.fee_asset,
                    now_str
                )
            )
