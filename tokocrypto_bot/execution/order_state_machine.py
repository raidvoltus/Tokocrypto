"""

MODULE: tokocrypto_bot.execution.order_state_machine
DESCRIPTION: Deterministic Order State Machine & Idempotency Key Generator for Tokocrypto.
"""

import hashlib
import logging
from enum import Enum
from typing import Set, Dict, Optional

logger = logging.getLogger("NVRA.OrderStateMachine")

class OrderStatus(str, Enum):
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    UNKNOWN = "UNKNOWN"
    RECONCILING = "RECONCILING"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class InvalidStateTransitionException(Exception):
    """Exception khusus jika terjadi transisi state ilegal."""
    pass

class OrderStateMachine:
    # Definisi Matriks Transisi Valid
    VALID_TRANSITIONS: Dict[OrderStatus, Set[OrderStatus]] = {
        OrderStatus.CREATED: {OrderStatus.SUBMITTING, OrderStatus.CANCELED, OrderStatus.REJECTED},
        OrderStatus.SUBMITTING: {OrderStatus.ACKNOWLEDGED, OrderStatus.UNKNOWN, OrderStatus.REJECTED},
        OrderStatus.ACKNOWLEDGED: {
            OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
            OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.UNKNOWN
        },
        OrderStatus.UNKNOWN: {OrderStatus.RECONCILING},
        OrderStatus.RECONCILING: {
            OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
            OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.UNKNOWN
        },
        OrderStatus.NEW: {
            OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED,
            OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.UNKNOWN
        },
        OrderStatus.PARTIALLY_FILLED: {
            OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.UNKNOWN
        },
        # Terminal states: Tidak boleh berpindah lagi kecuali diaudit ulang oleh reconciler
        OrderStatus.FILLED: set(),
        OrderStatus.CANCELED: set(),
        OrderStatus.REJECTED: set(),
        OrderStatus.EXPIRED: set(),
    }

    @classmethod
    def validate_transition(cls, current_state: OrderStatus, target_state: OrderStatus) -> bool:
        """Memvalidasi apakah transisi dari current_state ke target_state diizinkan."""
        if current_state == target_state:
            return True

        allowed = cls.VALID_TRANSITIONS.get(current_state, set())
        if target_state not in allowed:
            err_msg = f"Illegal order state transition attempt: [{current_state.value}] -> [{target_state.value}]"
            logger.error(err_msg)
            raise InvalidStateTransitionException(err_msg)
        return True

    @staticmethod
    def generate_client_order_id(execution_id: str, signal_id: str, symbol: str, side: str) -> str:
        """
        Menghasilkan clientOrderId deterministik yang mematuhi constraint Tokocrypto/Binance:
        - Maksimal 36 Karakter.
        - Karakter diizinkan: Alfanumerik, hyphen (-), underscore (_).
        """
        raw_seed = f"{execution_id}:{signal_id}:{symbol}:{side}"
        seed_hash = hashlib.sha256(raw_seed.encode('utf-8')).hexdigest()[:12].upper()
        
        clean_symbol = symbol.replace("_", "").replace("-", "")[:6]
        side_code = "B" if side.upper() == "BUY" else "S"

        # Format: QBOT-{hash12}-{symbol}-{side} -> Misal: QBOT-A1B2C3D4E5F6-BTCUSD-B (Total 26 Karakter <= 36)
        client_order_id = f"QBOT-{seed_hash}-{clean_symbol}-{side_code}"

        if len(client_order_id) > 36:
            client_order_id = client_order_id[:36]

        return client_order_id

