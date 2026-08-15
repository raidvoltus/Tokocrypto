"""
MODULE: tokocrypto_bot.application
DESCRIPTION: P1-F Autonomous Worker Loop Orchestrator with Strict Cycle Isolation, Paper/Live Modes & P0-C Reconciliation.
"""

import sys
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Import Sub-Sistem P0 (Persistence, Recovery, State Machine)
from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.state_manager import StateManager
from tokocrypto_bot.persistence.lifecycle_state import ApplicationState
from tokocrypto_bot.execution.order_state_machine import OrderStateMachine, OrderStatus
from tokocrypto_bot.execution.reconciliation import HardenedReconciliationEngine
from tokocrypto_bot.recovery.startup_recovery import StartupRecoveryOrchestrator

# Import Sub-Sistem P1 (Universe, Market, Features, ML, Decision, Portfolio)
from tokocrypto_bot.strategy.pair_universe import PairUniverseEngine, PairUniverseConfig
from tokocrypto_bot.strategy.market_data import MarketDataEngine
from tokocrypto_bot.strategy.features import FeatureEngine, FeatureFrame
from tokocrypto_bot.ml.inference import MLInferenceEngine, PredictionResult
from tokocrypto_bot.strategy.selector import AdaptiveStrategySelector
from tokocrypto_bot.strategy.decision import DecisionEngine, Decision, DecisionAction
from tokocrypto_bot.strategy.portfolio import (
    RiskGate, PositionSizer, MultiPairPortfolioRanker, PortfolioState,
    PositionPlan, RiskDecision, RiskAction
)
from tokocrypto_bot.exchange.tokocrypto_client import TokocryptoDirectClient

logger = logging.getLogger("NVRA.Application")


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    LIVE = "LIVE"


class PaperExchangeAdapter:
    """Mock/Paper Adapter untuk simulasi eksekusi tanpa risiko kapital fisik."""

    def __init__(self, initial_balance_usdt: float = 10000.0):
        self.balance_usdt = initial_balance_usdt
        self.paper_positions: Dict[str, float] = {}
        self.paper_orders: Dict[str, Dict[str, Any]] = {}

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if symbol:
            return [o for o in self.paper_orders.values() if o["symbol"] == symbol and o["status"] == "NEW"]
        return [o for o in self.paper_orders.values() if o["status"] == "NEW"]

    def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        return self.paper_orders.get(client_order_id)

    def fetch_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        return []

    def fetch_account_balances(self) -> Dict[str, Dict[str, float]]:
        return {"USDT": {"free": self.balance_usdt, "locked": 0.0}}

    def simulate_paper_order(self, plan: PositionPlan) -> Dict[str, Any]:
        """Simulasi pengisian instant untuk PAPER mode."""
        ex_id = f"PAPER-EX-{int(time.time() * 1000)}"
        order_dict = {
            "orderId": ex_id,
            "clientOrderId": plan.risk_decision.symbol,
            "symbol": plan.symbol,
            "status": "FILLED",
            "price": str(plan.target_price),
            "origQty": str(plan.calculated_quantity),
            "executedQty": str(plan.calculated_quantity),
            "cummulativeQuoteQty": str(plan.approved_notional_usdt)
        }
        self.paper_orders[plan.risk_decision.symbol] = order_dict

        if plan.action == DecisionAction.BUY:
            self.balance_usdt -= plan.approved_notional_usdt
            self.paper_positions[plan.symbol] = self.paper_positions.get(plan.symbol, 0.0) + plan.approved_notional_usdt
        elif plan.action == DecisionAction.SELL:
            self.balance_usdt += plan.approved_notional_usdt
            self.paper_positions[plan.symbol] = max(0.0, self.paper_positions.get(plan.symbol, 0.0) - plan.approved_notional_usdt)

        return order_dict


class AutonomousTradingWorker:
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        db_path: Optional[str] = None,
        api_key: str = "",
        api_secret: str = ""
    ):
        self.mode = mode
        self.db_mgr = DatabaseManager(db_path=db_path)
        self.state_mgr = StateManager(self.db_mgr)

        # 1. Setup Exchange Adapter berdasarkan Mode
        if self.mode == ExecutionMode.LIVE:
            logger.warning("INITIALIZING LIVE EXECUTION MODE (Real Capital at risk!).")
            self.exchange = TokocryptoDirectClient(api_key, api_secret)
        else:
            logger.info(f"INITIALIZING {self.mode.value} EXECUTION MODE (Paper Simulator Active).")
            self.exchange = PaperExchangeAdapter()

        # 2. Infrastructure & Recovery Initialization (P0)
        self.orchestrator = StartupRecoveryOrchestrator(self.db_mgr, self.exchange)
        self.reconciler = HardenedReconciliationEngine(
            self.state_mgr, self.orchestrator.lifecycle_mgr, self.exchange
        )

        # 3. Quantitative & ML Engines Initialization (P1)
        self.universe_engine = PairUniverseEngine()
        self.market_data_engine = MarketDataEngine()
        self.feature_engine = FeatureEngine()
        self.ml_engine = MLInferenceEngine()
        self.strategy_selector = AdaptiveStrategySelector()
        self.decision_engine = DecisionEngine()
        self.risk_gate = RiskGate()
        self.position_sizer = PositionSizer()

    def run_worker_loop(self, poll_interval_sec: float = 10.0) -> None:
        """Entry point utama worker loop otonom."""
        logger.info(f"Starting NVRA Autonomous Trading Worker Loop [{self.mode.value} MODE]...")

        # Step A: Boot Verification & P0-D Startup Recovery Gate
        boot_state = self.orchestrator.run_startup_recovery_gate()
        if boot_state != ApplicationState.READY:
            logger.critical(f"Startup Recovery Gate did NOT yield READY (Got: {boot_state.value}). Loop Halted.")
            return

        # Main Scan & Execution Cycle Loop
        try:
            while True:
                cycle_start_time = time.time()
                self._run_single_autonomous_cycle()

                # Write Heartbeat to Database
                elapsed = time.time() - cycle_start_time
                self.orchestrator.lifecycle_mgr.write_heartbeat({
                    "cycle_duration_sec": round(elapsed, 2),
                    "execution_mode": self.mode.value
                })

                time.sleep(poll_interval_sec)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received via KeyboardInterrupt.")
            self.orchestrator.shutdown_mgr.execute_graceful_shutdown("KeyboardInterrupt")

    def _run_single_autonomous_cycle(self) -> None:
        """Eksekusi 1 siklus pemindaian multi-pair end-to-end terisolasi."""
        lifecycle = self.orchestrator.lifecycle_mgr
        current_app_state = lifecycle.current_state

        if current_app_state in (ApplicationState.SAFE_MODE, ApplicationState.PAUSED):
            logger.warning(f"Worker cycle skipped: Application is in [{current_app_state.value}] state.")
            return

        lifecycle.set_state(ApplicationState.TRADING, "Beginning Multi-Pair Autonomous Cycle")

        # 1. SCANNING: Dynamic Pair Discovery
        active_universe = self.universe_engine.get_active_universe()
        if not active_universe:
            logger.warning("Active pair universe is empty. Cycle finished.")
            return

        # 2. ANALYZING: Fetch Market Data, Features, ML Inference & Strategy Selection
        feature_frames: Dict[str, FeatureFrame] = {}
        predictions: Dict[str, PredictionResult] = {}
        candidate_decisions: Dict[str, Decision] = {}
        current_prices: Dict[str, float] = {}

        for rule in active_universe:
            symbol = rule.symbol
            try:
                # Isolated Pair Execution Guard
                klines_df = self.market_data_engine.get_klines_dataframe(symbol, interval="5m", limit=210)
                if klines_df.empty:
                    continue

                ff = self.feature_engine.compute_features(klines_df, symbol)
                feature_frames[symbol] = ff
                if not ff.is_valid:
                    continue

                current_prices[symbol] = klines_df["close"].iloc[-1]

                pred = self.ml_engine.predict(ff)
                predictions[symbol] = pred

                # Adaptive Strategy Selection & Signal Generation
                candidate_sig, score, regime_ctx = self.strategy_selector.select_best_signal(symbol, ff, pred)
                
                # Evaluate Candidate into Decision
                decision = self.decision_engine.evaluate(symbol, ff, pred, current_price=current_prices[symbol])
                candidate_decisions[symbol] = decision

            except Exception as e:
                logger.error(f"Isolated Exception processing symbol [{symbol}]: {e}", exc_info=True)

        # 3. RISK_CHECK & POSITION SIZING
        portfolio_state = self._build_portfolio_state(current_app_state)
        candidate_plans: Dict[str, PositionPlan] = {}

        for symbol, decision in candidate_decisions.items():
            if decision.action not in (DecisionAction.BUY, DecisionAction.SELL):
                continue

            # Deduplication Check: Skip jika order/posisi sudah ada
            if self._has_unresolved_or_active_position(symbol, portfolio_state):
                logger.info(f"Deduplication Guard: Order/Position already active for [{symbol}]. Skipping.")
                continue

            ff = feature_frames[symbol]
            price = current_prices[symbol]

            risk_dec = self.risk_gate.evaluate_trade_risk(decision, ff, portfolio_state, current_price=price)
            plan = self.position_sizer.create_position_plan(decision, risk_dec, current_price=price)

            if plan.action in (DecisionAction.BUY, DecisionAction.SELL) and plan.approved_notional_usdt > 0:
                candidate_plans[symbol] = plan

        # Portfolio Multi-Pair Ranking & Capital Allocation
        approved_plans = MultiPairPortfolioRanker.rank_and_filter_plans(
            candidate_plans, portfolio_state.available_balance_usdt
        )

        # 4. EXECUTING & RECONCILING
        if approved_plans:
            lifecycle.set_state(ApplicationState.RECONCILING, f"Executing {len(approved_plans)} approved position plans")
            for plan in approved_plans:
                self._execute_single_position_plan(plan)

        # 5. POST-CYCLE RECONCILIATION & PERSISTENCE COMMIT
        self.reconciler.execute_foundation_gate_reconciliation()
        lifecycle.set_state(ApplicationState.READY, "Cycle execution complete. System READY.")

    def _execute_single_position_plan(self, plan: PositionPlan) -> None:
        """Mengeksekusi satu PositionPlan melalui OrderStateMachine & StateManager."""
        symbol = plan.symbol
        side = plan.action.value
        execution_id = f"EXEC-{int(time.time() * 1000)}"
        signal_id = f"SIG-{symbol}-{int(time.time())}"

        client_order_id = OrderStateMachine.generate_client_order_id(execution_id, signal_id, symbol, side)

        # P0-B: Persist Order Intent CREATED
        created_ok = self.state_mgr.create_order_intent(
            client_order_id=client_order_id,
            execution_id=execution_id,
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            order_type="LIMIT",
            price=plan.target_price,
            quantity=plan.calculated_quantity,
            initial_status=OrderStatus.CREATED.value
        )
        if not created_ok:
            return

        # P0-A: Transition to SUBMITTING before Network Request
        OrderStateMachine.validate_transition(OrderStatus.CREATED, OrderStatus.SUBMITTING)
        self.state_mgr.transition_order_state(client_order_id, "CREATED", "SUBMITTING", "NETWORK_REQUEST_INITIATED")

        # Network Transmit
        try:
            if self.mode == ExecutionMode.LIVE:
                # Call Real Tokocrypto Direct Client (NO BLIND RETRY ON POST)
                live_client: TokocryptoDirectClient = self.exchange
                res = live_client.post_order_non_retry(
                    symbol=symbol, side=side, order_type="LIMIT",
                    quantity=plan.calculated_quantity, price=plan.target_price,
                    client_order_id=client_order_id
                )
                ex_id = str(res.get("orderId", ""))
                self.state_mgr.transition_order_state(client_order_id, "SUBMITTING", "ACKNOWLEDGED", "HTTP_POST_200_OK", exchange_order_id=ex_id)
            else:
                # Simulate Paper Order
                paper_adapter: PaperExchangeAdapter = self.exchange
                res = paper_adapter.simulate_paper_order(plan)
                ex_id = res["orderId"]
                self.state_mgr.transition_order_state(client_order_id, "SUBMITTING", "FILLED", "PAPER_SIMULATION_FILLED", exchange_order_id=ex_id)

        except Exception as e:
            # Network Timeout / Unknown Handling: NEVER BLIND RETRY!
            logger.error(f"Network error submitting order [{client_order_id}]: {e}. Marking state as UNKNOWN.")
            OrderStateMachine.validate_transition(OrderStatus.SUBMITTING, OrderStatus.UNKNOWN)
            self.state_mgr.transition_order_state(client_order_id, "SUBMITTING", "UNKNOWN", f"NETWORK_EXCEPTION_{type(e).__name__}")

    def _build_portfolio_state(self, app_state: ApplicationState) -> PortfolioState:
        """Constructs PortfolioState object from persistent DB and Exchange balances."""
        try:
            balances = self.exchange.fetch_account_balances()
            usdt_free = balances.get("USDT", {}).get("free", 0.0)
        except Exception:
            usdt_free = 0.0

        unresolved = self.state_mgr.get_unresolved_orders()
        is_reconciled = len(unresolved) == 0

        return PortfolioState(
            total_equity_usdt=usdt_free,
            available_balance_usdt=usdt_free,
            current_portfolio_exposure_usdt=0.0,
            daily_realized_pnl_usdt=0.0,
            peak_equity_usdt=usdt_free,
            current_drawdown_pct=0.0,
            cusum_statistic=0.0,
            consecutive_losses=0,
            active_positions={},
            app_lifecycle_state=app_state,
            is_reconciliation_clean=is_reconciled,
            is_kill_switch_active=False
        )

    def _has_unresolved_or_active_position(self, symbol: str, p_state: PortfolioState) -> bool:
        """Memeriksa apakah simbol memiliki order gantung atau posisi aktif."""
        unresolved = self.state_mgr.get_unresolved_orders()
        if any(o["symbol"] == symbol for o in unresolved):
            return True
        return symbol in p_state.active_positions


if __name__ == "__main__":
    worker = AutonomousTradingWorker(mode=ExecutionMode.PAPER)
    worker.run_worker_loop()
