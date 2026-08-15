"""
MODULE: tokocrypto_bot.strategy.portfolio
DESCRIPTION: Portfolio Risk Authority & Dynamic Position Sizing Engine (P1-E).
AUTHORITY: Absolute Safety Authority (Overrides ML, Strategy, Selector, & Gemini).
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set

from tokocrypto_bot.strategy.features import FeatureFrame
from tokocrypto_bot.strategy.decision import Decision, DecisionAction
from tokocrypto_bot.persistence.lifecycle_state import ApplicationState

logger = logging.getLogger("NVRA.PortfolioRisk")


class RiskState(str, Enum):
    NORMAL = "NORMAL"          # Full strategy & normal position sizing
    CAUTION = "CAUTION"        # Sizing dipotong 50%
    RESTRICTED = "RESTRICTED"   # Hanya sinyal kualitas tertinggi (High EV/Prob)
    HALTED = "HALTED"          # Blokir total seluruh order baru (Hanya manage posisi aktif)


class RiskAction(str, Enum):
    ALLOW = "ALLOW"            # Sizing penuh disetujui
    REDUCE = "REDUCE"          # Sizing dipotong akibat batasan kap/risiko
    REJECT = "REJECT"          # Order ditolak sepenuhnya


@dataclass(frozen=True)
class PortfolioState:
    total_equity_usdt: float
    available_balance_usdt: float
    current_portfolio_exposure_usdt: float
    daily_realized_pnl_usdt: float
    peak_equity_usdt: float
    current_drawdown_pct: float
    cusum_statistic: float
    consecutive_losses: int
    active_positions: Dict[str, float] = field(default_factory=dict)  # Symbol -> Notional Value
    app_lifecycle_state: ApplicationState = ApplicationState.READY
    is_reconciliation_clean: bool = True
    is_kill_switch_active: bool = False


@dataclass(frozen=True)
class RiskDecision:
    symbol: str
    action: RiskAction
    requested_notional: float
    approved_notional: float
    risk_per_trade_pct: float
    portfolio_exposure_pct: float
    stop_distance_pct: float
    max_loss_usdt: float
    risk_score: float
    risk_flags: List[str]
    reason_codes: List[str]


@dataclass(frozen=True)
class PositionPlan:
    symbol: str
    timestamp: int
    action: DecisionAction
    approved_notional_usdt: float
    calculated_quantity: float
    target_price: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    risk_decision: RiskDecision


@dataclass(frozen=True)
class PortfolioRiskConfig:
    max_portfolio_exposure_pct: float = 0.50   # Maksimal 50% total equity terpapar di crypto
    max_single_asset_exposure_pct: float = 0.15# Maksimal 15% equity per single asset
    base_risk_per_trade_pct: float = 0.01      # Base Risk 1.0% equity per trade
    atr_multiplier_stop: float = 2.0           # Stop loss distance = 2.0 * ATR
    max_daily_loss_pct: float = 0.04           # Daily loss limit 4% -> Trigger HALTED
    max_drawdown_halt_pct: float = 0.10        # Max drawdown 10% -> Trigger HALTED
    cusum_threshold: float = 5.0               # Threshold CUSUM anomaly
    max_consecutive_losses: int = 4            # Limit loss beruntun sebelum RESTRICTED
    min_notional_usdt: float = 10.0            # Tokocrypto minimum notional requirement


class RiskGate:
    """Otoritas Keselamatan Tunggal Sistem."""

    def __init__(self, config: Optional[PortfolioRiskConfig] = None):
        self.config = config or PortfolioRiskConfig()

    def evaluate_risk_state(self, p_state: PortfolioState) -> RiskState:
        """Evaluasi state machine risiko sistem berdasarkan telemetri portfolio."""
        if (
            p_state.app_lifecycle_state in (ApplicationState.SAFE_MODE, ApplicationState.PAUSED)
            or not p_state.is_reconciliation_clean
            or p_state.is_kill_switch_active
            or p_state.current_drawdown_pct >= self.config.max_drawdown_halt_pct
            or (p_state.total_equity_usdt > 0 and abs(min(0.0, p_state.daily_realized_pnl_usdt)) / p_state.total_equity_usdt >= self.config.max_daily_loss_pct)
            or p_state.cusum_statistic >= self.config.cusum_threshold
        ):
            return RiskState.HALTED

        if p_state.consecutive_losses >= self.config.max_consecutive_losses or p_state.current_drawdown_pct >= (self.config.max_drawdown_halt_pct * 0.6):
            return RiskState.RESTRICTED

        if p_state.current_drawdown_pct >= (self.config.max_drawdown_halt_pct * 0.3):
            return RiskState.CAUTION

        return RiskState.NORMAL

    def evaluate_trade_risk(
        self,
        decision: Decision,
        feature_frame: FeatureFrame,
        p_state: PortfolioState,
        current_price: float
    ) -> RiskDecision:
        reasons: List[str] = []
        flags: List[str] = []
        symbol = decision.symbol

        # 1. GLOBAL CRITICAL GATES
        if p_state.app_lifecycle_state != ApplicationState.READY and p_state.app_lifecycle_state != ApplicationState.TRADING:
            reasons.append(f"CRITICAL_LIFECYCLE_STATE_{p_state.app_lifecycle_state.value}")
            return self._build_reject_decision(symbol, 0.0, reasons, flags)

        if not p_state.is_reconciliation_clean:
            reasons.append("CRITICAL_RECONCILIATION_UNCLEAN")
            return self._build_reject_decision(symbol, 0.0, reasons, flags)

        if p_state.is_kill_switch_active:
            reasons.append("CRITICAL_KILL_SWITCH_ACTIVE")
            return self._build_reject_decision(symbol, 0.0, reasons, flags)

        current_risk_state = self.evaluate_risk_state(p_state)
        flags.append(f"RISK_STATE_{current_risk_state.value}")

        if current_risk_state == RiskState.HALTED:
            reasons.append("RISK_STATE_HALTED")
            return self._build_reject_decision(symbol, 0.0, reasons, flags)

        if decision.action not in (DecisionAction.BUY, DecisionAction.SELL):
            reasons.append("NO_TRADE_ACTION")
            return self._build_reject_decision(symbol, 0.0, reasons, flags)

        # 2. TRADE & REGIME LEVEL GATES
        if not feature_frame.is_valid:
            reasons.append("INVALID_FEATURES")
            return self._build_reject_decision(symbol, 0.0, reasons, flags)

        atr = feature_frame.features.get("ATR", 0.0)
        if atr <= 0 or current_price <= 0:
            reasons.append("INVALID_PRICE_OR_ATR")
            return self._build_reject_decision(symbol, 0.0, reasons, flags)

        # 3. PORTFOLIO EXPOSURE GATES
        current_asset_exp = p_state.active_positions.get(symbol, 0.0)
        max_asset_exp = p_state.total_equity_usdt * self.config.max_single_asset_exposure_pct
        avail_asset_capacity = max(0.0, max_asset_exp - current_asset_exp)

        if avail_asset_capacity < self.config.min_notional_usdt:
            reasons.append("SINGLE_ASSET_EXPOSURE_CAP_REACHED")
            return self._build_reject_decision(symbol, 0.0, reasons, flags)

        total_avail_portfolio_capacity = max(0.0, (p_state.total_equity_usdt * self.config.max_portfolio_exposure_pct) - p_state.current_portfolio_exposure_usdt)
        if total_avail_portfolio_capacity < self.config.min_notional_usdt:
            reasons.append("TOTAL_PORTFOLIO_EXPOSURE_CAP_REACHED")
            return self._build_reject_decision(symbol, 0.0, reasons, flags)

        # 4. DYNAMIC POSITION SIZING CALCULATION
        stop_distance_pct = (atr * self.config.atr_multiplier_stop) / current_price
        risk_capital = p_state.total_equity_usdt * self.config.base_risk_per_trade_pct

        # Apply Risk State Multipliers
        if current_risk_state == RiskState.CAUTION:
            risk_capital *= 0.50
            flags.append("SIZING_CUT_50PCT_CAUTION")
        elif current_risk_state == RiskState.RESTRICTED:
            risk_capital *= 0.25
            flags.append("SIZING_CUT_75PCT_RESTRICTED")

        calculated_notional = risk_capital / max(1e-4, stop_distance_pct)

        # Double-Cap Sizing: Min of calculated, asset cap, portfolio cap, and available cash
        approved_notional = min(
            calculated_notional,
            avail_asset_capacity,
            total_avail_portfolio_capacity,
            p_state.available_balance_usdt
        )

        if approved_notional < self.config.min_notional_usdt:
            reasons.append("APPROVED_NOTIONAL_BELOW_EXCHANGE_MINIMUM")
            return self._build_reject_decision(symbol, calculated_notional, reasons, flags)

        action = RiskAction.ALLOW if approved_notional >= (calculated_notional * 0.95) else RiskAction.REDUCE
        if action == RiskAction.REDUCE:
            reasons.append("NOTIONAL_REDUCED_BY_EXPOSURE_OR_CAPACITY_LIMITS")
        else:
            reasons.append("RISK_GATE_PASS")

        portfolio_exp_pct = (p_state.current_portfolio_exposure_usdt + approved_notional) / max(1.0, p_state.total_equity_usdt)
        max_loss = approved_notional * stop_distance_pct

        return RiskDecision(
            symbol=symbol,
            action=action,
            requested_notional=calculated_notional,
            approved_notional=approved_notional,
            risk_per_trade_pct=self.config.base_risk_per_trade_pct,
            portfolio_exposure_pct=portfolio_exp_pct,
            stop_distance_pct=stop_distance_pct,
            max_loss_usdt=max_loss,
            risk_score=1.0 - (approved_notional / max(1.0, p_state.total_equity_usdt)),
            risk_flags=flags,
            reason_codes=reasons
        )

    def _build_reject_decision(self, symbol: str, requested: float, reasons: List[str], flags: List[str]) -> RiskDecision:
        return RiskDecision(
            symbol=symbol,
            action=RiskAction.REJECT,
            requested_notional=requested,
            approved_notional=0.0,
            risk_per_trade_pct=0.0,
            portfolio_exposure_pct=0.0,
            stop_distance_pct=0.0,
            max_loss_usdt=0.0,
            risk_score=1.0,
            risk_flags=flags,
            reason_codes=reasons
        )


class PositionSizer:
    """Penerjemah Decision & RiskDecision menjadi PositionPlan konkret."""

    @staticmethod
    def create_position_plan(
        decision: Decision,
        risk_decision: RiskDecision,
        current_price: float
    ) -> PositionPlan:
        if risk_decision.action == RiskAction.REJECT or risk_decision.approved_notional <= 0 or current_price <= 0:
            return PositionPlan(
                symbol=decision.symbol,
                timestamp=decision.timestamp,
                action=DecisionAction.NO_TRADE,
                approved_notional_usdt=0.0,
                calculated_quantity=0.0,
                target_price=current_price,
                stop_loss_price=None,
                take_profit_price=None,
                risk_decision=risk_decision
            )

        qty = risk_decision.approved_notional / current_price

        return PositionPlan(
            symbol=decision.symbol,
            timestamp=decision.timestamp,
            action=decision.action,
            approved_notional_usdt=risk_decision.approved_notional,
            calculated_quantity=qty,
            target_price=current_price,
            stop_loss_price=decision.stop_loss,
            take_profit_price=decision.take_profit,
            risk_decision=risk_decision
        )


class MultiPairPortfolioRanker:
    """Peringkat Peluang Portfolio untuk Mengalokasikan Kapital Terbatas."""

    @staticmethod
    def rank_and_filter_plans(
        plans: Dict[str, PositionPlan],
        available_capital_usdt: float
    ) -> List[PositionPlan]:
        # Filter hanya rencana yang diizinkan (ALLOW / REDUCE)
        valid_plans = [p for p in plans.values() if p.action in (DecisionAction.BUY, DecisionAction.SELL) and p.approved_notional_usdt > 0]
        
        # Urutkan berdasarkan Expected Value & Confidence Decision
        valid_plans.sort(key=lambda p: (p.risk_decision.risk_score, p.approved_notional_usdt), reverse=True)

        approved_plans: List[PositionPlan] = []
        remaining_cap = available_capital_usdt

        for plan in valid_plans:
            if remaining_cap >= plan.approved_notional_usdt:
                approved_plans.append(plan)
                remaining_cap -= plan.approved_notional_usdt
            elif remaining_cap >= 10.0:  # Min Notional
                # Scale down last feasible plan
                scaled_plan = PositionPlan(
                    symbol=plan.symbol,
                    timestamp=plan.timestamp,
                    action=plan.action,
                    approved_notional_usdt=remaining_cap,
                    calculated_quantity=remaining_cap / plan.target_price,
                    target_price=plan.target_price,
                    stop_loss_price=plan.stop_loss_price,
                    take_profit_price=plan.take_profit_price,
                    risk_decision=plan.risk_decision
                )
                approved_plans.append(scaled_plan)
                remaining_cap = 0.0
                break

        return approved_plans
