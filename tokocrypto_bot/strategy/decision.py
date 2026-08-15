"""
MODULE: tokocrypto_bot.strategy.decision
DESCRIPTION: Multi-Pair Signal & Decision Engine with Detailed Audit Reason Codes (P1-D).
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from tokocrypto_bot.strategy.features import FeatureFrame
from tokocrypto_bot.ml.inference import PredictionResult

logger = logging.getLogger("NVRA.DecisionEngine")

STRATEGY_VERSION = "2026.1.0"


class DecisionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class Decision:
    symbol: str
    timestamp: int
    action: DecisionAction
    probability: float
    confidence: float
    expected_value: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reason_codes: List[str]
    strategy_version: str
    model_version: str


@dataclass(frozen=True)
class DecisionThresholds:
    min_buy_probability: float = 0.65
    min_sell_probability: float = 0.65
    min_confidence: float = 0.30
    min_expected_value: float = 0.015  # Minimal EV 1.5%
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


class DecisionEngine:
    def __init__(self, thresholds: Optional[DecisionThresholds] = None):
        self.thresholds = thresholds or DecisionThresholds()

    def evaluate(
        self,
        symbol: str,
        feature_frame: FeatureFrame,
        prediction: PredictionResult,
        current_position_qty: float = 0.0,
        current_price: float = 0.0
    ) -> Decision:
        reasons: List[str] = []

        # 1. Verification of ML Inference Result
        if not prediction.is_valid:
            reasons.append(f"ML_INVALID_{prediction.status_code}")
            return self._build_decision(symbol, feature_frame.timestamp, DecisionAction.NO_TRADE, 0.0, 0.0, 0.0, None, None, reasons, prediction.model_version)

        # 2. Verification of Feature Frame
        if not feature_frame.is_valid:
            reasons.append(f"FEATURES_INVALID_{feature_frame.error_reason}")
            return self._build_decision(symbol, feature_frame.timestamp, DecisionAction.NO_TRADE, 0.0, 0.0, 0.0, None, None, reasons, prediction.model_version)

        prob_up = prediction.probability_up
        prob_down = prediction.probability_down
        confidence = prediction.confidence
        feats = feature_frame.features

        rsi = feats.get("RSI14", 50.0)
        macd_hist = feats.get("MACD_HIST", 0.0)
        ema_ratio = feats.get("ema_ratio", 1.0)
        atr = feats.get("ATR", 0.0)

        # 3. Decision Logic: BUY Evaluation
        if prob_up >= self.thresholds.min_buy_probability:
            reasons.append("ML_BUY_PROBABILITY_PASS")
            
            # Technical Filters
            tech_pass = True
            if rsi > self.thresholds.rsi_overbought:
                reasons.append("RSI_OVERBOUGHT_FAIL")
                tech_pass = False
            if ema_ratio < 0.98:
                reasons.append("BEARISH_EMA_TREND_FAIL")
                tech_pass = False

            if tech_pass:
                reasons.append("TECHNICAL_CONFIRMATION_PASS")

                # Expected Value & Dynamic Risk Targets
                sl_price = current_price - (2.0 * atr) if current_price > 0 and atr > 0 else None
                tp_price = current_price + (3.0 * atr) if current_price > 0 and atr > 0 else None

                # Simplified EV: (Prob_Up * Reward) - (Prob_Down * Risk)
                reward_pct = (3.0 * atr) / current_price if current_price > 0 else 0.03
                risk_pct = (2.0 * atr) / current_price if current_price > 0 else 0.02
                ev = (prob_up * reward_pct) - (prob_down * risk_pct)

                if ev >= self.thresholds.min_expected_value:
                    reasons.append("EV_PASS")
                    return self._build_decision(symbol, feature_frame.timestamp, DecisionAction.BUY, prob_up, confidence, ev, sl_price, tp_price, reasons, prediction.model_version)
                else:
                    reasons.append("EV_THRESHOLD_FAIL")

        # 4. Decision Logic: SELL / EXIT Evaluation
        elif prob_down >= self.thresholds.min_sell_probability or (current_position_qty > 0 and rsi > self.thresholds.rsi_overbought):
            reasons.append("ML_OR_TECH_SELL_SIGNAL")
            sl_price = current_price + (2.0 * atr) if current_price > 0 and atr > 0 else None
            tp_price = current_price - (3.0 * atr) if current_price > 0 and atr > 0 else None
            ev = (prob_down * 0.03) - (prob_up * 0.02)
            
            return self._build_decision(symbol, feature_frame.timestamp, DecisionAction.SELL, prob_down, confidence, ev, sl_price, tp_price, reasons, prediction.model_version)

        # 5. Default Fallback: HOLD / NO_TRADE
        reasons.append("SIGNAL_NEUTRAL")
        action = DecisionAction.HOLD if current_position_qty > 0 else DecisionAction.NO_TRADE
        return self._build_decision(symbol, feature_frame.timestamp, action, max(prob_up, prob_down), confidence, 0.0, None, None, reasons, prediction.model_version)

    def evaluate_multi_pair(
        self,
        feature_frames: Dict[str, FeatureFrame],
        predictions: Dict[str, PredictionResult],
        current_positions: Optional[Dict[str, float]] = None,
        current_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Decision]:
        """Batch evaluasi untuk seluruh pair di Pair Universe."""
        decisions = {}
        positions = current_positions or {}
        prices = current_prices or {}

        for sym, ff in feature_frames.items():
            pred = predictions.get(sym, PredictionResult(sym, 0, "NONE", "NONE", 0.0, 0.0, 0.0, False, "MISSING_PREDICTION"))
            pos_qty = positions.get(sym, 0.0)
            price = prices.get(sym, 0.0)
            decisions[sym] = self.evaluate(sym, ff, pred, pos_qty, price)

        return decisions

    def _build_decision(
        self,
        symbol: str,
        timestamp: int,
        action: DecisionAction,
        prob: float,
        conf: float,
        ev: float,
        sl: Optional[float],
        tp: Optional[float],
        reasons: List[str],
        model_ver: str
    ) -> Decision:
        return Decision(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            probability=prob,
            confidence=conf,
            expected_value=ev,
            stop_loss=sl,
            take_profit=tp,
            reason_codes=reasons,
            strategy_version=STRATEGY_VERSION,
            model_version=model_ver
        )
