"""
MODULE: tokocrypto_bot.strategy.strategies.scalping
DESCRIPTION: High-Frequency Microstructure & Volatility Scalping Strategy.
"""

from typing import Set, Optional
from tokocrypto_bot.strategy.strategies.base import BaseStrategy, CandidateSignal, StrategySignalSide
from tokocrypto_bot.strategy.features import FeatureFrame
from tokocrypto_bot.strategy.regime import MarketRegime


class ScalpingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="ScalpingStrategy", version="2026.1.0", timeframe="1m")

    @property
    def applicable_regimes(self) -> Set[MarketRegime]:
        # Scalping sangat efektif di pasar ber-volatilitas tinggi
        return {MarketRegime.TRENDING_HIGH_VOL, MarketRegime.RANGE_HIGH_VOL}

    def generate_candidate_signal(self, feature_frame: FeatureFrame) -> Optional[CandidateSignal]:
        if not feature_frame.is_valid:
            return None

        feats = feature_frame.features
        rsi = feats.get("RSI14", 50.0)
        pband = feats.get("bb_pband", 0.5)
        vwma_dev = feats.get("vwma_dev", 0.0)

        # Logika Micro Scalp: Oversold Bollinger %B + Low VWMA Dev
        if pband < 0.15 and rsi < 35.0 and vwma_dev < -0.01:
            return CandidateSignal(
                strategy_name=self.name,
                symbol=feature_frame.symbol,
                timestamp=feature_frame.timestamp,
                side=StrategySignalSide.BUY,
                raw_confidence=0.80,
                expected_value=0.018,
                stop_loss_pct=0.008,   # Tight SL 0.8%
                take_profit_pct=0.016, # Target TP 1.6%
                reason="Scalp Buy: %B oversold + VWMA deviation dip"
            )
        elif pband > 0.85 and rsi > 65.0:
            return CandidateSignal(
                strategy_name=self.name,
                symbol=feature_frame.symbol,
                timestamp=feature_frame.timestamp,
                side=StrategySignalSide.SELL,
                raw_confidence=0.75,
                expected_value=0.015,
                stop_loss_pct=0.008,
                take_profit_pct=0.015,
                reason="Scalp Sell: %B overbought top"
            )

        return None
