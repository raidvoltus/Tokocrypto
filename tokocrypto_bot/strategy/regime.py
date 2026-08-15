"""
MODULE: tokocrypto_bot.strategy.regime
DESCRIPTION: Market & Pair Regime Classification Engine for Multi-Strategy Selection.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

from tokocrypto_bot.strategy.features import FeatureFrame

logger = logging.getLogger("NVRA.RegimeClassifier")


class MarketRegime(str, Enum):
    TRENDING_HIGH_VOL = "TRENDING_HIGH_VOL"
    TRENDING_LOW_VOL = "TRENDING_LOW_VOL"
    RANGE_HIGH_VOL = "RANGE_HIGH_VOL"
    RANGE_LOW_VOL = "RANGE_LOW_VOL"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class RegimeContext:
    symbol: str
    timestamp: int
    regime: MarketRegime
    adx_value: float
    volatility_score: float
    is_liquid: bool
    summary: str


class RegimeClassifier:
    def __init__(self, adx_trending_threshold: float = 25.0, high_vol_threshold: float = 0.02):
        self.adx_threshold = adx_trending_threshold
        self.high_vol_threshold = high_vol_threshold

    def classify(self, feature_frame: FeatureFrame, is_liquid: bool = True) -> RegimeContext:
        """Mengidentifikasi rezim pasar dari FeatureFrame P1-B."""
        if not feature_frame.is_valid:
            return RegimeContext(
                symbol=feature_frame.symbol, timestamp=feature_frame.timestamp,
                regime=MarketRegime.NEUTRAL, adx_value=0.0, volatility_score=0.0,
                is_liquid=is_liquid, summary=f"Invalid Features: {feature_frame.error_reason}"
            )

        feats = feature_frame.features
        di_plus = feats.get("DI_plus", 0.0)
        di_minus = feats.get("DI_minus", 0.0)
        volatility = feats.get("volatility_regime", 0.0)
        
        # Estimasi ADX dari selisih DI+ / DI-
        di_sum = max(1e-8, di_plus + di_minus)
        adx_approx = (abs(di_plus - di_minus) / di_sum) * 100.0

        is_trending = adx_approx >= self.adx_threshold
        is_high_vol = volatility >= self.high_vol_threshold

        if is_trending and is_high_vol:
            regime = MarketRegime.TRENDING_HIGH_VOL
        elif is_trending and not is_high_vol:
            regime = MarketRegime.TRENDING_LOW_VOL
        elif not is_trending and is_high_vol:
            regime = MarketRegime.RANGE_HIGH_VOL
        else:
            regime = MarketRegime.RANGE_LOW_VOL

        summary = f"ADX={adx_approx:.1f}, Vol={volatility:.4f} -> {regime.value}"
        return RegimeContext(
            symbol=feature_frame.symbol,
            timestamp=feature_frame.timestamp,
            regime=regime,
            adx_value=adx_approx,
            volatility_score=volatility,
            is_liquid=is_liquid,
            summary=summary
        )
