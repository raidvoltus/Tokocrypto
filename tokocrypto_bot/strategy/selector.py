"""
MODULE: tokocrypto_bot.strategy.selector
DESCRIPTION: Adaptive Strategy Selector & Dynamic Scoring Engine.
"""

import logging
from typing import List, Dict, Optional, Tuple

from tokocrypto_bot.strategy.features import FeatureFrame
from tokocrypto_bot.strategy.regime import RegimeClassifier, RegimeContext, MarketRegime
from tokocrypto_bot.strategy.strategies.base import BaseStrategy, CandidateSignal
from tokocrypto_bot.strategy.strategies.scalping import ScalpingStrategy
from tokocrypto_bot.ml.inference import MLInferenceEngine, PredictionResult

logger = logging.getLogger("NVRA.StrategySelector")


class AdaptiveStrategySelector:
    def __init__(self, strategies: Optional[List[BaseStrategy]] = None):
        # Default Strategy Registry
        self.strategies: List[BaseStrategy] = strategies or [
            ScalpingStrategy(),
            # Tempat mendaftarkan MomentumStrategy(), MeanReversionStrategy(), BreakoutStrategy()
        ]
        self.regime_classifier = RegimeClassifier()

    def select_best_signal(
        self,
        symbol: str,
        feature_frame: FeatureFrame,
        prediction: PredictionResult,
        is_liquid: bool = True
    ) -> Tuple[Optional[CandidateSignal], float, RegimeContext]:
        """
        Memilih kandidat sinyal terbaik berdasarkan Adaptive Strategy Scoring.
        Mematikan strategi secara otomatis jika rezim tidak cocok.
        """
        regime_ctx = self.regime_classifier.classify(feature_frame, is_liquid)
        
        best_candidate: Optional[CandidateSignal] = None
        best_score: float = -1.0

        if not feature_frame.is_valid or not prediction.is_valid:
            return None, 0.0, regime_ctx

        for strategy in self.strategies:
            if not strategy.is_enabled:
                continue

            # Check Regime Compatibility
            if regime_ctx.regime not in strategy.applicable_regimes:
                continue

            candidate = strategy.generate_candidate_signal(feature_frame)
            if not candidate:
                continue

            # Calculate Adaptive Score
            regime_fit = 1.0
            ml_confidence = prediction.confidence
            liquidity_factor = 1.0 if is_liquid else 0.5
            ev_factor = max(0.0, candidate.expected_value)

            # Strategy Score Formula
            score = regime_fit * (0.4 * prediction.probability_up + 0.3 * ml_confidence + 0.3 * (ev_factor * 10)) * liquidity_factor

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate, best_score, regime_ctx
