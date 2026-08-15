"""
MODULE: tests.test_p1_adaptive_multi_strategy
DESCRIPTION: Unit and Integration Test Suite for Adaptive Multi-Strategy Engine.
"""

import pytest
import pandas as pd

from tokocrypto_bot.strategy.features import FeatureEngine
from tokocrypto_bot.strategy.regime import RegimeClassifier, MarketRegime
from tokocrypto_bot.strategy.selector import AdaptiveStrategySelector
from tokocrypto_bot.strategy.strategies.scalping import ScalpingStrategy
from tokocrypto_bot.ml.inference import PredictionResult


def create_high_volatility_df(count: int = 210) -> pd.DataFrame:
    timestamps = [1700000000000 + (i * 60000) for i in range(count)]
    data = []
    base_price = 1000.0
    for i in range(count):
        # Ayunan harga besar untuk memicu high volatility %B oversold
        price = base_price + (10.0 if i % 2 == 0 else -10.0)
        if i == count - 2:  # Dip harga tajam pada candle closed terakhir
            price = 950.0
        data.append({
            "timestamp": timestamps[i],
            "open": price, "high": price + 5.0, "low": price - 5.0, "close": price,
            "volume": 2000.0, "is_complete": i < (count - 1)
        })
    return pd.DataFrame(data)


def test_regime_classification_and_strategy_selection():
    fe = FeatureEngine()
    selector = AdaptiveStrategySelector()

    df = create_high_volatility_df(210)
    ff = fe.compute_features(df, "SOLUSDT")
    assert ff.is_valid is True

    # Mock Prediction Result (ML Probability High)
    mock_pred = PredictionResult(
        symbol="SOLUSDT", timestamp=ff.timestamp, model_version="2026.1",
        feature_version=ff.feature_version, probability_up=0.78, probability_down=0.22,
        confidence=0.56, is_valid=True, status_code="OK"
    )

    best_candidate, score, regime_ctx = selector.select_best_signal("SOLUSDT", ff, mock_pred, is_liquid=True)

    # Verification: High Volatility harus mengaktifkan ScalpingStrategy
    assert regime_ctx.regime in (MarketRegime.TRENDING_HIGH_VOL, MarketRegime.RANGE_HIGH_VOL)
    if best_candidate:
        assert best_candidate.strategy_name == "ScalpingStrategy"
        assert score > 0.0
