"""
MODULE: tests.test_p1_features_and_inference
DESCRIPTION: Test Suite for P1-B (Feature Engine) and P1-C (ML Inference Engine).
"""

import os
import pickle
import pytest
import numpy as np
import pandas as pd
from typing import List

from tokocrypto_bot.strategy.market_data import OHLCVFrame, DataSource
from tokocrypto_bot.strategy.features import FeatureEngine, FeatureFrame, MIN_REQUIRED_CANDLES, FEATURE_VERSION
from tokocrypto_bot.ml.inference import MLInferenceEngine, PredictionResult


def generate_dummy_klines(count: int = 210) -> pd.DataFrame:
    now_ms = 1700000000000
    timestamps = [now_ms + (i * 60000) for i in range(count)]
    
    data = []
    base_price = 50000.0
    for i, ts in enumerate(timestamps):
        price = base_price + (i * 1.5)
        data.append({
            "timestamp": ts,
            "open": price,
            "high": price + 5.0,
            "low": price - 5.0,
            "close": price + 2.0,
            "volume": 100.0 + i,
            "source": DataSource.TOKOCRYPTO.value,
            "is_complete": i < (count - 1)  # Candle terakhir belum closed
        })
    return pd.DataFrame(data)


def test_feature_engine_prevents_unclosed_candles_and_validates_minimum_history():
    fe = FeatureEngine()
    
    # 1. Less than minimum candles test
    df_short = generate_dummy_klines(100)
    ff_short = fe.compute_features(df_short, "BTCUSDT")
    assert ff_short.is_valid is False
    assert "Insufficient history" in ff_short.error_reason

    # 2. Sufficient candles test & Verify Unclosed Candle exclusion
    df_full = generate_dummy_klines(210)
    ff_full = fe.compute_features(df_full, "BTCUSDT")
    assert ff_full.is_valid is True
    # Candle timestamp harus menunjuk pada index -2 (candle closed terakhir), bukan index -1 (unclosed)
    expected_last_closed_ts = int(df_full.iloc[-2]["timestamp"])
    assert ff_full.timestamp == expected_last_closed_ts


def test_feature_engine_handles_nan_inf():
    fe = FeatureEngine()
    df = generate_dummy_klines(210)
    # Inject zero to trigger division by zero in close/volume
    df["close"] = 0.0
    
    ff = fe.compute_features(df, "BTCUSDT")
    assert ff.is_valid is False
    assert "NaN/Inf" in ff.error_reason or "Insufficient" in ff.error_reason


def test_ml_inference_model_unavailable_policy(tmp_path):
    # Pass non-existent model path
    dummy_path = str(tmp_path / "non_existent_model.pkl")
    engine = MLInferenceEngine(model_path=dummy_path)
    
    fe = FeatureEngine()
    df = generate_dummy_klines(210)
    ff = fe.compute_features(df, "BTCUSDT")
    
    result = engine.predict(ff)
    assert result.is_valid is False
    assert result.status_code == "MODEL_UNAVAILABLE"
    assert "NO_TRADE" in result.reason


def test_ml_inference_success_with_mock_model(tmp_path):
    # Create Mock Model Artifact
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])

    model_path = str(tmp_path / "mock_champion.pkl")
    artifact = {"model": MockModel(), "version": "2026.1.1"}
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    engine = MLInferenceEngine(model_path=model_path)
    fe = FeatureEngine()
    df = generate_dummy_klines(210)
    ff = fe.compute_features(df, "BTCUSDT")

    result = engine.predict(ff)
    assert result.is_valid is True
    assert result.status_code == "OK"
    assert result.probability_up == 0.7
    assert result.probability_down == 0.3
    assert round(result.confidence, 2) == 0.40
