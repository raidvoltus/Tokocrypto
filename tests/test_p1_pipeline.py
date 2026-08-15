"""
MODULE: tests.test_p1_pipeline
DESCRIPTION: Comprehensive Contract & Integration Tests for P1-B -> P1-C -> P1-D Pipeline.
"""

import os
import pickle
import pytest
import numpy as np
import pandas as pd

from tokocrypto_bot.strategy.features import FeatureEngine, FEATURE_VERSION
from tokocrypto_bot.ml.inference import MLInferenceEngine
from tokocrypto_bot.strategy.decision import DecisionEngine, DecisionAction


def create_mock_klines(symbol: str, count: int = 210) -> pd.DataFrame:
    now_ms = 1700000000000
    timestamps = [now_ms + (i * 60000) for i in range(count)]
    data = []
    base_price = 1000.0
    for i, ts in enumerate(timestamps):
        price = base_price + (i * 0.5)
        data.append({
            "timestamp": ts,
            "open": price,
            "high": price + 2.0,
            "low": price - 2.0,
            "close": price + 1.0,
            "volume": 500.0,
            "is_complete": i < (count - 1)
        })
    return pd.DataFrame(data)


def test_full_p1_pipeline_multi_pair_integration(tmp_path):
    # Setup Mock Model
    class MockModel:
        def predict_proba(self, X):
            # Probs: [Prob_Down, Prob_Up]
            return np.array([[0.2, 0.8]])

    model_path = str(tmp_path / "champion_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": MockModel(), "version": "2026.1_TEST"}, f)

    # Instantiate Pipeline Engines
    feature_engine = FeatureEngine()
    inference_engine = MLInferenceEngine(model_path=model_path)
    decision_engine = DecisionEngine()

    # Create Multi-Pair Data (BTCUSDT & ETHUSDT)
    klines_map = {
        "BTCUSDT": create_mock_klines("BTCUSDT", 210),
        "ETHUSDT": create_mock_klines("ETHUSDT", 210)
    }

    # Execute Pipeline
    feature_frames = feature_engine.compute_multi_pair_features(klines_map)
    predictions = inference_engine.predict_multi_pair(feature_frames)
    decisions = decision_engine.evaluate_multi_pair(
        feature_frames, predictions, current_prices={"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}
    )

    # Verification BTCUSDT Decision
    btc_dec = decisions["BTCUSDT"]
    assert btc_dec.action == DecisionAction.BUY
    assert btc_dec.probability == 0.8
    assert "ML_BUY_PROBABILITY_PASS" in btc_dec.reason_codes
    assert "EV_PASS" in btc_dec.reason_codes
    assert btc_dec.stop_loss is not None


def test_pipeline_fallback_on_missing_model(tmp_path):
    feature_engine = FeatureEngine()
    inference_engine = MLInferenceEngine(model_path=str(tmp_path / "missing.pkl"))
    decision_engine = DecisionEngine()

    klines_map = {"SOLUSDT": create_mock_klines("SOLUSDT", 210)}

    feature_frames = feature_engine.compute_multi_pair_features(klines_map)
    predictions = inference_engine.predict_multi_pair(feature_frames)
    decisions = decision_engine.evaluate_multi_pair(feature_frames, predictions)

    sol_dec = decisions["SOLUSDT"]
    assert sol_dec.action == DecisionAction.NO_TRADE
    assert "ML_INVALID_MODEL_UNAVAILABLE" in sol_dec.reason_codes
