"""
TEST SUITE: tokocrypto_bot.ml.inference
DESCRIPTION: Test ML inference engine with model resolution, integrity validation, and fail-closed behavior.

Test Coverage:
- Valid model loading with predictions
- NVRA_MODEL_PATH override
- Missing model → is_valid=False → NO_TRADE
- Corrupt/invalid model → is_valid=False → NO_TRADE
- Feature-version mismatch
- SHA-256 integrity mismatch
- Valid SHA-256 model
- Exception handling cannot bypass fail-closed behavior
- Batch prediction fails closed for all pairs
"""

import os
import sys
import pytest
import tempfile
import hashlib
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from dataclasses import dataclass

from tokocrypto_bot.ml.inference import MLInferenceEngine, PredictionResult
from tokocrypto_bot.strategy.features import FeatureFrame, FEATURE_VERSION


# Test fixtures
@pytest.fixture
def valid_feature_frame():
    """Create a valid FeatureFrame for testing."""
    features = {
        "EMA50": 100.5,
        "EMA200": 99.8,
        "RSI14": 55.0,
        "ROC": 0.05,
        "ATR": 2.0,
        "volatility_regime": 0.02,
        "MACD_HIST": 0.1,
        "DI_plus": 25.0,
        "DI_minus": 20.0,
        "ema_ratio": 1.007,
        "bb_pband": 0.5,
        "obv_vs_ma": 1000.0,
        "cmf": 0.3,
        "vwma_dev": 0.02,
        "drawdown_20": -0.05
    }
    return FeatureFrame(
        timestamp=1693497600000,
        symbol="BTCUSDT",
        feature_version=FEATURE_VERSION,
        features=features,
        is_valid=True
    )


@pytest.fixture
def invalid_feature_frame():
    """Create an invalid FeatureFrame for testing."""
    return FeatureFrame(
        timestamp=0,
        symbol="BTCUSDT",
        feature_version=FEATURE_VERSION,
        features={},
        is_valid=False,
        error_reason="Insufficient history"
    )


@pytest.fixture
def mock_sklearn_model():
    """Create a mock sklearn model that implements predict_proba."""
    model = Mock()
    model.predict_proba = Mock(return_value=[[0.3, 0.7]])  # 30% DOWN, 70% UP
    return model


@pytest.fixture
def mock_xgboost_model():
    """Create a mock xgboost model that only implements predict."""
    model = Mock()
    model.predict = Mock(return_value=[0.75])  # 75% probability up
    delattr(model, 'predict_proba')  # Ensure predict_proba doesn't exist
    return model


@pytest.fixture
def temp_model_file(mock_sklearn_model):
    """Create a temporary model file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "champion_model.pkl"
        artifact = {"model": mock_sklearn_model, "version": "1.0.0"}
        with open(model_path, "wb") as f:
            pickle.dump(artifact, f)
        yield model_path


class TestMLInferenceEngineModelLoading:
    """Test model loading with cross-platform resolution."""
    
    def test_model_loads_successfully_with_explicit_path(self, temp_model_file):
        """Valid model loads successfully when explicit path provided."""
        engine = MLInferenceEngine(model_path=str(temp_model_file))
        assert engine.model is not None
        assert engine.model_version == "1.0.0"
        assert engine.model_path == str(temp_model_file)
    
    def test_model_unavailable_fails_closed(self):
        """Model unavailable (no path resolution) → fail closed."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('tokocrypto_bot.ml.inference.resolve_and_validate_model_path') as mock_resolve:
                mock_resolve.return_value = (None, False)
                engine = MLInferenceEngine()
                assert engine.model is None
                assert engine.model_version == "UNAVAILABLE"
    
    def test_model_path_validation_fails(self, temp_model_file):
        """Model found but validation fails (hash mismatch) → fail closed."""
        with patch('tokocrypto_bot.ml.inference.resolve_and_validate_model_path') as mock_resolve:
            mock_resolve.return_value = (temp_model_file, False)  # Path exists but validation failed
            engine = MLInferenceEngine()
            assert engine.model is None
            assert engine.model_version == "UNAVAILABLE"
    
    def test_corrupt_model_file_fails_closed(self):
        """Corrupt/invalid model file → fail closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_model = Path(tmpdir) / "bad_model.pkl"
            corrupt_model.write_bytes(b"not valid pickle data")
            
            engine = MLInferenceEngine(model_path=str(corrupt_model))
            assert engine.model is None
            assert engine.model_version == "CORRUPT"
    
    def test_legacy_model_without_version_metadata(self):
        """Legacy model (raw object, no dict wrapper) loads and gets LEGACY version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "legacy_model.pkl"
            # Serialize raw model, not wrapped in dict
            legacy_model = Mock()
            legacy_model.predict_proba = Mock(return_value=[[0.3, 0.7]])
            with open(model_path, "wb") as f:
                pickle.dump(legacy_model, f)
            
            engine = MLInferenceEngine(model_path=str(model_path))
            assert engine.model is not None
            assert engine.model_version == "LEGACY_1.0"


class TestMLInferenceEngineEnvironmentVariables:
    """Test NVRA_MODEL_PATH and NVRA_MODEL_SHA256 environment variable handling."""
    
    def test_nvra_model_path_override(self, temp_model_file):
        """NVRA_MODEL_PATH environment variable overrides default resolution."""
        with patch.dict(os.environ, {'NVRA_MODEL_PATH': str(temp_model_file)}):
            with patch('tokocrypto_bot.ml.inference.resolve_and_validate_model_path') as mock_resolve:
                # Simulate the real behavior: resolve_and_validate checks NVRA_MODEL_PATH first
                mock_resolve.return_value = (temp_model_file, True)
                engine = MLInferenceEngine()
                assert engine.model is not None
                assert engine.model_version == "1.0.0"
    
    def test_sha256_validation_passed(self, temp_model_file):
        """Valid SHA-256 in NVRA_MODEL_SHA256 allows model loading."""
        with open(temp_model_file, "rb") as f:
            model_data = f.read()
        
        sha256_hash = hashlib.sha256()
        sha256_hash.update(model_data)
        correct_hash = sha256_hash.hexdigest()
        
        with patch.dict(os.environ, {
            'NVRA_MODEL_PATH': str(temp_model_file),
            'NVRA_MODEL_SHA256': correct_hash
        }):
            with patch('tokocrypto_bot.ml.inference.resolve_and_validate_model_path') as mock_resolve:
                mock_resolve.return_value = (temp_model_file, True)
                engine = MLInferenceEngine()
                assert engine.model is not None
    
    def test_sha256_validation_failed(self, temp_model_file):
        """Invalid SHA-256 in NVRA_MODEL_SHA256 → fail closed."""
        wrong_hash = "0" * 64
        
        with patch.dict(os.environ, {
            'NVRA_MODEL_PATH': str(temp_model_file),
            'NVRA_MODEL_SHA256': wrong_hash
        }):
            with patch('tokocrypto_bot.ml.inference.resolve_and_validate_model_path') as mock_resolve:
                mock_resolve.return_value = (temp_model_file, False)  # Hash validation failed
                engine = MLInferenceEngine()
                assert engine.model is None
                assert engine.model_version == "UNAVAILABLE"


class TestMLInferencePredictionFailClosed:
    """Test that all invalid model states produce is_valid=False predictions."""
    
    def test_predict_model_unavailable_returns_invalid(self, valid_feature_frame):
        """Model unavailable → PredictionResult.is_valid=False."""
        with patch('tokocrypto_bot.ml.inference.resolve_and_validate_model_path') as mock_resolve:
            mock_resolve.return_value = (None, False)
            engine = MLInferenceEngine()
            
            result = engine.predict(valid_feature_frame)
            
            assert result.is_valid is False
            assert result.confidence == 0.0
            assert result.status_code == "MODEL_UNAVAILABLE"
            assert result.probability_up == 0.0
            assert result.probability_down == 0.0
    
    def test_predict_feature_frame_invalid_returns_invalid(self, invalid_feature_frame, temp_model_file):
        """Invalid feature frame → PredictionResult.is_valid=False."""
        engine = MLInferenceEngine(model_path=str(temp_model_file))
        
        result = engine.predict(invalid_feature_frame)
        
        assert result.is_valid is False
        assert result.confidence == 0.0
        assert result.status_code == "INVALID_INPUT"
    
    def test_predict_feature_version_mismatch_returns_invalid(self, valid_feature_frame, temp_model_file):
        """Feature version mismatch → PredictionResult.is_valid=False."""
        engine = MLInferenceEngine(model_path=str(temp_model_file))
        
        # Create frame with mismatched feature version
        mismatched_frame = FeatureFrame(
            timestamp=valid_feature_frame.timestamp,
            symbol=valid_feature_frame.symbol,
            feature_version="WRONG_VERSION",
            features=valid_feature_frame.features,
            is_valid=True
        )
        
        result = engine.predict(mismatched_frame)
        
        assert result.is_valid is False
        assert result.confidence == 0.0
        assert result.status_code == "FEATURE_MISMATCH"
    
    def test_predict_missing_feature_column_returns_invalid(self, valid_feature_frame, temp_model_file):
        """Missing feature column → PredictionResult.is_valid=False."""
        engine = MLInferenceEngine(model_path=str(temp_model_file))
        
        # Create frame missing a required feature
        broken_features = valid_feature_frame.features.copy()
        del broken_features["EMA50"]  # Remove a required feature
        
        broken_frame = FeatureFrame(
            timestamp=valid_feature_frame.timestamp,
            symbol=valid_feature_frame.symbol,
            feature_version=valid_feature_frame.feature_version,
            features=broken_features,
            is_valid=True
        )
        
        result = engine.predict(broken_frame)
        
        assert result.is_valid is False
        assert result.confidence == 0.0
        assert result.status_code == "FEATURE_MISMATCH"
    
    def test_predict_nan_in_features_returns_invalid(self, valid_feature_frame, temp_model_file):
        """NaN in feature vector → PredictionResult.is_valid=False."""
        engine = MLInferenceEngine(model_path=str(temp_model_file))
        
        # Inject NaN
        broken_features = valid_feature_frame.features.copy()
        broken_features["RSI14"] = float('nan')
        
        broken_frame = FeatureFrame(
            timestamp=valid_feature_frame.timestamp,
            symbol=valid_feature_frame.symbol,
            feature_version=valid_feature_frame.feature_version,
            features=broken_features,
            is_valid=True
        )
        
        result = engine.predict(broken_frame)
        
        assert result.is_valid is False
        assert result.confidence == 0.0
        assert result.status_code == "INVALID_INPUT"
    
    def test_predict_inf_in_features_returns_invalid(self, valid_feature_frame, temp_model_file):
        """Inf in feature vector → PredictionResult.is_valid=False."""
        engine = MLInferenceEngine(model_path=str(temp_model_file))
        
        # Inject Inf
        broken_features = valid_feature_frame.features.copy()
        broken_features["ATR"] = float('inf')
        
        broken_frame = FeatureFrame(
            timestamp=valid_feature_frame.timestamp,
            symbol=valid_feature_frame.symbol,
            feature_version=valid_feature_frame.feature_version,
            features=broken_features,
            is_valid=True
        )
        
        result = engine.predict(broken_frame)
        
        assert result.is_valid is False
        assert result.confidence == 0.0
        assert result.status_code == "INVALID_INPUT"


class TestMLInferenceValidPredictions:
    """Test valid inference scenarios."""
    
    def test_sklearn_model_with_predict_proba(self, valid_feature_frame, mock_sklearn_model):
        """Valid sklearn model with predict_proba produces valid prediction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            artifact = {"model": mock_sklearn_model, "version": "2.0.0"}
            with open(model_path, "wb") as f:
                pickle.dump(artifact, f)
            
            engine = MLInferenceEngine(model_path=str(model_path))
            result = engine.predict(valid_feature_frame)
            
            assert result.is_valid is True
            assert result.status_code == "OK"
            assert result.probability_down == 0.3
            assert result.probability_up == 0.7
            assert result.confidence == 0.4  # abs(0.7 - 0.3)
            assert result.model_version == "2.0.0"
    
    def test_xgboost_model_with_predict_only(self, valid_feature_frame, mock_xgboost_model):
        """Valid xgboost model with only predict() produces valid prediction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            artifact = {"model": mock_xgboost_model, "version": "1.5.0"}
            with open(model_path, "wb") as f:
                pickle.dump(artifact, f)
            
            engine = MLInferenceEngine(model_path=str(model_path))
            result = engine.predict(valid_feature_frame)
            
            assert result.is_valid is True
            assert result.status_code == "OK"
            assert result.probability_up == 0.75
            assert result.probability_down == 0.25
            assert result.confidence == 0.5  # abs(0.75 - 0.25)
    
    def test_inference_exception_caught_returns_invalid(self, valid_feature_frame):
        """Inference exception is caught and returns is_valid=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            # Create a model that raises on inference
            bad_model = Mock()
            bad_model.predict_proba = Mock(side_effect=RuntimeError("Model error!"))
            artifact = {"model": bad_model, "version": "1.0.0"}
            with open(model_path, "wb") as f:
                pickle.dump(artifact, f)
            
            engine = MLInferenceEngine(model_path=str(model_path))
            result = engine.predict(valid_feature_frame)
            
            assert result.is_valid is False
            assert result.confidence == 0.0
            assert result.status_code == "INFERENCE_ERROR"
            assert "Model error!" in result.reason


class TestMLInferenceMultiPair:
    """Test batch prediction behavior."""
    
    def test_batch_predict_all_invalid_when_model_unavailable(self, valid_feature_frame):
        """Batch prediction returns all is_valid=False when model unavailable."""
        with patch('tokocrypto_bot.ml.inference.resolve_and_validate_model_path') as mock_resolve:
            mock_resolve.return_value = (None, False)
            engine = MLInferenceEngine()
            
            frames = {
                "BTCUSDT": valid_feature_frame,
                "ETHUSDT": valid_feature_frame,
                "BNBUSDT": valid_feature_frame,
            }
            
            results = engine.predict_multi_pair(frames)
            
            assert len(results) == 3
            for symbol, result in results.items():
                assert result.is_valid is False
                assert result.confidence == 0.0
                assert result.symbol == symbol
    
    def test_batch_predict_valid_model_processes_all_pairs(self, valid_feature_frame, mock_sklearn_model):
        """Batch prediction with valid model processes all pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            artifact = {"model": mock_sklearn_model, "version": "1.0.0"}
            with open(model_path, "wb") as f:
                pickle.dump(artifact, f)
            
            engine = MLInferenceEngine(model_path=str(model_path))
            
            frames = {
                "BTCUSDT": valid_feature_frame,
                "ETHUSDT": valid_feature_frame,
            }
            
            results = engine.predict_multi_pair(frames)
            
            assert len(results) == 2
            for symbol, result in results.items():
                assert result.is_valid is True
                assert result.symbol == symbol
                assert result.probability_up == 0.7
                assert result.probability_down == 0.3


class TestFailClosedContract:
    """Test that the fail-closed contract is maintained across all edge cases."""
    
    def test_no_exception_can_make_invalid_model_produce_valid_prediction(self):
        """Verify that no exception or edge case can bypass the is_valid=False contract."""
        with patch('tokocrypto_bot.ml.inference.resolve_and_validate_model_path') as mock_resolve:
            mock_resolve.return_value = (None, False)  # Model unavailable
            engine = MLInferenceEngine()
            
            # Try various feature frames
            test_cases = [
                # Valid frame
                FeatureFrame(
                    timestamp=1693497600000,
                    symbol="BTCUSDT",
                    feature_version=FEATURE_VERSION,
                    features={
                        "EMA50": 100.5, "EMA200": 99.8, "RSI14": 55.0, "ROC": 0.05, "ATR": 2.0,
                        "volatility_regime": 0.02, "MACD_HIST": 0.1, "DI_plus": 25.0, "DI_minus": 20.0,
                        "ema_ratio": 1.007, "bb_pband": 0.5, "obv_vs_ma": 1000.0, "cmf": 0.3,
                        "vwma_dev": 0.02, "drawdown_20": -0.05
                    },
                    is_valid=True
                ),
                # Invalid frame
                FeatureFrame(
                    timestamp=0,
                    symbol="BTCUSDT",
                    feature_version=FEATURE_VERSION,
                    features={},
                    is_valid=False,
                    error_reason="Invalid"
                ),
                # Mismatched version
                FeatureFrame(
                    timestamp=1693497600000,
                    symbol="BTCUSDT",
                    feature_version="WRONG",
                    features={},
                    is_valid=True
                ),
            ]
            
            for frame in test_cases:
                result = engine.predict(frame)
                # Every single prediction must fail closed
                assert result.is_valid is False, f"Frame {frame.symbol} produced valid prediction when model unavailable"
                assert result.confidence == 0.0, f"Frame {frame.symbol} has non-zero confidence"
