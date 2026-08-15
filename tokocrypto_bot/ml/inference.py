"""
MODULE: tokocrypto_bot.ml.inference
DESCRIPTION: Multi-Pair ML Inference Engine with Strict Fallback to NO_TRADE (P1-C).
             Uses cross-platform model_loader for model resolution and integrity validation.
FIX P1-CRITICAL: Refactored to use model_loader; fails closed to NO_TRADE if model unavailable.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from tokocrypto_bot.strategy.features import FeatureFrame, EXPECTED_FEATURE_COLUMNS, FEATURE_VERSION
from tokocrypto_bot.ml.model_loader import resolve_and_validate_model_path

logger = logging.getLogger("NVRA.MLInference")


@dataclass(frozen=True)
class PredictionResult:
    symbol: str
    timestamp: int
    model_version: str
    feature_version: str
    probability_up: float
    probability_down: float
    confidence: float
    is_valid: bool
    status_code: str
    reason: str = ""


class MLInferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML Inference Engine.
        
        Args:
            model_path: Optional explicit model path override.
                       If provided, used instead of environment/default resolution.
                       For testing only.
        
        Model loading strategy:
        1. If model_path argument provided: use it directly
        2. Otherwise: use model_loader resolution (NVRA_MODEL_PATH, repo, app-data)
        3. If model unavailable/invalid: is_valid=False (fail closed)
        """
        self.model: Optional[Any] = None
        self.model_version: str = "UNLOADED"
        self.model_path: Optional[str] = None
        
        # Determine which path to use
        if model_path:
            # Explicit override (testing)
            resolved_path, is_valid = model_path, True
        else:
            # Use model_loader resolution with integrity validation
            resolved_path, is_valid = resolve_and_validate_model_path()
        
        if resolved_path and is_valid:
            self.model_path = str(resolved_path)
            self._load_model()
        else:
            # Model unavailable or invalid: fail closed
            logger.warning(
                "ML model unavailable or integrity check failed. "
                "Inference will return is_valid=False; trading will default to NO_TRADE."
            )
            self.model = None
            self.model_version = "UNAVAILABLE"

    def _load_model(self) -> None:
        """
        Load model from disk.
        
        Model is expected to be either:
        1. A dict with 'model' and 'version' keys
        2. A raw sklearn/xgboost model object
        
        If loading fails for any reason, model remains None (fail closed).
        """
        if not self.model_path:
            return
        
        try:
            import pickle
            with open(self.model_path, "rb") as f:
                artifact = pickle.load(f)
            
            if isinstance(artifact, dict) and "model" in artifact:
                self.model = artifact["model"]
                self.model_version = artifact.get("version", "1.0.0")
                logger.info(f"Model loaded successfully: version={self.model_version}")
            else:
                self.model = artifact
                self.model_version = "LEGACY_1.0"
                logger.info(f"Legacy model loaded (no version metadata)")
        except Exception as e:
            logger.error(f"Failed loading model from {self.model_path}: {e}")
            self.model = None
            self.model_version = "CORRUPT"

    def predict(self, feature_frame: FeatureFrame) -> PredictionResult:
        """
        Generate prediction for a single pair.
        
        Fail-closed behavior:
        - If model unavailable: is_valid=False
        - If feature frame invalid: is_valid=False
        - If feature version mismatch: is_valid=False
        - If inference error: is_valid=False
        
        All invalid predictions produce confidence=0.0, preventing BUY/SELL decisions.
        """
        # Gate 1: Model availability
        if self.model is None:
            return PredictionResult(
                symbol=feature_frame.symbol,
                timestamp=feature_frame.timestamp,
                model_version=self.model_version,
                feature_version=feature_frame.feature_version,
                probability_up=0.0,
                probability_down=0.0,
                confidence=0.0,
                is_valid=False,
                status_code="MODEL_UNAVAILABLE",
                reason="Champion model not loaded. Trading disabled."
            )

        # Gate 2: Feature frame validity
        if not feature_frame.is_valid:
            return PredictionResult(
                symbol=feature_frame.symbol,
                timestamp=feature_frame.timestamp,
                model_version=self.model_version,
                feature_version=feature_frame.feature_version,
                probability_up=0.0,
                probability_down=0.0,
                confidence=0.0,
                is_valid=False,
                status_code="INVALID_INPUT",
                reason=f"FeatureFrame invalid: {feature_frame.error_reason}"
            )

        # Gate 3: Feature version compatibility
        if feature_frame.feature_version != FEATURE_VERSION:
            return PredictionResult(
                symbol=feature_frame.symbol,
                timestamp=feature_frame.timestamp,
                model_version=self.model_version,
                feature_version=feature_frame.feature_version,
                probability_up=0.0,
                probability_down=0.0,
                confidence=0.0,
                is_valid=False,
                status_code="FEATURE_MISMATCH",
                reason=f"Feature version mismatch ({FEATURE_VERSION} vs {feature_frame.feature_version})"
            )

        # Gate 4: Feature column availability
        try:
            vector = [feature_frame.features[col] for col in EXPECTED_FEATURE_COLUMNS]
        except KeyError as e:
            return PredictionResult(
                symbol=feature_frame.symbol,
                timestamp=feature_frame.timestamp,
                model_version=self.model_version,
                feature_version=feature_frame.feature_version,
                probability_up=0.0,
                probability_down=0.0,
                confidence=0.0,
                is_valid=False,
                status_code="FEATURE_MISMATCH",
                reason=f"Missing feature column: {e}"
            )

        # Gate 5: Feature vector integrity (NaN/Inf)
        X = np.array([vector], dtype=np.float64)
        if np.isnan(X).any() or np.isinf(X).any():
            return PredictionResult(
                symbol=feature_frame.symbol,
                timestamp=feature_frame.timestamp,
                model_version=self.model_version,
                feature_version=feature_frame.feature_version,
                probability_up=0.0,
                probability_down=0.0,
                confidence=0.0,
                is_valid=False,
                status_code="INVALID_INPUT",
                reason="NaN or Inf in feature vector."
            )

        # Gate 6: Model inference execution
        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                prob_down = float(probs[0])
                prob_up = float(probs[1])
            else:
                pred = float(self.model.predict(X)[0])
                prob_up = max(0.0, min(1.0, pred))
                prob_down = 1.0 - prob_up

            confidence = abs(prob_up - prob_down)
            return PredictionResult(
                symbol=feature_frame.symbol,
                timestamp=feature_frame.timestamp,
                model_version=self.model_version,
                feature_version=feature_frame.feature_version,
                probability_up=prob_up,
                probability_down=prob_down,
                confidence=confidence,
                is_valid=True,
                status_code="OK",
                reason="Inference OK"
            )
        except Exception as e:
            logger.error(f"Inference exception for {feature_frame.symbol}: {e}")
            return PredictionResult(
                symbol=feature_frame.symbol,
                timestamp=feature_frame.timestamp,
                model_version=self.model_version,
                feature_version=feature_frame.feature_version,
                probability_up=0.0,
                probability_down=0.0,
                confidence=0.0,
                is_valid=False,
                status_code="INFERENCE_ERROR",
                reason=f"Model error: {e}"
            )

    def predict_multi_pair(self, feature_frames: Dict[str, FeatureFrame]) -> Dict[str, PredictionResult]:
        """Batch inference untuk seluruh pair yang tersedia."""
        return {sym: self.predict(ff) for sym, ff in feature_frames.items()}
