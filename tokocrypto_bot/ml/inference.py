"""
MODULE: tokocrypto_bot.ml.inference
DESCRIPTION: Multi-Pair ML Inference Engine with Strict Fallback to NO_TRADE (P1-C).
"""

import os
import pickle
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from tokocrypto_bot.strategy.features import FeatureFrame, EXPECTED_FEATURE_COLUMNS, FEATURE_VERSION

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
        self.model_path = model_path or os.path.expandvars(r"%LOCALAPPDATA%\NVRA\Trading\models\champion_model.pkl")
        self.model: Optional[Any] = None
        self.model_version: str = "UNLOADED"
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            self.model = None
            self.model_version = "NONE"
            return

        try:
            with open(self.model_path, "rb") as f:
                artifact = pickle.load(f)

            if isinstance(artifact, dict) and "model" in artifact:
                self.model = artifact["model"]
                self.model_version = artifact.get("version", "1.0.0")
            else:
                self.model = artifact
                self.model_version = "LEGACY_1.0"
        except Exception as e:
            logger.error(f"Failed loading model: {e}")
            self.model = None
            self.model_version = "CORRUPT"

    def predict(self, feature_frame: FeatureFrame) -> PredictionResult:
        if self.model is None:
            return PredictionResult(
                symbol=feature_frame.symbol, timestamp=feature_frame.timestamp,
                model_version=self.model_version, feature_version=feature_frame.feature_version,
                probability_up=0.0, probability_down=0.0, confidence=0.0,
                is_valid=False, status_code="MODEL_UNAVAILABLE",
                reason="Champion model not available on disk."
            )

        if not feature_frame.is_valid:
            return PredictionResult(
                symbol=feature_frame.symbol, timestamp=feature_frame.timestamp,
                model_version=self.model_version, feature_version=feature_frame.feature_version,
                probability_up=0.0, probability_down=0.0, confidence=0.0,
                is_valid=False, status_code="INVALID_INPUT",
                reason=f"FeatureFrame invalid: {feature_frame.error_reason}"
            )

        if feature_frame.feature_version != FEATURE_VERSION:
            return PredictionResult(
                symbol=feature_frame.symbol, timestamp=feature_frame.timestamp,
                model_version=self.model_version, feature_version=feature_frame.feature_version,
                probability_up=0.0, probability_down=0.0, confidence=0.0,
                is_valid=False, status_code="FEATURE_MISMATCH",
                reason=f"Feature version mismatch ({FEATURE_VERSION} vs {feature_frame.feature_version})"
            )

        try:
            vector = [feature_frame.features[col] for col in EXPECTED_FEATURE_COLUMNS]
        except KeyError as e:
            return PredictionResult(
                symbol=feature_frame.symbol, timestamp=feature_frame.timestamp,
                model_version=self.model_version, feature_version=feature_frame.feature_version,
                probability_up=0.0, probability_down=0.0, confidence=0.0,
                is_valid=False, status_code="FEATURE_MISMATCH",
                reason=f"Missing feature column: {e}"
            )

        X = np.array([vector], dtype=np.float64)
        if np.isnan(X).any() or np.isinf(X).any():
            return PredictionResult(
                symbol=feature_frame.symbol, timestamp=feature_frame.timestamp,
                model_version=self.model_version, feature_version=feature_frame.feature_version,
                probability_up=0.0, probability_down=0.0, confidence=0.0,
                is_valid=False, status_code="INVALID_INPUT",
                reason="NaN or Inf in feature vector."
            )

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
                symbol=feature_frame.symbol, timestamp=feature_frame.timestamp,
                model_version=self.model_version, feature_version=feature_frame.feature_version,
                probability_up=prob_up, probability_down=prob_down, confidence=confidence,
                is_valid=True, status_code="OK", reason="Inference OK"
            )
        except Exception as e:
            return PredictionResult(
                symbol=feature_frame.symbol, timestamp=feature_frame.timestamp,
                model_version=self.model_version, feature_version=feature_frame.feature_version,
                probability_up=0.0, probability_down=0.0, confidence=0.0,
                is_valid=False, status_code="INFERENCE_ERROR", reason=f"Model error: {e}"
            )

    def predict_multi_pair(self, feature_frames: Dict[str, FeatureFrame]) -> Dict[str, PredictionResult]:
        """Batch inference untuk seluruh pair yang tersedia."""
        return {sym: self.predict(ff) for sym, ff in feature_frames.items()}
