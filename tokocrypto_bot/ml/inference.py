"""
MODULE: tokocrypto_bot.ml.inference
DESCRIPTION: Hardened ML Inference Engine supporting MODEL_UNAVAILABLE state and strict NO_TRADE policy.
COMPATIBILITY: Consumes FeatureFrame from P1-B.
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
    status_code: str  # OK, MODEL_UNAVAILABLE, FEATURE_MISMATCH, INVALID_INPUT, INFERENCE_ERROR
    reason: str = ""


class MLInferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.expandvars(r"%LOCALAPPDATA%\NVRA\Trading\models\champion_model.pkl")
        self.model: Optional[Any] = None
        self.model_version: str = "UNLOADED"
        self._load_model()

    def _load_model(self) -> None:
        """Memuat model persisten dari disk jika tersedia."""
        if not os.path.exists(self.model_path):
            logger.info(f"Model file not found at {self.model_path}. Engine operating in MODEL_UNAVAILABLE state.")
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

            logger.info(f"Loaded ML Champion Model v{self.model_version} successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            self.model = None
            self.model_version = "CORRUPT"

    def predict(self, feature_frame: FeatureFrame) -> PredictionResult:
        """
        Menghasilkan probabilitas prediksi deterministik berbasis FeatureFrame P1-B.
        STRICT FAIL-SAFE POLICY: Segala kegagalan menghasilkan status is_valid=False (NO_TRADE).
        """
        # 1. State Model Missing Check
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
                reason="Champion model is not available on disk. Triggering NO_TRADE."
            )

        # 2. Validation Input Feature Frame
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
                reason=f"FeatureFrame is invalid: {feature_frame.error_reason}"
            )

        # 3. Feature Version & Column Compatibility Check
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
                reason=f"Feature version mismatch (Expected {FEATURE_VERSION}, got {feature_frame.feature_version})"
            )

        # Construct Stable Feature Vector
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
                reason=f"Missing expected feature column: {e}"
            )

        # 4. Check NaN / Inf
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
                reason="NaN or Inf detected in inference vector."
            )

        # 5. Model Inference Execution
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
                reason="Inference successful."
            )
        except Exception as e:
            logger.error(f"Inference execution failed for [{feature_frame.symbol}]: {e}", exc_info=True)
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
                reason=f"Model execution error: {str(e)}"
            )
