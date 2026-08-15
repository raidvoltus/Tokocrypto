"""
MODULE: tokocrypto_bot.strategy.features
DESCRIPTION: Deterministic Feature Engineering Engine for NVRA Trading System.
COMPATIBILITY: Accepts P1-A MarketDataEngine OHLCV Data.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("NVRA.FeatureEngine")

FEATURE_VERSION = "2026.1.0"
MIN_REQUIRED_CANDLES = 200

EXPECTED_FEATURE_COLUMNS = [
    "EMA50", "EMA200", "RSI14", "ROC", "ATR",
    "volatility_regime", "MACD_HIST", "DI_plus", "DI_minus",
    "ema_ratio", "bb_pband", "obv_vs_ma", "cmf", "vwma_dev", "drawdown_20"
]


@dataclass(frozen=True)
class FeatureFrame:
    timestamp: int
    symbol: str
    feature_version: str
    features: Dict[str, float]
    is_valid: bool
    error_reason: str = ""


class FeatureEngine:
    def __init__(self, feature_version: str = FEATURE_VERSION):
        self.feature_version = feature_version

    def compute_features(self, df_klines: pd.DataFrame, symbol: str) -> FeatureFrame:
        """
        Menghitung Feature Vector deterministik dari DataFrame OHLCV (Output P1-A).
        Hanya mengonsumsi candle yang sudah CLOSED untuk mencegah look-ahead bias.
        """
        if df_klines is None or df_klines.empty:
            return FeatureFrame(0, symbol, self.feature_version, {}, False, "Empty DataFrame provided.")

        # 1. Filter Hanya Closed Candles
        if "is_complete" in df_klines.columns:
            closed_df = df_klines[df_klines["is_complete"] == True].copy()
        else:
            closed_df = df_klines.iloc[:-1].copy()

        # 2. Minimum History Validation
        if len(closed_df) < MIN_REQUIRED_CANDLES:
            err_msg = f"Insufficient history. Required >= {MIN_REQUIRED_CANDLES}, got {len(closed_df)}"
            logger.warning(f"[{symbol}] {err_msg}")
            return FeatureFrame(0, symbol, self.feature_version, {}, False, err_msg)

        closed_df.sort_values("timestamp", ascending=True, inplace=True)
        closed_df.reset_index(drop=True, inplace=True)

        try:
            # 3. Indicator Computations
            close = closed_df["close"].values
            high = closed_df["high"].values
            low = closed_df["low"].values
            volume = closed_df["volume"].values

            # EMAs & Ratios
            ema50 = self._ema(close, 50)
            ema200 = self._ema(close, 200)
            ema_ratio = ema50 / np.where(ema200 == 0, np.nan, ema200)

            # RSI 14
            rsi14 = self._rsi(close, 14)

            # ROC 12
            roc = np.zeros_like(close)
            roc[12:] = (close[12:] - close[:-12]) / np.where(close[:-12] == 0, np.nan, close[:-12])

            # ATR 14
            tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
            tr = np.insert(tr, 0, high[0] - low[0])
            atr = self._ema(tr, 14)

            # Volatility Regime (Annualized Rolling Std / Mean)
            returns = np.diff(np.log(np.where(close == 0, 1e-8, close)))
            returns = np.insert(returns, 0, 0.0)
            volatility_regime = pd.Series(returns).rolling(20).std().fillna(0.0).values

            # MACD
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            signal_line = self._ema(macd_line, 9)
            macd_hist = macd_line - signal_line

            # Directional Movement (DMI / ADX helper)
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            plus_dm = np.insert(plus_dm, 0, 0.0)
            minus_dm = np.insert(minus_dm, 0, 0.0)
            
            atr_safe = np.where(atr == 0, 1e-8, atr)
            di_plus = (self._ema(plus_dm, 14) / atr_safe) * 100.0
            di_minus = (self._ema(minus_dm, 14) / atr_safe) * 100.0

            # Bollinger Bands %B
            sma20 = pd.Series(close).rolling(20).mean().values
            std20 = pd.Series(close).rolling(20).std().values
            upper_bb = sma20 + (2.0 * std20)
            lower_bb = sma20 - (2.0 * std20)
            bb_range = upper_bb - lower_bb
            bb_pband = (close - lower_bb) / np.where(bb_range == 0, np.nan, bb_range)

            # OBV vs MA
            obv = np.cumsum(np.sign(np.diff(close, prepend=close[0])) * volume)
            obv_ma = pd.Series(obv).rolling(20).mean().values
            obv_vs_ma = obv - obv_ma

            # CMF (Chaikin Money Flow 20)
            mfv = np.where((high - low) == 0, 0.0, ((close - low) - (high - close)) / (high - low)) * volume
            cmf = pd.Series(mfv).rolling(20).sum().values / np.maximum(pd.Series(volume).rolling(20).sum().values, 1e-8)

            # VWMA Dev
            vwma = pd.Series(close * volume).rolling(20).sum().values / np.maximum(pd.Series(volume).rolling(20).sum().values, 1e-8)
            vwma_dev = (close - vwma) / np.where(vwma == 0, np.nan, vwma)

            # Drawdown 20
            rolling_max = pd.Series(high).rolling(20).max().values
            drawdown_20 = (close - rolling_max) / np.where(rolling_max == 0, np.nan, rolling_max)

            # Extract Latest Closed Candle Features (-1)
            idx = -1
            latest_timestamp = int(closed_df["timestamp"].iloc[idx])

            raw_features = {
                "EMA50": float(ema50[idx]),
                "EMA200": float(ema200[idx]),
                "RSI14": float(rsi14[idx]),
                "ROC": float(roc[idx]),
                "ATR": float(atr[idx]),
                "volatility_regime": float(volatility_regime[idx]),
                "MACD_HIST": float(macd_hist[idx]),
                "DI_plus": float(di_plus[idx]),
                "DI_minus": float(di_minus[idx]),
                "ema_ratio": float(ema_ratio[idx]),
                "bb_pband": float(bb_pband[idx]),
                "obv_vs_ma": float(obv_vs_ma[idx]),
                "cmf": float(cmf[idx]),
                "vwma_dev": float(vwma_dev[idx]),
                "drawdown_20": float(drawdown_20[idx])
            }

            # 4. Handling & Validation NaN / Inf
            for col, val in raw_features.items():
                if np.isnan(val) or np.isinf(val):
                    err = f"NaN/Inf detected in feature [{col}]: {val}"
                    logger.error(f"[{symbol}] {err}")
                    return FeatureFrame(latest_timestamp, symbol, self.feature_version, {}, False, err)

            return FeatureFrame(
                timestamp=latest_timestamp,
                symbol=symbol,
                feature_version=self.feature_version,
                features=raw_features,
                is_valid=True
            )

        except Exception as e:
            logger.error(f"Error computing features for [{symbol}]: {e}", exc_info=True)
            return FeatureFrame(0, symbol, self.feature_version, {}, False, f"Exception: {str(e)}")

    def _ema(self, values: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(values).ewm(span=period, adjust=False).mean().values

    def _rsi(self, values: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(values)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        gain_series = pd.Series(gain).ewm(alpha=1.0/period, adjust=False).mean()
        loss_series = pd.Series(loss).ewm(alpha=1.0/period, adjust=False).mean()

        rs = gain_series / np.where(loss_series == 0, 1e-8, loss_series)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return np.insert(rsi.values, 0, 50.0)
