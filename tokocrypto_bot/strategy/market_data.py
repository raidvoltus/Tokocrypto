"""
MODULE: tokocrypto_bot.strategy.market_data
DESCRIPTION: Enhanced Market Data Engine with Fixed Fallback URL Config & Provenance.
FIXED: Corrected hardcoded fallback URL bug to use self.binance_fallback_url.
"""

import time
import requests
import logging
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger("NVRA.MarketData")


class DataSource(str, Enum):
    TOKOCRYPTO = "TOKOCRYPTO"
    BINANCE_PUBLIC_API = "BINANCE_PUBLIC_API"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class OHLCVFrame:
    timestamp: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: DataSource
    is_complete: bool


class MarketDataEngine:
    def __init__(
        self,
        tokocrypto_base_url: str = "https://api.tokocrypto.com",
        binance_fallback_url: str = "https://api.binance.com",  # Standard Direct Public API
        max_staleness_seconds: float = 300.0
    ):
        self.tokocrypto_url = tokocrypto_base_url.rstrip("/")
        self.binance_fallback_url = binance_fallback_url.rstrip("/")  # FIXED: Digunakan konsisten
        self.max_staleness_seconds = max_staleness_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NVRA-DataEngine/2026.5"})

    def fetch_klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[OHLCVFrame]:
        # 1. Primary Source: Tokocrypto API
        frames = self._fetch_tokocrypto_klines(symbol, interval, limit)
        if frames:
            return frames

        # 2. Fallback Source: Binance Public API (Murni Data Market)
        logger.warning(f"Tokocrypto market data failed for [{symbol}]. Activating Binance Fallback ({self.binance_fallback_url})...")
        frames = self._fetch_binance_fallback_klines(symbol, interval, limit)
        if frames:
            return frames

        logger.critical(f"All market data sources failed for [{symbol}]!")
        return []

    def get_klines_dataframe(self, symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
        frames = self.fetch_klines(symbol, interval, limit)
        if not frames:
            return pd.DataFrame()

        data = [
            {
                "timestamp": f.timestamp,
                "open": f.open,
                "high": f.high,
                "low": f.low,
                "close": f.close,
                "volume": f.volume,
                "source": f.source.value,
                "is_complete": f.is_complete
            }
            for f in frames
        ]
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def _fetch_tokocrypto_klines(self, symbol: str, interval: str, limit: int) -> Optional[List[OHLCVFrame]]:
        endpoint = f"{self.tokocrypto_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            res = self.session.get(endpoint, params=params, timeout=5.0)
            res.raise_for_status()
            raw_klines = res.json()
            return self._parse_raw_klines(symbol, raw_klines, DataSource.TOKOCRYPTO)
        except Exception as e:
            logger.error(f"Tokocrypto Klines fetch error for [{symbol}]: {e}")
            return None

    def _fetch_binance_fallback_klines(self, symbol: str, interval: str, limit: int) -> Optional[List[OHLCVFrame]]:
        # FIXED: Menggunakan self.binance_fallback_url sesuai konfigurasi constructor
        endpoint = f"{self.binance_fallback_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            res = self.session.get(endpoint, params=params, timeout=5.0)
            res.raise_for_status()
            raw_klines = res.json()
            return self._parse_raw_klines(symbol, raw_klines, DataSource.BINANCE_PUBLIC_API)
        except Exception as e:
            logger.error(f"Binance Fallback Klines fetch error for [{symbol}] via {endpoint}: {e}")
            return None

    def _parse_raw_klines(self, symbol: str, raw_klines: List[Any], source: DataSource) -> List[OHLCVFrame]:
        frames = []
        now_ms = int(time.time() * 1000)

        for k in raw_klines:
            open_time = int(k[0])
            close_time = int(k[6]) if len(k) > 6 else open_time + 59999
            is_complete = close_time < now_ms

            frame = OHLCVFrame(
                timestamp=open_time,
                symbol=symbol,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                source=source,
                is_complete=is_complete
            )
            frames.append(frame)

        if frames:
            last_candle_age_sec = (now_ms - frames[-1].timestamp) / 1000.0
            if last_candle_age_sec > self.max_staleness_seconds:
                logger.warning(f"Market data for [{symbol}] stale ({last_candle_age_sec:.1f}s > max {self.max_staleness_seconds}s)")

        return frames
