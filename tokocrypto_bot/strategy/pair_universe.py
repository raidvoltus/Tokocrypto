"""
MODULE: tokocrypto_bot.strategy.pair_universe
DESCRIPTION: Dynamic Pair Discovery, Filtering, and Ranking Engine for Tokocrypto.
"""

import time
import requests
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger("NVRA.PairUniverse")


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    min_price: float
    max_price: float
    tick_size: float
    min_qty: float
    max_qty: float
    step_size: float
    min_notional: float
    is_spot_trading_allowed: bool


@dataclass(frozen=True)
class PairUniverseConfig:
    allowed_quote_assets: Set[str] = field(default_factory=lambda: {"USDT", "BIDR", "IDR", "BTC", "ETH"})
    min_24h_volume_usdt: float = 50000.0  # Minimal Volume 24 jam setara $50,000 USDT
    max_active_pairs: int = 50  # Batas maksimal pair aktif yang dipindai per siklus
    cache_ttl_seconds: float = 3600.0  # Refresh exchangeInfo setiap 1 jam


class PairUniverseEngine:
    def __init__(
        self,
        base_url: str = "https://api.tokocrypto.com",
        config: Optional[PairUniverseConfig] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.config = config or PairUniverseConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NVRA-PairUniverse/2026.5"})

        self._cached_universe: List[SymbolRules] = []
        self._last_update_time: float = 0.0

    def get_active_universe(self, force_refresh: bool = False) -> List[SymbolRules]:
        """
        Mengembalikan daftar Pair Aktif terverifikasi yang memenuhi kriteria likuiditas & filter exchange.
        """
        now = time.time()
        if not force_refresh and self._cached_universe and (now - self._last_update_time < self.config.cache_ttl_seconds):
            return self._cached_universe

        logger.info("Fetching fresh exchangeInfo and 24h ticker data from Tokocrypto...")
        symbol_rules = self._fetch_exchange_info()
        if not symbol_rules:
            logger.warning("Failed to fetch exchangeInfo. Utilizing cached universe if available.")
            return self._cached_universe

        # Volume & Ticker Filtering
        ticker_24h = self._fetch_24h_tickers()
        active_universe = []

        for rule in symbol_rules:
            if not rule.is_spot_trading_allowed or rule.status != "TRADING":
                continue

            if rule.quote_asset not in self.config.allowed_quote_assets:
                continue

            # Filtering Volume 24 Jam
            volume_24h_usdt = ticker_24h.get(rule.symbol, 0.0)
            if volume_24h_usdt < self.config.min_24h_volume_usdt:
                continue

            active_universe.append(rule)

        # Ranking Berdasarkan Volume (Top Liquidity First)
        active_universe.sort(key=lambda r: ticker_24h.get(r.symbol, 0.0), reverse=True)
        active_universe = active_universe[:self.config.max_active_pairs]

        self._cached_universe = active_universe
        self._last_update_time = now

        logger.info(f"Dynamic Pair Universe updated: {len(active_universe)} active pairs selected for trading scanning.")
        return self._cached_universe

    def _fetch_exchange_info(self) -> List[SymbolRules]:
        endpoint = f"{self.base_url}/api/v3/exchangeInfo"
        try:
            res = self.session.get(endpoint, timeout=10.0)
            res.raise_for_status()
            data = res.json()
            
            rules_list = []
            for s in data.get("symbols", []):
                symbol = s["symbol"]
                base_asset = s["baseAsset"]
                quote_asset = s["quoteAsset"]
                status = s["status"]
                is_spot = s.get("isSpotTradingAllowed", True)

                # Extract Filters (PRICE_FILTER, LOT_SIZE, MIN_NOTIONAL)
                min_price = tick_size = max_price = 0.0
                min_qty = step_size = max_qty = 0.0
                min_notional = 10.0  # Default notional $10

                for f in s.get("filters", []):
                    f_type = f.get("filterType")
                    if f_type == "PRICE_FILTER":
                        min_price = float(f.get("minPrice", 0))
                        max_price = float(f.get("maxPrice", 0))
                        tick_size = float(f.get("tickSize", 0))
                    elif f_type == "LOT_SIZE":
                        min_qty = float(f.get("minQty", 0))
                        max_qty = float(f.get("maxQty", 0))
                        step_size = float(f.get("stepSize", 0))
                    elif f_type in ("MIN_NOTIONAL", "NOTIONAL"):
                        min_notional = float(f.get("minNotional", f.get("notional", 10.0)))

                rules = SymbolRules(
                    symbol=symbol,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    status=status,
                    min_price=min_price,
                    max_price=max_price,
                    tick_size=tick_size,
                    min_qty=min_qty,
                    max_qty=max_qty,
                    step_size=step_size,
                    min_notional=min_notional,
                    is_spot_trading_allowed=is_spot
                )
                rules_list.append(rules)

            return rules_list
        except Exception as e:
            logger.error(f"Error fetching exchangeInfo: {e}")
            return []

    def _fetch_24h_tickers(self) -> Dict[str, float]:
        """Mengambil data 24h quoteVolume untuk seluruh pair."""
        endpoint = f"{self.base_url}/api/v3/ticker/24hr"
        try:
            res = self.session.get(endpoint, timeout=10.0)
            res.raise_for_status()
            data = res.json()
            
            volumes = {}
            for item in data:
                symbol = item["symbol"]
                # Quote volume memberikan estimasi nilai transaksi dalam mata uang kutipan
                quote_volume = float(item.get("quoteVolume", 0.0))
                volumes[symbol] = quote_volume
            return volumes
        except Exception as e:
            logger.error(f"Error fetching 24h ticker data: {e}")
            return {}
