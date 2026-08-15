"""
MODULE: tokocrypto_bot.exchange.tokocrypto_client
DESCRIPTION: Direct Tokocrypto API v3 / open/v1 Client with HMAC-SHA256 Auth & Non-retry POST logic.
FIX P0-CRITICAL: Corrected syntax error (import hashlib)
"""

import hmac
import hashlib
import time
import requests
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode

logger = logging.getLogger("NVRA.TokocryptoClient")

class TokocryptoDirectClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.tokocrypto.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "User-Agent": "NVRA-TradingEngine/2026.5"
        })

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def fetch_account_balances(self) -> Dict[str, Dict[str, float]]:
        """GET /api/v3/account - Mengambil saldo dompet terverifikasi."""
        endpoint = f"{self.base_url}/api/v3/account"
        params = {"timestamp": int(time.time() * 1000)}
        params["signature"] = self._generate_signature(params)

        res = self.session.get(endpoint, params=params, timeout=10.0)
        res.raise_for_status()
        data = res.json()

        balances = {}
        for item in data.get("balances", []):
            asset = item["asset"]
            free = float(item["free"])
            locked = float(item["locked"])
            if free > 0 or locked > 0:
                balances[asset] = {"free": free, "locked": locked}
        return balances

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """GET /api/v3/openOrders - Mengambil order aktif."""
        endpoint = f"{self.base_url}/api/v3/openOrders"
        params = {"timestamp": int(time.time() * 1000)}
        if symbol:
            params["symbol"] = symbol
        params["signature"] = self._generate_signature(params)

        res = self.session.get(endpoint, params=params, timeout=10.0)
        res.raise_for_status()
        return res.json()

    def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        """GET /api/v3/order by origClientOrderId."""
        endpoint = f"{self.base_url}/api/v3/order"
        params = {
            "symbol": symbol,
            "origClientOrderId": client_order_id,
            "timestamp": int(time.time() * 1000)
        }
        params["signature"] = self._generate_signature(params)

        try:
            res = self.session.get(endpoint, params=params, timeout=10.0)
            if res.status_code == 404:
                return None
            res.raise_for_status()
            return res.json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (404, 400):
                return None
            raise e

    def fetch_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """GET /api/v3/myTrades - Mengambil riwayat fill eksekusi."""
        endpoint = f"{self.base_url}/api/v3/myTrades"
        params = {
            "symbol": symbol,
            "limit": limit,
            "timestamp": int(time.time() * 1000)
        }
        params["signature"] = self._generate_signature(params)

        res = self.session.get(endpoint, params=params, timeout=10.0)
        res.raise_for_status()
        return res.json()

    def post_order_non_retry(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        client_order_id: str
    ) -> Dict[str, Any]:
        """
        POST /api/v3/order - Mengirim request order tanpa blind retry.
        Jika terjadi Timeout, Exception dilempar agar caller mengisolasi status ke UNKNOWN.
        """
        endpoint = f"{self.base_url}/api/v3/order"
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
            "newClientOrderId": client_order_id,
            "timestamp": int(time.time() * 1000)
        }
        if price and order_type.upper() == "LIMIT":
            params["price"] = price
            params["timeInForce"] = "GTC"

        params["signature"] = self._generate_signature(params)

        # STRICT RULE: Single-shot HTTP POST without automated retry
        res = self.session.post(endpoint, data=params, timeout=8.0)
        res.raise_for_status()
        return res.json()
