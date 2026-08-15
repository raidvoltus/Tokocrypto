"""
MODULE: tokocrypto_bot.strategy.strategies.base
DESCRIPTION: Abstract Base Class and Immutable Contract for all NVRA Trading Strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, List, Optional
from enum import Enum

from tokocrypto_bot.strategy.features import FeatureFrame
from tokocrypto_bot.strategy.regime import MarketRegime


class StrategySignalSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class CandidateSignal:
    strategy_name: str
    symbol: str
    timestamp: int
    side: StrategySignalSide
    raw_confidence: float
    expected_value: float
    stop_loss_pct: float
    take_profit_pct: float
    reason: str


class BaseStrategy(ABC):
    def __init__(self, name: str, version: str, timeframe: str = "5m"):
        self.name = name
        self.version = version
        self.timeframe = timeframe
        self.is_enabled = True

    @property
    @abstractmethod
    def applicable_regimes(self) -> Set[MarketRegime]:
        """Daftar MarketRegime yang valid untuk strategi ini."""
        pass

    @abstractmethod
    def generate_candidate_signal(self, feature_frame: FeatureFrame) -> Optional[CandidateSignal]:
        """Menghasilkan kandidat sinyal jika syarat teknikal strategi terpenuhi."""
        pass
