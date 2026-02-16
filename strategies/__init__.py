#!/usr/bin/env python3
"""
Strategy plugin system with auto-discovery.
"""
from pathlib import Path
from typing import List, Type
import importlib
import inspect

from .base_strategy import BaseStrategy, Signal

# Import all strategy classes
from .momentum import MomentumStrategy
from .breakout import BreakoutStrategy
from .reversal import ReversalStrategy
from .vwap_reversion import VWAPReversionStrategy
from .rsi_divergence import RSIDivergenceStrategy
from .bb_squeeze import BBSqueezeStrategy
from .ema_crossover import EMACrossoverStrategy
from .volume_spike import VolumeSpikeStrategy
from .support_resistance import SupportResistanceStrategy
from .macd_divergence import MACDDivergenceStrategy
from .candle_patterns import CandlePatternsStrategy


def get_all_strategies() -> List[BaseStrategy]:
    """
    Auto-discover and instantiate all strategy classes.
    
    Returns:
        List of instantiated strategy objects
    """
    strategies = [
        MomentumStrategy(),
        BreakoutStrategy(),
        ReversalStrategy(),
        VWAPReversionStrategy(),
        RSIDivergenceStrategy(),
        BBSqueezeStrategy(),
        EMACrossoverStrategy(),
        VolumeSpikeStrategy(),
        SupportResistanceStrategy(),
        MACDDivergenceStrategy(),
        CandlePatternsStrategy(),
    ]
    
    return strategies


__all__ = [
    'BaseStrategy',
    'Signal',
    'get_all_strategies',
    'MomentumStrategy',
    'BreakoutStrategy',
    'ReversalStrategy',
    'VWAPReversionStrategy',
    'RSIDivergenceStrategy',
    'BBSqueezeStrategy',
    'EMACrossoverStrategy',
    'VolumeSpikeStrategy',
    'SupportResistanceStrategy',
    'MACDDivergenceStrategy',
    'CandlePatternsStrategy',
]
