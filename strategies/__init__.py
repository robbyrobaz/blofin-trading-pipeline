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
from .volume_mean_reversion import VolumeMeanReversionStrategy

# New strategies (batch 2 — 2026-02-18)
from .cross_asset_correlation import CrossAssetCorrelationStrategy
from .volatility_regime_switch import VolatilityRegimeSwitchStrategy
from .ml_random_forest_15m import MLRandomForest15mStrategy
from .orderflow_imbalance import OrderflowImbalanceStrategy
from .ensemble_top3 import EnsembleTop3Strategy

# Ghost strategies implemented (2026-02-18)
from .mtf_trend_align import MTFTrendAlignStrategy
from .ml_gbt_5m import MLGbt5mStrategy
from .mtf_momentum_confirm import MTFMomentumConfirmStrategy


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
        VolumeMeanReversionStrategy(),
        # Batch 2: 2026-02-18
        CrossAssetCorrelationStrategy(),
        VolatilityRegimeSwitchStrategy(),
        MLRandomForest15mStrategy(),
        OrderflowImbalanceStrategy(),
        EnsembleTop3Strategy(),
        # Ghost strategies implemented: 2026-02-18
        MTFTrendAlignStrategy(),
        MLGbt5mStrategy(),
        MTFMomentumConfirmStrategy(),
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
    'VolumeMeanReversionStrategy',
    # Batch 2
    'CrossAssetCorrelationStrategy',
    'VolatilityRegimeSwitchStrategy',
    'MLRandomForest15mStrategy',
    'OrderflowImbalanceStrategy',
    'EnsembleTop3Strategy',
    # Ghost strategies implemented
    'MTFTrendAlignStrategy',
    'MLGbt5mStrategy',
    'MTFMomentumConfirmStrategy',
]
