"""
Blofin Stack Feature Library
=============================

Modular feature engineering system for crypto trading.

Modules:
- feature_manager: Central API for feature computation
- price_features: Basic price-based features
- volume_features: Volume-based features and indicators
- technical_indicators: Technical analysis indicators (RSI, MACD, etc.)
- volatility_features: Volatility measures (ATR, Bollinger Bands, etc.)
- market_regime: Market regime detection and classification
"""

from .feature_manager import FeatureManager

__version__ = "1.0.0"
__all__ = ["FeatureManager"]
