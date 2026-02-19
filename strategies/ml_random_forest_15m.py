#!/usr/bin/env python3
"""
ML Random Forest 15m Strategy — candle-based interface.

Random Forest classifier trained on 20 technical features.
Uses context_candles directly for both training (lazy, first call per symbol)
and inference. Requires sklearn; returns None gracefully if unavailable.
"""
import os
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime

from .base_strategy import BaseStrategy

CONFIDENCE_THRESHOLD = float(os.getenv("RF15M_CONFIDENCE_THRESH", "0.60"))


class MLRandomForest15mStrategy(BaseStrategy):
    """Random Forest classifier on OHLCV candles with 20 technical features."""

    name = "ml_random_forest_15m"
    version = "2.0"
    description = "Random Forest classifier on candle OHLCV data with 20 features"

    def __init__(self):
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self._models: Dict[str, Any] = {}
        self._retrain_interval_hours = 24
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        try:
            from sklearn.ensemble import RandomForestClassifier  # noqa
            from sklearn.preprocessing import StandardScaler     # noqa
            return True
        except ImportError:
            return False

    def _ema(self, prices, period):
        result = np.zeros(len(prices))
        k = 2.0 / (period + 1)
        result[0] = prices[0]
        for i in range(1, len(prices)):
            result[i] = prices[i] * k + result[i - 1] * (1 - k)
        return result

    def _rsi(self, prices, period=14):
        result = np.full(len(prices), np.nan)
        if len(prices) < period + 1:
            return result
        deltas = np.diff(prices)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        ag = np.mean(gains[:period])
        al = np.mean(losses[:period])
        for i in range(period, len(prices)):
            g = max(0.0, deltas[i - 1])
            l = max(0.0, -deltas[i - 1])
            ag = (ag * (period - 1) + g) / period
            al = (al * (period - 1) + l) / period
            result[i] = 100 - (100 / (1 + ag / al)) if al > 0 else 100.0
        return result

    def _atr(self, high, low, close, period=14):
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i]  - close[i-1]))
        result = np.full(n, np.nan)
        result[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
        return result

    def _sma(self, prices, period):
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    def _bollinger(self, prices, period=20, std_mult=2.0):
        mid = self._sma(prices, period)
        std = np.array([
            np.std(prices[max(0, i - period + 1):i + 1]) if i >= period - 1 else np.nan
            for i in range(len(prices))
        ])
        return mid + std_mult * std, mid, mid - std_mult * std

    def _candles_to_arrays(self, candles: list):
        """Convert list-of-dicts candles to numpy arrays."""
        n = len(candles)
        open_  = np.array([c['open']   for c in candles], dtype=float)
        high   = np.array([c['high']   for c in candles], dtype=float)
        low    = np.array([c['low']    for c in candles], dtype=float)
        close  = np.array([c['close']  for c in candles], dtype=float)
        volume = np.array([c['volume'] for c in candles], dtype=float)
        return open_, high, low, close, volume

    def _build_features(self, open_, high, low, close, volume):
        """Build (X, y) feature/target arrays. Needs ≥55 candles."""
        n = len(close)
        if n < 55:
            return np.array([]), np.array([])

        rsi7  = self._rsi(close, 7)
        rsi14 = self._rsi(close, 14)
        rsi21 = self._rsi(close, 21)
        e5    = self._ema(close, 5)
        e10   = self._ema(close, 10)
        e20   = self._ema(close, 20)
        e50   = self._ema(close, 50)
        bb_up, _, bb_low = self._bollinger(close, 20, 2.0)
        atr14  = self._atr(high, low, close, 14)
        vol_s20 = self._sma(volume, 20)

        features, targets = [], []
        for i in range(55, n - 1):
            if (np.isnan(rsi7[i]) or np.isnan(rsi14[i]) or np.isnan(rsi21[i]) or
                    np.isnan(bb_up[i]) or np.isnan(atr14[i]) or
                    np.isnan(vol_s20[i]) or vol_s20[i] == 0 or
                    np.isnan(e50[i]) or e50[i] == 0 or close[i] == 0):
                continue

            bb_range = bb_up[i] - bb_low[i]
            bb_pos   = (close[i] - bb_low[i]) / bb_range if bb_range > 0 else 0.5
            roc3  = (close[i] - close[i - 3])  / close[i - 3]  * 100 if close[i - 3]  > 0 else 0
            roc5  = (close[i] - close[i - 5])  / close[i - 5]  * 100 if close[i - 5]  > 0 else 0
            roc10 = (close[i] - close[i - 10]) / close[i - 10] * 100 if close[i - 10] > 0 else 0
            body   = (close[i] - open_[i]) / close[i] * 100
            wick_u = (high[i] - max(open_[i], close[i])) / close[i] * 100
            wick_l = (min(open_[i], close[i]) - low[i]) / close[i] * 100
            vol_z  = (volume[i] - vol_s20[i]) / (np.std(volume[max(0, i-20):i]) + 1e-8)
            ret1   = (close[i] - close[i - 1]) / close[i - 1] * 100 if close[i - 1] > 0 else 0
            ret2   = (close[i-1] - close[i-2]) / close[i-2] * 100 if close[i-2] > 0 else 0

            features.append([
                rsi7[i]/100, rsi14[i]/100, rsi21[i]/100,
                (e5[i]  - e50[i]) / e50[i],
                (e10[i] - e50[i]) / e50[i],
                (e20[i] - e50[i]) / e50[i],
                (close[i] - e5[i])  / e5[i],
                (close[i] - e20[i]) / e20[i],
                bb_pos,
                atr14[i] / close[i],
                roc3/10, roc5/10, roc10/10,
                body/5, wick_u/3, wick_l/3,
                min(vol_z/3, 5.0),
                (high[i] - low[i]) / close[i] * 100,
                ret1, ret2,
            ])
            targets.append(1 if close[i + 1] > close[i] else 0)

        if not features:
            return np.array([]), np.array([])
        return np.array(features), np.array(targets)

    def _train_model(self, symbol: str, candles: list):
        """Train RF model from provided candles."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        if len(candles) < 150:
            return

        open_, high, low, close, volume = self._candles_to_arrays(candles)
        X, y = self._build_features(open_, high, low, close, volume)
        if len(X) < 100:
            return

        split = int(len(X) * 0.70)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[:split])

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        rf.fit(X_train_s, y[:split])
        self._models[symbol] = {
            'model': rf, 'scaler': scaler,
            'trained_at': datetime.now().timestamp(),
        }

    def _should_retrain(self, symbol: str) -> bool:
        if symbol not in self._models:
            return True
        trained_at = self._models[symbol].get('trained_at', 0)
        return (datetime.now().timestamp() - trained_at) > self._retrain_interval_hours * 3600

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if not self._sklearn_available:
            return None

        if len(context_candles) < 60:
            return None

        if self._should_retrain(symbol):
            self._train_model(symbol, context_candles)

        if symbol not in self._models:
            return None

        model_data = self._models[symbol]
        model  = model_data['model']
        scaler = model_data['scaler']

        open_, high, low, close, volume = self._candles_to_arrays(context_candles)
        X, _ = self._build_features(open_, high, low, close, volume)
        if len(X) == 0:
            return None

        feat_s = scaler.transform(X[-1:])
        prob = model.predict_proba(feat_s)[0, 1]

        if prob >= self.confidence_threshold:
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(prob, 3),
                'prob_up': round(prob, 3),
                'timeframe': '15m',
            }
        elif prob <= (1 - self.confidence_threshold):
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(1 - prob, 3),
                'prob_up': round(prob, 3),
                'timeframe': '15m',
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "retrain_interval_hours": self._retrain_interval_hours,
            "trained_symbols": list(self._models.keys()),
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if "confidence_threshold" in params:
            self.confidence_threshold = float(params["confidence_threshold"])
        if "retrain_interval_hours" in params:
            self._retrain_interval_hours = float(params["retrain_interval_hours"])
