#!/usr/bin/env python3
"""
ML Random Forest 15m Strategy

Random forest classifier trained on 20 technical features at 15m timeframe.
Trains on first 70% of available data per symbol.
Generates BUY/SELL signals only when confidence >= 60%.

Comparison target: ml_gbt_5m (GBT on 5m candles).

Reference: new_strategies_backtest.py → strategy_ml_random_forest_15m
"""

import os
import sqlite3
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy, Signal

DB_PATH = os.getenv("BLOFIN_DB_PATH", "data/blofin_monitor.db")
CONFIDENCE_THRESHOLD = float(os.getenv("RF15M_CONFIDENCE_THRESH", "0.60"))
LOOKBACK_DAYS = int(os.getenv("RF15M_LOOKBACK_DAYS", "10"))


class MLRandomForest15mStrategy(BaseStrategy):
    """
    Random Forest classifier on 15-minute OHLCV candles.

    Features (20):
      RSI(7,14,21), EMA ratios (5/50, 10/50, 20/50),
      price vs EMA, BB position, normalized ATR,
      ROC(3,5,10), candle body/wicks, volume z-score,
      1-bar return, 2-bar lagged return.

    Trains once per symbol on startup (lazy, on first signal request),
    then operates in inference mode.
    """

    name = "ml_random_forest_15m"
    version = "1.0"
    description = "Random Forest classifier on 15m candles with 20 features"

    def __init__(self):
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.lookback_days = LOOKBACK_DAYS
        self._models: Dict[str, Any] = {}   # symbol → (model, scaler, last_trained_ts)
        self._retrain_interval_hours = 24
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            return True
        except ImportError:
            return False

    def _load_ohlcv_15m(self, symbol: str) -> Optional[np.ndarray]:
        """Load 15m OHLCV candles from DB."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            end_ts = int(datetime.now().timestamp() * 1000)
            start_ts = end_ts - (self.lookback_days * 24 * 3600 * 1000)
            c.execute(
                "SELECT ts_ms, price FROM ticks WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC",
                (symbol, start_ts, end_ts)
            )
            rows = c.fetchall()
            conn.close()

            if len(rows) < 100:
                return None

            ticks = np.array(rows, dtype=float)
            period_ms = 15 * 60 * 1000
            first_period = (ticks[0, 0] // period_ms) * period_ms
            pidx = ((ticks[:, 0] - first_period) // period_ms).astype(int)

            candles = []
            for pid in np.unique(pidx):
                mask = pidx == pid
                pts = ticks[mask]
                if len(pts) < 2:
                    continue
                candles.append([
                    pts[0, 0], pts[0, 1],
                    pts[:, 1].max(), pts[:, 1].min(),
                    pts[-1, 1], float(len(pts))
                ])

            return np.array(candles) if len(candles) >= 50 else None
        except Exception:
            return None

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
            result[i] = 100 - (100 / (1 + ag / al)) if al > 0 else 100
        return result

    def _atr(self, high, low, close, period=14):
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
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

    def _build_features(self, candles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) feature matrix."""
        n = len(candles)
        if n < 55:
            return np.array([]), np.array([])

        open_  = candles[:, 1]
        high   = candles[:, 2]
        low    = candles[:, 3]
        close  = candles[:, 4]
        volume = candles[:, 5]

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
            vol_z  = (volume[i] - vol_s20[i]) / (np.std(volume[max(0, i - 20):i]) + 1e-8)
            ret1   = (close[i] - close[i - 1]) / close[i - 1] * 100 if close[i - 1] > 0 else 0
            ret2   = (close[i - 1] - close[i - 2]) / close[i - 2] * 100 if close[i - 2] > 0 else 0

            features.append([
                rsi7[i] / 100, rsi14[i] / 100, rsi21[i] / 100,
                (e5[i]  - e50[i]) / e50[i],
                (e10[i] - e50[i]) / e50[i],
                (e20[i] - e50[i]) / e50[i],
                (close[i] - e5[i])  / e5[i],
                (close[i] - e20[i]) / e20[i],
                bb_pos,
                atr14[i] / close[i],
                roc3 / 10, roc5 / 10, roc10 / 10,
                body / 5, wick_u / 3, wick_l / 3,
                min(vol_z / 3, 5.0),
                (high[i] - low[i]) / close[i] * 100,
                ret1, ret2,
            ])
            targets.append(1 if close[i + 1] > close[i] else 0)

        return (np.array(features), np.array(targets)) if features else (np.array([]), np.array([]))

    def _train_model(self, symbol: str):
        """Train RF model for a symbol. Called lazily on first detect()."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        candles = self._load_ohlcv_15m(symbol)
        if candles is None or len(candles) < 150:
            return

        X, y = self._build_features(candles)
        if len(X) < 100:
            return

        split = int(len(X) * 0.70)
        X_train = X[:split]
        y_train = y[:split]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        rf.fit(X_train_s, y_train)
        self._models[symbol] = {
            'model': rf, 'scaler': scaler,
            'candles': candles,   # keep reference for inference features
            'trained_at': datetime.now().timestamp(),
        }

    def _should_retrain(self, symbol: str) -> bool:
        if symbol not in self._models:
            return True
        trained_at = self._models[symbol].get('trained_at', 0)
        return (datetime.now().timestamp() - trained_at) > self._retrain_interval_hours * 3600

    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:

        if not self._sklearn_available:
            return None

        if self._should_retrain(symbol):
            self._train_model(symbol)

        if symbol not in self._models:
            return None

        model_data = self._models[symbol]
        model  = model_data['model']
        scaler = model_data['scaler']

        # Build feature vector from recent price history
        window_60m = self._slice_window(prices, ts_ms, 90 * 60)
        if len(window_60m) < 60:
            return None

        # Assemble pseudo-candle array from window
        px_arr = np.array([p for _, p in window_60m])
        n = len(px_arr)
        if n < 55:
            return None

        # Derive arrays for feature building
        # Create a minimal candle-like structure from the tick stream
        fake_candles = np.zeros((n, 6))
        for i in range(n):
            fake_candles[i] = [0, px_arr[i], px_arr[i], px_arr[i], px_arr[i], 1.0]

        X, _ = self._build_features(fake_candles)
        if len(X) == 0:
            return None

        # Use last available feature vector
        feat = X[-1:].copy()
        feat_s = scaler.transform(feat)
        prob = model.predict_proba(feat_s)[0, 1]

        if prob >= self.confidence_threshold:
            return Signal(
                symbol=symbol, signal="BUY", strategy=self.name,
                confidence=round(prob, 3),
                details={"prob_up": round(prob, 3), "timeframe": "15m"}
            )
        elif prob <= (1 - self.confidence_threshold):
            return Signal(
                symbol=symbol, signal="SELL", strategy=self.name,
                confidence=round(1 - prob, 3),
                details={"prob_up": round(prob, 3), "timeframe": "15m"}
            )

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "lookback_days": self.lookback_days,
            "retrain_interval_hours": self._retrain_interval_hours,
            "trained_symbols": list(self._models.keys()),
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if "confidence_threshold" in params:
            self.confidence_threshold = float(params["confidence_threshold"])
        if "lookback_days" in params:
            self.lookback_days = int(params["lookback_days"])
        if "retrain_interval_hours" in params:
            self._retrain_interval_hours = float(params["retrain_interval_hours"])
