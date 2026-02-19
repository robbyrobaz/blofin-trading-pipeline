#!/usr/bin/env python3
"""
ML Gradient Boosted Tree strategy on 5m features.
Loads a trained GBT model if available; falls back to a multi-factor heuristic
that approximates what a GBT learns from these features.

Features used:
  - 5m return (momentum)
  - RSI(14) on 5m prices
  - Bollinger Band position (z-score)
  - Volume ratio (current vs. 5m mean)
"""
import os
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class MLGbt5mStrategy(BaseStrategy):
    """Gradient boosted tree entry signals on 5m features."""

    name = "ml_gbt_5m"
    version = "1.0"
    description = "Gradient boosted tree on 5m features (returns, RSI, BB position, volume ratio)"

    def __init__(self):
        self.window_seconds = int(os.getenv("ML_GBT_WINDOW_SECONDS", "300"))
        self.buy_threshold = float(os.getenv("ML_GBT_BUY_THRESHOLD", "0.60"))
        self.sell_threshold = float(os.getenv("ML_GBT_SELL_THRESHOLD", "0.60"))
        self.model_path = os.getenv(
            "ML_GBT_MODEL_PATH",
            "/home/rob/.openclaw/workspace/blofin-stack/ml_pipeline/models/gbt_5m.pkl",
        )
        self._model = self._load_model()

    def _load_model(self):
        """Attempt to load a pickled sklearn model. Returns None if not found."""
        path = Path(self.model_path)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _compute_features(
        self,
        price: float,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]],
        ts_ms: int,
    ) -> Optional[Dict[str, float]]:
        """Extract 5m features. Returns None if insufficient data."""
        window_p = self._slice_window(prices, ts_ms, self.window_seconds)
        window_v = self._slice_window(volumes, ts_ms, self.window_seconds)

        if len(window_p) < 10:
            return None

        price_vals = [p for _, p in window_p]
        vol_vals = [v for _, v in window_v] if window_v else []

        # 5m return
        first = window_p[0][1]
        ret_5m = (price - first) / first if first > 0 else 0.0

        # RSI(14) on 5m prices
        rsi = self._compute_rsi(price_vals)

        # Bollinger Band z-score
        mean = sum(price_vals) / len(price_vals)
        variance = sum((p - mean) ** 2 for p in price_vals) / len(price_vals)
        std = variance ** 0.5
        bb_z = (price - mean) / std if std > 0 else 0.0

        # Volume ratio: current vs. 5m mean
        if vol_vals:
            mean_vol = sum(vol_vals) / len(vol_vals)
            vol_ratio = vol_vals[-1] / mean_vol if mean_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        return {
            "ret_5m": ret_5m,
            "rsi": rsi,
            "bb_z": bb_z,
            "vol_ratio": vol_ratio,
        }

    def _compute_rsi(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss > 0:
            return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        return 100.0 if avg_gain > 0 else 50.0

    def _heuristic_score(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Multi-factor heuristic approximating GBT learned weights.
        Returns (buy_score, sell_score) each in [0, 1].
        """
        ret = features["ret_5m"]
        rsi = features["rsi"]
        bb_z = features["bb_z"]
        vol_ratio = features["vol_ratio"]

        # Normalize inputs
        # Return: positive = bullish, negative = bearish
        ret_score = min(1.0, max(-1.0, ret / 0.02))  # scale: ±2% = ±1

        # RSI: <30 oversold (bullish), >70 overbought (bearish)
        rsi_buy_score = max(0.0, (30.0 - rsi) / 30.0)
        rsi_sell_score = max(0.0, (rsi - 70.0) / 30.0)

        # BB z-score: negative = below mean (potential buy), positive = above (potential sell)
        bb_buy_score = max(0.0, min(1.0, (-bb_z) / 2.0))
        bb_sell_score = max(0.0, min(1.0, bb_z / 2.0))

        # Volume: high volume confirms signal
        vol_conf = min(1.5, vol_ratio) / 1.5

        # Weights (approximating typical GBT feature importance for price prediction)
        # ret: 35%, RSI: 30%, BB: 25%, vol: 10%
        buy_raw = 0.35 * max(0.0, ret_score) + 0.30 * rsi_buy_score + 0.25 * bb_buy_score + 0.10 * vol_conf
        sell_raw = 0.35 * max(0.0, -ret_score) + 0.30 * rsi_sell_score + 0.25 * bb_sell_score + 0.10 * vol_conf

        # Normalize to [0, 1] — max raw is ~1.0 with all factors aligned
        buy_score = min(1.0, buy_raw)
        sell_score = min(1.0, sell_raw)

        return buy_score, sell_score

    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]],
    ) -> Optional[Signal]:
        features = self._compute_features(price, prices, volumes, ts_ms)
        if features is None:
            return None

        if self._model is not None:
            # Use trained model: expects [[ret_5m, rsi, bb_z, vol_ratio]]
            try:
                X = [[features["ret_5m"], features["rsi"], features["bb_z"], features["vol_ratio"]]]
                proba = self._model.predict_proba(X)[0]
                # Assume classes: 0=SELL, 1=HOLD, 2=BUY or binary 0=no, 1=BUY
                if len(proba) == 2:
                    buy_score = proba[1]
                    sell_score = 1.0 - proba[1]
                elif len(proba) == 3:
                    sell_score, _, buy_score = proba
                else:
                    buy_score, sell_score = self._heuristic_score(features)
            except Exception:
                buy_score, sell_score = self._heuristic_score(features)
        else:
            buy_score, sell_score = self._heuristic_score(features)

        if buy_score >= self.buy_threshold and buy_score > sell_score:
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=round(buy_score, 4),
                details={
                    "model": "gbt" if self._model else "heuristic",
                    "ret_5m": round(features["ret_5m"] * 100, 4),
                    "rsi": round(features["rsi"], 2),
                    "bb_z": round(features["bb_z"], 4),
                    "vol_ratio": round(features["vol_ratio"], 4),
                    "buy_score": round(buy_score, 4),
                },
            )

        if sell_score >= self.sell_threshold and sell_score > buy_score:
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=round(sell_score, 4),
                details={
                    "model": "gbt" if self._model else "heuristic",
                    "ret_5m": round(features["ret_5m"] * 100, 4),
                    "rsi": round(features["rsi"], 2),
                    "bb_z": round(features["bb_z"], 4),
                    "vol_ratio": round(features["vol_ratio"], 4),
                    "sell_score": round(sell_score, 4),
                },
            )

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "model_path": self.model_path,
            "model_loaded": self._model is not None,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if "window_seconds" in params:
            self.window_seconds = int(params["window_seconds"])
        if "buy_threshold" in params:
            self.buy_threshold = float(params["buy_threshold"])
        if "sell_threshold" in params:
            self.sell_threshold = float(params["sell_threshold"])
        if "model_path" in params:
            self.model_path = str(params["model_path"])
            self._model = self._load_model()
