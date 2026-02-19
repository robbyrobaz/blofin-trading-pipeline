#!/usr/bin/env python3
"""
ML Gradient Boosted Tree Strategy — candle-based interface.

Loads a trained sklearn GBT model if available; otherwise uses a
multi-factor heuristic that approximates what a well-trained GBT learns.

Features computed from candle data:
  - 5-bar return (short momentum)
  - RSI(14) on close
  - Bollinger Band z-score (close vs 20-SMA, normalized by std)
  - Volume ratio (current vs 20-bar average)
  - ATR ratio (current true range vs 14-period ATR)
"""
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import math

from .base_strategy import BaseStrategy, Signal


class MLGbt5mStrategy(BaseStrategy):
    """GBT on 5-bar candle features (returns, RSI, BB z-score, vol ratio, ATR)."""

    name = "ml_gbt_5m"
    version = "2.0"
    description = "GBT/heuristic on candle features: return, RSI, BB z-score, vol ratio, ATR"

    def __init__(self):
        self.min_candles = 30
        self.momentum_bars = 5          # short-term return lookback
        self.rsi_period = 14
        self.bb_period = 20
        self.vol_sma_period = 20
        self.atr_period = 14
        self.buy_threshold = 0.38       # tuned for 1m candle heuristic scoring
        self.sell_threshold = 0.38
        self.model_path = (
            "/home/rob/.openclaw/workspace/blofin-stack/ml_pipeline/models/gbt_5m.pkl"
        )
        self._model = self._load_model()

    def _load_model(self):
        try:
            p = Path(self.model_path)
            if p.exists():
                with open(p, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _calc_rsi(self, closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            gains.append(max(d, 0.0))
            losses.append(max(-d, 0.0))
        ag = sum(gains[:period]) / period
        al = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            return 100.0 if ag > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    def _calc_atr(self, candles: List[dict], period: int = 14) -> float:
        trs = []
        for i in range(1, len(candles)):
            c = candles[i]
            pc = candles[i - 1]['close']
            tr = max(c['high'] - c['low'], abs(c['high'] - pc), abs(c['low'] - pc))
            trs.append(tr)
        if not trs:
            return 0.0
        return sum(trs[-period:]) / min(len(trs), period)

    def _extract_features(self, candles: List[dict]) -> Optional[Dict[str, float]]:
        closes = [c['close'] for c in candles]
        volumes = [c['volume'] for c in candles]
        price = closes[-1]

        # 5-bar return
        if len(closes) < self.momentum_bars + 1:
            return None
        ret = (closes[-1] - closes[-self.momentum_bars - 1]) / closes[-self.momentum_bars - 1]

        # RSI
        rsi = self._calc_rsi(closes, self.rsi_period)

        # BB z-score
        bb_window = closes[-self.bb_period:]
        bb_mean = sum(bb_window) / len(bb_window)
        bb_var = sum((p - bb_mean) ** 2 for p in bb_window) / len(bb_window)
        bb_std = math.sqrt(bb_var) if bb_var > 0 else 0.0
        bb_z = (price - bb_mean) / bb_std if bb_std > 0 else 0.0

        # Volume ratio
        vol_avg = sum(volumes[-self.vol_sma_period:]) / min(len(volumes), self.vol_sma_period)
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

        # ATR ratio: current TR vs average ATR
        atr = self._calc_atr(candles, self.atr_period)
        cur_tr = max(
            candles[-1]['high'] - candles[-1]['low'],
            abs(candles[-1]['high'] - candles[-2]['close']),
            abs(candles[-1]['low'] - candles[-2]['close']),
        ) if len(candles) >= 2 else 0.0
        atr_ratio = cur_tr / atr if atr > 0 else 1.0

        return {
            'ret': ret,
            'rsi': rsi,
            'bb_z': bb_z,
            'vol_ratio': vol_ratio,
            'atr_ratio': atr_ratio,
        }

    # ------------------------------------------------------------------
    # Heuristic scoring (approximates GBT)
    # ------------------------------------------------------------------

    def _heuristic_score(self, f: Dict[str, float]) -> Tuple[float, float]:
        """Returns (buy_score, sell_score) each in [0, 1]."""
        ret = f['ret']
        rsi = f['rsi']
        bb_z = f['bb_z']
        vol_ratio = f['vol_ratio']
        atr_ratio = f['atr_ratio']

        # Momentum (positive → bullish)
        ret_bull = max(0.0, min(1.0, ret / 0.005))    # +0.5% return = full score
        ret_bear = max(0.0, min(1.0, -ret / 0.005))

        # RSI
        rsi_bull = max(0.0, (35.0 - rsi) / 35.0)      # RSI < 35 → bullish
        rsi_bear = max(0.0, (rsi - 65.0) / 35.0)      # RSI > 65 → bearish

        # BB z-score
        bb_bull = max(0.0, min(1.0, -bb_z / 1.5))     # below mean → bullish
        bb_bear = max(0.0, min(1.0, bb_z / 1.5))      # above mean → bearish

        # Volume (confirms both directions)
        vol_conf = min(1.0, vol_ratio / 2.0)

        # ATR (high ATR = breakout, confirms momentum)
        atr_conf = min(1.0, atr_ratio / 2.0)

        # Weighted combination (GBT-like importance)
        buy_score = (
            0.30 * ret_bull +
            0.25 * rsi_bull +
            0.25 * bb_bull +
            0.10 * vol_conf +
            0.10 * atr_conf
        )
        sell_score = (
            0.30 * ret_bear +
            0.25 * rsi_bear +
            0.25 * bb_bear +
            0.10 * vol_conf +
            0.10 * atr_conf
        )

        return min(1.0, buy_score), min(1.0, sell_score)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        features = self._extract_features(context_candles)
        if features is None:
            return None

        if self._model is not None:
            try:
                X = [[features['ret'], features['rsi'], features['bb_z'], features['vol_ratio']]]
                proba = self._model.predict_proba(X)[0]
                if len(proba) == 2:
                    buy_score, sell_score = proba[1], 1.0 - proba[1]
                elif len(proba) == 3:
                    sell_score, _, buy_score = proba
                else:
                    buy_score, sell_score = self._heuristic_score(features)
            except Exception:
                buy_score, sell_score = self._heuristic_score(features)
        else:
            buy_score, sell_score = self._heuristic_score(features)

        if buy_score >= self.buy_threshold and buy_score > sell_score:
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(buy_score, 4),
                'model': 'gbt' if self._model else 'heuristic',
                'ret_pct': round(features['ret'] * 100, 4),
                'rsi': round(features['rsi'], 2),
                'bb_z': round(features['bb_z'], 4),
                'vol_ratio': round(features['vol_ratio'], 4),
                'buy_score': round(buy_score, 4),
            }

        if sell_score >= self.sell_threshold and sell_score > buy_score:
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(sell_score, 4),
                'model': 'gbt' if self._model else 'heuristic',
                'ret_pct': round(features['ret'] * 100, 4),
                'rsi': round(features['rsi'], 2),
                'bb_z': round(features['bb_z'], 4),
                'vol_ratio': round(features['vol_ratio'], 4),
                'sell_score': round(sell_score, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'momentum_bars': self.momentum_bars,
            'model_loaded': self._model is not None,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'buy_threshold' in params:
            self.buy_threshold = float(params['buy_threshold'])
        if 'sell_threshold' in params:
            self.sell_threshold = float(params['sell_threshold'])
        if 'model_path' in params:
            self.model_path = str(params['model_path'])
            self._model = self._load_model()
