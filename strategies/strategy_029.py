#!/usr/bin/env python3
"""
Strategy Name: Bollinger Squeeze Breakout
Type: volatility
Timeframe: 5m
Description: Detects Bollinger Band squeeze (low volatility compression) and trades the breakout direction.
Uses Keltner Channel overlap to confirm squeeze, ATR for stop placement, and volume surge for confirmation.
Adapted for ranging_low_volatility regime where squeezes precede directional moves.
"""

class Strategy:
    def __init__(self):
        self.name = "bollinger_squeeze_breakout"
        self.params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_atr_mult": 1.5,
            "atr_period": 14,
            "volume_surge_mult": 1.4,
            "squeeze_lookback": 6,
            "min_squeeze_bars": 4,
            "rsi_filter_low": 35,
            "rsi_filter_high": 65,
            "sl_atr_mult": 1.8,
            "tp_atr_mult": 3.2,
        }

    def _calc_bb(self, closes, period, std_mult):
        if len(closes) < period:
            return None, None, None
        window = closes[-period:]
        mid = sum(window) / period
        variance = sum((c - mid) ** 2 for c in window) / period
        std = variance ** 0.5
        return mid, mid + std_mult * std, mid - std_mult * std

    def _calc_atr(self, candles, period):
        if len(candles) < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            h = candles[i]["high"]
            l = candles[i]["low"]
            pc = candles[i - 1]["close"]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(trs) / len(trs)

    def _calc_ema(self, values, period):
        if len(values) < period:
            return None
        k = 2 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = v * k + ema * (1 - k)
        return ema

    def _is_squeeze(self, candles, indicators):
        p = self.params
        closes = [c["close"] for c in candles]
        if len(closes) < max(p["bb_period"], p["kc_period"]) + p["squeeze_lookback"]:
            return False, 0

        squeeze_count = 0
        for offset in range(p["squeeze_lookback"]):
            idx = len(closes) - offset
            window_closes = closes[:idx]
            _, bb_upper, bb_lower = self._calc_bb(window_closes, p["bb_period"], p["bb_std"])
            if bb_upper is None:
                return False, 0

            atr = self._calc_atr(candles[:idx], p["kc_period"])
            if atr is None:
                return False, 0
            ema = self._calc_ema(window_closes, p["kc_period"])
            kc_upper = ema + p["kc_atr_mult"] * atr
            kc_lower = ema - p["kc_atr_mult"] * atr

            if bb_upper < kc_upper and bb_lower > kc_lower:
                squeeze_count += 1

        return squeeze_count >= p["min_squeeze_bars"], squeeze_count

    def analyze(self, candles, indicators):
        if len(candles) < 30:
            return "HOLD"

        p = self.params
        closes = [c["close"] for c in candles]

        in_squeeze, squeeze_bars = self._is_squeeze(candles, indicators)
        if not in_squeeze:
            return "HOLD"

        _, bb_upper, bb_lower = self._calc_bb(closes, p["bb_period"], p["bb_std"])
        if bb_upper is None:
            return "HOLD"

        current_close = closes[-1]
        prev_close = closes[-2]

        breakout_up = prev_close <= bb_upper and current_close > bb_upper
        breakout_down = prev_close >= bb_lower and current_close < bb_lower

        if not breakout_up and not breakout_down:
            return "HOLD"

        volumes = [c["volume"] for c in candles]
        avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
        current_vol = volumes[-1]
        if current_vol < avg_vol * p["volume_surge_mult"]:
            return "HOLD"

        rsi = indicators.get("rsi")
        if rsi is not None:
            rsi_val = rsi if isinstance(rsi, (int, float)) else rsi[-1] if hasattr(rsi, "__getitem__") else None
            if rsi_val is not None:
                if breakout_up and rsi_val > p["rsi_filter_high"]:
                    return "HOLD"
                if breakout_down and rsi_val < p["rsi_filter_low"]:
                    return "HOLD"

        if breakout_up:
            return "BUY"
        if breakout_down:
            return "SELL"
        return "HOLD"

    def get_confidence(self, candles, indicators):
        if len(candles) < 30:
            return 0.0

        p = self.params
        closes = [c["close"] for c in candles]
        _, squeeze_bars = self._is_squeeze(candles, indicators)

        squeeze_score = min(squeeze_bars / p["squeeze_lookback"], 1.0) * 0.35

        volumes = [c["volume"] for c in candles]
        avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        volume_score = min((vol_ratio - 1.0) / 1.5, 1.0) * 0.35

        _, bb_upper, bb_lower = self._calc_bb(closes, p["bb_period"], p["bb_std"])
        if bb_upper and bb_lower and (bb_upper - bb_lower) > 0:
            bandwidth = (bb_upper - bb_lower) / ((bb_upper + bb_lower) / 2)
            tightness_score = max(0, 1.0 - bandwidth * 20) * 0.3
        else:
            tightness_score = 0.0

        return round(min(squeeze_score + volume_score + tightness_score, 1.0), 3)