#!/usr/bin/env python3
"""
Strategy Name: Bollinger Squeeze Breakout
Type: volatility
Timeframe: 15m
Description: Detects Bollinger Band squeezes (low volatility compression) and trades the breakout direction using Keltner Channel confirmation and volume surge validation. Optimized for ranging_low_volatility regimes where squeezes precede directional moves.
"""
from typing import Optional


class Strategy:
    def __init__(self):
        self.name = "bollinger_squeeze_breakout"
        self.version = "2.1"
        self.description = "BB squeeze breakout with Keltner Channel + volume surge"
        self.params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_atr_mult": 1.5,
            "atr_period": 14,
            # Note: volume = tick-count (nearly constant). Set to 0.5 to disable volume gate.
            # Real volume surges require actual trading volume data.
            "volume_surge_mult": 0.5,
            "lookback_squeeze": 6,
            "min_squeeze_bars": 4,
            "rsi_filter_low": 30,
            "rsi_filter_high": 70,
            "stop_atr_mult": 2.0,
            "take_profit_atr_mult": 3.0,
        }

    def _calc_keltner(self, candles, period, atr_mult):
        if len(candles) < period:
            return None, None, None
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        ema = closes[-period]
        k = 2.0 / (period + 1)
        for c in closes[-period + 1:]:
            ema = c * k + ema * (1 - k)
        trs = []
        for i in range(-period, 0):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        atr = sum(trs) / len(trs)
        return ema, ema + atr_mult * atr, ema - atr_mult * atr

    def _is_squeeze(self, candles, indicators):
        bb_upper = indicators.get("bbands_upper") or indicators.get("bb_upper")
        bb_lower = indicators.get("bbands_lower") or indicators.get("bb_lower")
        if bb_upper is None or bb_lower is None:
            return False, 0
        kc_mid, kc_upper, kc_lower = self._calc_keltner(
            candles, self.params["kc_period"], self.params["kc_atr_mult"]
        )
        if kc_upper is None:
            return False, 0
        squeeze_on = bb_upper < kc_upper and bb_lower > kc_lower
        return squeeze_on, kc_upper, kc_lower

    def analyze(self, candles, indicators):
        if len(candles) < max(self.params["bb_period"], self.params["kc_period"]) + self.params["lookback_squeeze"]:
            return "HOLD"

        bb_upper = indicators.get("bbands_upper") or indicators.get("bb_upper")
        bb_lower = indicators.get("bbands_lower") or indicators.get("bb_lower")
        bb_mid = indicators.get("bbands_middle") or indicators.get("bb_mid")
        rsi = indicators.get("rsi")
        atr = indicators.get("atr")

        if bb_upper is None or bb_lower is None or rsi is None:
            return "HOLD"

        squeeze_result = self._is_squeeze(candles, indicators)
        current_squeeze = squeeze_result[0]

        # We want squeeze to have JUST released (was on, now off)
        # Check recent bars for squeeze history
        bb_width_current = (bb_upper - bb_lower) / bb_mid if bb_mid and bb_mid > 0 else 0
        lookback = self.params["lookback_squeeze"]
        recent_closes = [c["close"] for c in candles[-lookback:]]
        recent_volumes = [c["volume"] for c in candles[-lookback:]]

        if len(recent_volumes) < 2:
            return "HOLD"

        avg_volume = sum(recent_volumes[:-1]) / max(len(recent_volumes) - 1, 1)
        current_volume = recent_volumes[-1]
        volume_surge = current_volume > avg_volume * self.params["volume_surge_mult"]

        close = candles[-1]["close"]
        prev_close = candles[-2]["close"]

        # Detect breakout from squeeze
        kc_mid, kc_upper, kc_lower = self._calc_keltner(
            candles, self.params["kc_period"], self.params["kc_atr_mult"]
        )
        if kc_upper is None:
            return "HOLD"

        # Squeeze release: price breaks outside Bollinger Bands with volume
        bullish_breakout = close > bb_upper and prev_close <= bb_upper and volume_surge
        bearish_breakout = close < bb_lower and prev_close >= bb_lower and volume_surge

        # RSI filter: avoid overbought buys and oversold sells
        if bullish_breakout and rsi < self.params["rsi_filter_high"]:
            return "BUY"
        elif bearish_breakout and rsi > self.params["rsi_filter_low"]:
            return "SELL"

        return "HOLD"

    def _calc_bb(self, closes, period, std_dev):
        """Returns (upper, mid, lower) Bollinger Bands for current candle."""
        if len(closes) < period:
            return None, None, None
        recent = closes[-period:]
        mid = sum(recent) / period
        var = sum((x - mid) ** 2 for x in recent) / period
        std = var ** 0.5
        return mid + std_dev * std, mid, mid - std_dev * std

    def _calc_rsi(self, closes, period=14):
        """Returns RSI value."""
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

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        """Standard detect() interface for backtester compatibility."""
        min_len = max(self.params["bb_period"], self.params["kc_period"]) + self.params["lookback_squeeze"]
        if len(context_candles) < min_len:
            return None

        closes = [c["close"] for c in context_candles]
        bb_upper, bb_mid, bb_lower = self._calc_bb(closes, self.params["bb_period"], self.params["bb_std"])
        if bb_upper is None or bb_mid is None or bb_lower is None:
            return None

        kc_mid, kc_upper, kc_lower = self._calc_keltner(
            context_candles, self.params["kc_period"], self.params["kc_atr_mult"]
        )
        if kc_upper is None:
            return None

        rsi = self._calc_rsi(closes, self.params["atr_period"])

        lookback = self.params["lookback_squeeze"]
        recent_volumes = [c["volume"] for c in context_candles[-lookback:]]
        if len(recent_volumes) < 2:
            return None
        avg_volume = sum(recent_volumes[:-1]) / max(len(recent_volumes) - 1, 1)
        current_volume = recent_volumes[-1]
        volume_surge = avg_volume > 0 and current_volume > avg_volume * self.params["volume_surge_mult"]

        close = context_candles[-1]["close"]
        prev_close = context_candles[-2]["close"] if len(context_candles) >= 2 else close

        # Squeeze: BB inside Keltner Channels
        squeeze_on = bb_upper < kc_upper and bb_lower > kc_lower

        # Breakout after squeeze: price exits BB with volume
        bullish_breakout = (close > bb_upper and prev_close <= bb_upper and
                            volume_surge and squeeze_on and
                            rsi < self.params["rsi_filter_high"])
        bearish_breakout = (close < bb_lower and prev_close >= bb_lower and
                            volume_surge and squeeze_on and
                            rsi > self.params["rsi_filter_low"])

        if not bullish_breakout and not bearish_breakout:
            # Also check: recently released from squeeze (previous bar was in squeeze)
            if len(context_candles) >= min_len + 1:
                prev_closes = closes[:-1]
                pb_upper, pb_mid, pb_lower = self._calc_bb(
                    prev_closes, self.params["bb_period"], self.params["bb_std"]
                )
                prev_kc = self._calc_keltner(
                    context_candles[:-1], self.params["kc_period"], self.params["kc_atr_mult"]
                )
                if pb_upper and prev_kc[1]:
                    prev_squeeze = pb_upper < prev_kc[1] and pb_lower > prev_kc[2]
                    if prev_squeeze and not squeeze_on and volume_surge:
                        bullish_breakout = close > bb_upper and rsi < self.params["rsi_filter_high"]
                        bearish_breakout = close < bb_lower and rsi > self.params["rsi_filter_low"]

        if bullish_breakout:
            confidence = self.get_confidence(context_candles, {
                "bb_upper": bb_upper, "bb_lower": bb_lower,
                "bbands_middle": bb_mid, "rsi": rsi
            })
            return {
                "signal": "BUY",
                "strategy": self.name,
                "confidence": round(max(0.50, confidence), 4),
                "bb_squeeze": squeeze_on,
                "rsi": round(rsi, 2),
            }

        if bearish_breakout:
            confidence = self.get_confidence(context_candles, {
                "bb_upper": bb_upper, "bb_lower": bb_lower,
                "bbands_middle": bb_mid, "rsi": rsi
            })
            return {
                "signal": "SELL",
                "strategy": self.name,
                "confidence": round(max(0.50, confidence), 4),
                "bb_squeeze": squeeze_on,
                "rsi": round(rsi, 2),
            }

        return None

    def get_confidence(self, candles, indicators):
        if len(candles) < self.params["bb_period"] + self.params["lookback_squeeze"]:
            return 0.0

        bb_upper = indicators.get("bbands_upper") or indicators.get("bb_upper")
        bb_lower = indicators.get("bbands_lower") or indicators.get("bb_lower")
        bb_mid = indicators.get("bbands_middle") or indicators.get("bb_mid")
        rsi = indicators.get("rsi")

        if not all([bb_upper, bb_lower, bb_mid]):
            return 0.0

        confidence = 0.5

        # Tighter squeeze = higher confidence breakout
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 1.0
        if bb_width < 0.02:
            confidence += 0.2
        elif bb_width < 0.04:
            confidence += 0.1

        # Volume confirmation
        recent_volumes = [c["volume"] for c in candles[-6:]]
        if len(recent_volumes) >= 2:
            avg_vol = sum(recent_volumes[:-1]) / max(len(recent_volumes) - 1, 1)
            if recent_volumes[-1] > avg_vol * 2.0:
                confidence += 0.15
            elif recent_volumes[-1] > avg_vol * 1.4:
                confidence += 0.08

        # RSI near midpoint = more room to run
        if rsi and 40 < rsi < 60:
            confidence += 0.1

        return min(confidence, 1.0)