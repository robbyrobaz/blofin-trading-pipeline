#!/usr/bin/env python3
"""
New Strategy Batch Backtester — 5 Novel Strategies
Runs against blofin tick data and reports results.

Strategies:
  1. cross_asset_correlation   — ETH as leading indicator for altcoins
  2. volatility_regime_switch  — Regime-aware (trend/range/chaos) entry rules
  3. ml_random_forest_15m      — Random forest on 15m candles (20 features)
  4. orderflow_imbalance       — Tick-direction buy/sell pressure imbalance
  5. ensemble_top3             — Vote across 3 best existing strategies

Run: python3 new_strategies_backtest.py
"""

import sqlite3
import numpy as np
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH = 'data/blofin_monitor.db'

ALTCOINS = ['SOL-USDT', 'SHIB-USDT', 'JUP-USDT', 'DOGE-USDT', 'XRP-USDT',
            'PEPE-USDT', 'WIF-USDT', 'AVAX-USDT', 'ADA-USDT', 'SUI-USDT']

ALL_SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'DOGE-USDT', 'XRP-USDT',
               'JUP-USDT', 'SHIB-USDT', 'WIF-USDT', 'PEPE-USDT', 'AVAX-USDT',
               'ADA-USDT', 'SUI-USDT', 'ARB-USDT']


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def load_ohlcv(symbol: str, timeframe_minutes: int = 5, days_back: int = 13) -> np.ndarray:
    """Load OHLCV as numpy array: [ts_ms, open, high, low, close, tick_count]"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days_back * 24 * 3600 * 1000)

    c.execute('SELECT ts_ms, price FROM ticks WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC',
              (symbol, start_ts, end_ts))
    ticks = c.fetchall()
    conn.close()

    if len(ticks) < 50:
        return np.array([])

    ticks = np.array(ticks, dtype=float)
    period_ms = timeframe_minutes * 60 * 1000
    first_period = (ticks[0, 0] // period_ms) * period_ms
    period_idx = ((ticks[:, 0] - first_period) // period_ms).astype(int)

    candles = []
    for pid in np.unique(period_idx):
        mask = period_idx == pid
        pts = ticks[mask]
        if len(pts) < 2:
            continue
        candles.append([
            pts[0, 0],          # ts_ms
            pts[0, 1],          # open
            pts[:, 1].max(),    # high
            pts[:, 1].min(),    # low
            pts[-1, 1],         # close
            float(len(pts))     # tick_count (volume proxy)
        ])

    return np.array(candles) if candles else np.array([])


def load_tick_directions(symbol: str, timeframe_minutes: int = 1, days_back: int = 13) -> np.ndarray:
    """
    Load ticks and compute uptick/downtick counts per period.
    Returns array: [ts_ms, open, high, low, close, tick_count, upticks, downticks]
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days_back * 24 * 3600 * 1000)

    c.execute('SELECT ts_ms, price FROM ticks WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC',
              (symbol, start_ts, end_ts))
    ticks = c.fetchall()
    conn.close()

    if len(ticks) < 50:
        return np.array([])

    ticks_arr = np.array(ticks, dtype=float)
    prices = ticks_arr[:, 1]

    # Compute tick directions: +1 = uptick, -1 = downtick, 0 = flat
    directions = np.zeros(len(prices))
    directions[1:] = np.sign(np.diff(prices))

    period_ms = timeframe_minutes * 60 * 1000
    first_period = (ticks_arr[0, 0] // period_ms) * period_ms
    period_idx = ((ticks_arr[:, 0] - first_period) // period_ms).astype(int)

    candles = []
    for pid in np.unique(period_idx):
        mask = period_idx == pid
        pts = ticks_arr[mask]
        dirs = directions[mask]
        if len(pts) < 2:
            continue
        upticks = float(np.sum(dirs > 0))
        downticks = float(np.sum(dirs < 0))
        candles.append([
            pts[0, 0],          # ts_ms
            pts[0, 1],          # open
            pts[:, 1].max(),    # high
            pts[:, 1].min(),    # low
            pts[-1, 1],         # close
            float(len(pts)),    # tick_count
            upticks,
            downticks,
        ])

    return np.array(candles) if candles else np.array([])


# ─────────────────────────────────────────────
# Indicator Library (self-contained)
# ─────────────────────────────────────────────
def ema(prices: np.ndarray, period: int) -> np.ndarray:
    result = np.zeros(len(prices))
    k = 2.0 / (period + 1)
    result[0] = prices[0]
    for i in range(1, len(prices)):
        result[i] = prices[i] * k + result[i - 1] * (1 - k)
    return result


def sma(prices: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1:i + 1])
    return result


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    if len(prices) < period + 1:
        return result
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(prices)):
        g = max(0.0, deltas[i - 1])
        l = max(0.0, -deltas[i - 1])
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        result[i] = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100
    return result


def bollinger_bands(prices: np.ndarray, period: int = 20, std_mult: float = 2.0):
    mid = sma(prices, period)
    std = np.array([np.std(prices[max(0, i - period + 1):i + 1]) if i >= period - 1 else np.nan
                    for i in range(len(prices))])
    return mid + std_mult * std, mid, mid - std_mult * std


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    result = np.full(len(high), np.nan)
    if len(high) < period + 1:
        return result
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    result[period - 1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index."""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Smooth
    tr_smooth = np.full(n, np.nan)
    pdm_smooth = np.full(n, np.nan)
    mdm_smooth = np.full(n, np.nan)

    tr_smooth[period] = np.sum(tr[1:period + 1])
    pdm_smooth[period] = np.sum(plus_dm[1:period + 1])
    mdm_smooth[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
        pdm_smooth[i] = pdm_smooth[i - 1] - pdm_smooth[i - 1] / period + plus_dm[i]
        mdm_smooth[i] = mdm_smooth[i - 1] - mdm_smooth[i - 1] / period + minus_dm[i]

    pdi = np.where(tr_smooth > 0, 100 * pdm_smooth / tr_smooth, 0)
    mdi = np.where(tr_smooth > 0, 100 * mdm_smooth / tr_smooth, 0)
    dx = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0)

    adx_out = np.full(n, np.nan)
    valid = ~np.isnan(dx)
    first_valid = period
    adx_out[first_valid + period - 1] = np.nanmean(dx[first_valid:first_valid + period])
    for i in range(first_valid + period, n):
        if not np.isnan(adx_out[i - 1]):
            adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period

    return adx_out


def vwap(prices: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        pv = prices[i - period + 1:i + 1] * volume[i - period + 1:i + 1]
        v = volume[i - period + 1:i + 1]
        tv = v.sum()
        if tv > 0:
            result[i] = pv.sum() / tv
    return result


# ─────────────────────────────────────────────
# Backtest Engine (same as strategy_library_backtest.py)
# ─────────────────────────────────────────────
def run_backtest(signals: np.ndarray, close: np.ndarray,
                 hold_bars: int = 6, fee_pct: float = 0.05) -> Dict:
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_dir = 0
    bars_held = 0

    for i in range(len(signals)):
        if in_trade:
            bars_held += 1
            if bars_held >= hold_bars:
                exit_price = close[i]
                if entry_dir == 1:
                    pnl = ((exit_price - entry_price) / entry_price * 100) - fee_pct * 2
                else:
                    pnl = ((entry_price - exit_price) / entry_price * 100) - fee_pct * 2
                trades.append({'pnl': pnl, 'dir': entry_dir})
                in_trade = False
                bars_held = 0

        if not in_trade and signals[i] != 0:
            in_trade = True
            entry_price = close[i]
            entry_dir = int(signals[i])
            bars_held = 0

    if not trades:
        return {'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0}

    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls)
    avg_pnl = np.mean(pnls)
    total_pnl = np.sum(pnls)

    returns = np.array(pnls) / 100
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24 * 12) / 50 \
        if len(returns) > 1 and np.std(returns) > 0 else 0

    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    max_dd = float(np.max(running_max - equity)) if len(equity) > 0 else 0

    return {'win_rate': win_rate, 'avg_pnl': avg_pnl, 'total_pnl': total_pnl,
            'sharpe': sharpe, 'max_dd': max_dd, 'n_trades': len(trades)}


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 1: CROSS-ASSET CORRELATION
# ETH as leading indicator — altcoin signals 2-4 bars later
# ═══════════════════════════════════════════════════════════════════════
_eth_cache: Optional[np.ndarray] = None

def _get_eth_candles(tf: int = 5) -> Optional[np.ndarray]:
    global _eth_cache
    if _eth_cache is None:
        _eth_cache = load_ohlcv('ETH-USDT', tf)
    return _eth_cache


def strategy_cross_asset_correlation(candles: np.ndarray, eth_candles: np.ndarray,
                                      eth_roc_period: int = 3,
                                      eth_threshold_pct: float = 0.8,
                                      lag_bars: int = 2) -> np.ndarray:
    """
    ETH-lead signal: when ETH ROC > threshold, expect altcoin to follow in lag_bars.
    Uses aligned timestamps between ETH and target symbol.
    """
    if len(candles) < 20 or len(eth_candles) < 20:
        return np.zeros(len(candles) if len(candles) > 0 else 1)

    # Align by timestamp — match candle ts_ms
    alt_ts = candles[:, 0]
    eth_ts = eth_candles[:, 0]
    eth_close = eth_candles[:, 4]

    # Build ETH ROC indexed by timestamp
    eth_roc_map: Dict[float, float] = {}
    for i in range(eth_roc_period, len(eth_close)):
        if eth_close[i - eth_roc_period] > 0:
            roc = (eth_close[i] - eth_close[i - eth_roc_period]) / eth_close[i - eth_roc_period] * 100
            eth_roc_map[eth_ts[i]] = roc

    signals = np.zeros(len(candles))

    for i in range(lag_bars, len(candles)):
        # Look up ETH ROC at (current_ts - lag_bars * period)
        # Match nearest ETH candle to the lagged timestamp
        target_ts = alt_ts[i - lag_bars]
        # Find closest ETH timestamp
        ts_diffs = np.abs(eth_ts - target_ts)
        eth_idx = int(np.argmin(ts_diffs))
        if ts_diffs[eth_idx] > 15 * 60 * 1000:  # More than 15m off → skip
            continue

        eth_roc_val = eth_roc_map.get(eth_ts[eth_idx], None)
        if eth_roc_val is None:
            continue

        # Also check altcoin hasn't already moved too much (avoid chasing)
        alt_roc = 0.0
        if candles[i - 1, 4] > 0:
            alt_roc = (candles[i, 4] - candles[i - 1, 4]) / candles[i - 1, 4] * 100

        if eth_roc_val >= eth_threshold_pct and abs(alt_roc) < 0.5:
            signals[i] = 1  # ETH pumped → alt should follow → BUY
        elif eth_roc_val <= -eth_threshold_pct and abs(alt_roc) < 0.5:
            signals[i] = -1  # ETH dumped → alt should follow → SELL

    return signals


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 2: VOLATILITY REGIME SWITCH
# Regime: trending / ranging / chaotic — different logic per regime
# ═══════════════════════════════════════════════════════════════════════
def strategy_volatility_regime_switch(candles: np.ndarray,
                                       atr_period: int = 14,
                                       atr_percentile_window: int = 50,
                                       adx_period: int = 14,
                                       adx_trend_thresh: float = 25.0,
                                       adx_chaos_thresh: float = 45.0) -> np.ndarray:
    """
    Regime detection:
      ATR percentile + ADX → classify each bar as TRENDING / RANGING / CHAOTIC
    Entry rules:
      TRENDING: triple-EMA alignment (5/13/21)
      RANGING:  BB mean reversion (price touches band → fade)
      CHAOTIC:  no trade
    """
    if len(candles) < max(atr_period, adx_period, atr_percentile_window) + 10:
        return np.zeros(len(candles))

    high  = candles[:, 2]
    low   = candles[:, 3]
    close = candles[:, 4]

    atr_vals = atr(high, low, close, atr_period)
    adx_vals = adx(high, low, close, adx_period)

    ema5  = ema(close, 5)
    ema13 = ema(close, 13)
    ema21 = ema(close, 21)

    bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)

    signals = np.zeros(len(close))
    min_idx = max(atr_period + atr_percentile_window, adx_period * 2 + 5, 25)

    for i in range(min_idx, len(close)):
        if np.isnan(atr_vals[i]) or np.isnan(adx_vals[i]):
            continue

        # ATR percentile rank over last atr_percentile_window bars
        recent_atrs = atr_vals[i - atr_percentile_window:i]
        valid_atrs = recent_atrs[~np.isnan(recent_atrs)]
        if len(valid_atrs) < 10:
            continue
        atr_pct = float(np.sum(valid_atrs <= atr_vals[i])) / len(valid_atrs)

        adx_val = adx_vals[i]

        # ── Regime Classification ──
        if adx_val > adx_chaos_thresh:
            regime = 'CHAOTIC'
        elif adx_val > adx_trend_thresh and atr_pct > 0.60:
            regime = 'TRENDING'
        else:
            regime = 'RANGING'

        if regime == 'CHAOTIC':
            continue

        elif regime == 'TRENDING':
            # EMA alignment: all three stacked → enter
            if ema5[i] > ema13[i] > ema21[i] and ema5[i - 1] <= ema13[i - 1]:
                signals[i] = 1  # Fresh bullish alignment
            elif ema5[i] < ema13[i] < ema21[i] and ema5[i - 1] >= ema13[i - 1]:
                signals[i] = -1

        elif regime == 'RANGING':
            if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
                continue
            # BB mean reversion
            if close[i] <= bb_lower[i] and close[i - 1] > bb_lower[i - 1]:
                signals[i] = 1  # Pierced lower band
            elif close[i] >= bb_upper[i] and close[i - 1] < bb_upper[i - 1]:
                signals[i] = -1

    return signals


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 3: ML RANDOM FOREST — 15m CANDLES
# Trains on 70% of data, tests on 30%.  Generates BUY/SELL signals.
# ═══════════════════════════════════════════════════════════════════════
def _build_feature_matrix(candles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 20-feature matrix for random forest.
    Features: RSI(7/14/21), EMA ratios, BB position, ATR normalized,
              ROC(3/5/10), volume z-score, OHLC ratios, candle body/wick.
    Returns (X, y) where y = 1 if next close > current close else 0.
    """
    n = len(candles)
    if n < 50:
        return np.array([]), np.array([])

    open_  = candles[:, 1]
    high   = candles[:, 2]
    low    = candles[:, 3]
    close  = candles[:, 4]
    volume = candles[:, 5]

    rsi7  = rsi(close, 7)
    rsi14 = rsi(close, 14)
    rsi21 = rsi(close, 21)

    e5  = ema(close, 5)
    e10 = ema(close, 10)
    e20 = ema(close, 20)
    e50 = ema(close, 50)

    bb_up, bb_mid, bb_low = bollinger_bands(close, 20, 2.0)
    atr14 = atr(high, low, close, 14)

    vol_sma20 = sma(volume, 20)

    features = []
    targets  = []

    start = 55  # Enough warmup for all indicators

    for i in range(start, n - 1):
        if (np.isnan(rsi7[i]) or np.isnan(rsi14[i]) or np.isnan(rsi21[i]) or
                np.isnan(bb_up[i]) or np.isnan(atr14[i]) or
                np.isnan(vol_sma20[i]) or vol_sma20[i] == 0 or
                np.isnan(e50[i]) or e50[i] == 0 or close[i] == 0):
            continue

        # 20 features
        bb_range = bb_up[i] - bb_low[i]
        bb_pos = (close[i] - bb_low[i]) / bb_range if bb_range > 0 else 0.5

        roc3  = (close[i] - close[i - 3])  / close[i - 3]  * 100 if close[i - 3]  > 0 else 0
        roc5  = (close[i] - close[i - 5])  / close[i - 5]  * 100 if close[i - 5]  > 0 else 0
        roc10 = (close[i] - close[i - 10]) / close[i - 10] * 100 if close[i - 10] > 0 else 0

        body   = (close[i] - open_[i]) / close[i] * 100
        wick_u = (high[i] - max(open_[i], close[i])) / close[i] * 100
        wick_l = (min(open_[i], close[i]) - low[i]) / close[i] * 100

        vol_z = (volume[i] - vol_sma20[i]) / (np.std(volume[max(0, i - 20):i]) + 1e-8)

        feat = [
            rsi7[i] / 100,          # 1
            rsi14[i] / 100,         # 2
            rsi21[i] / 100,         # 3
            (e5[i] - e50[i]) / e50[i],   # 4: short/long EMA spread
            (e10[i] - e50[i]) / e50[i],  # 5
            (e20[i] - e50[i]) / e50[i],  # 6
            (close[i] - e5[i]) / e5[i],  # 7: price vs short EMA
            (close[i] - e20[i]) / e20[i],# 8
            bb_pos,                  # 9
            atr14[i] / close[i],     # 10: normalized ATR
            roc3 / 10,               # 11
            roc5 / 10,               # 12
            roc10 / 10,              # 13
            body / 5,                # 14
            wick_u / 3,              # 15
            wick_l / 3,              # 16
            min(vol_z / 3, 5),       # 17: capped vol z-score
            (high[i] - low[i]) / close[i] * 100,  # 18: candle range %
            (close[i] - close[i - 1]) / close[i - 1] * 100 if close[i - 1] > 0 else 0,  # 19: 1-bar return
            (close[i - 1] - close[i - 2]) / close[i - 2] * 100 if close[i - 2] > 0 else 0,  # 20: lag-2 return
        ]

        features.append(feat)
        # Target: 1 = next close goes up, 0 = goes down
        targets.append(1 if close[i + 1] > close[i] else 0)

    if not features:
        return np.array([]), np.array([])

    return np.array(features), np.array(targets)


def strategy_ml_random_forest_15m(candles: np.ndarray) -> np.ndarray:
    """
    Train Random Forest on first 70% of 15m candles.
    Generate signals on last 30% (test period).
    Returns signal array for full candle length (zeros in train period).
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("    [RF] sklearn not available — skipping")
        return np.zeros(len(candles))

    if len(candles) < 150:
        return np.zeros(len(candles))

    X, y = _build_feature_matrix(candles)
    if len(X) < 100:
        return np.zeros(len(candles))

    # Temporal split: 70% train / 30% test (no shuffling — strict time ordering)
    split = int(len(X) * 0.70)
    X_train, y_train = X[:split], y[:split]
    X_test           = X[split:]

    if len(X_train) < 50 or len(X_test) < 10:
        return np.zeros(len(candles))

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    probs = rf.predict_proba(X_test)[:, 1]  # P(up)

    # Only signal when confident
    signals = np.zeros(len(candles))
    # Offset: features start at warmup=55, test starts at split+55
    feature_start = 55
    test_offset = feature_start + split

    for j, prob in enumerate(probs):
        candle_idx = test_offset + j
        if candle_idx >= len(candles) - 1:
            break
        if prob >= 0.60:
            signals[candle_idx] = 1   # High confidence UP
        elif prob <= 0.40:
            signals[candle_idx] = -1  # High confidence DOWN

    return signals


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 4: ORDERFLOW IMBALANCE (tick-direction proxy)
# Uptick/downtick pressure ratio from raw ticks
# ═══════════════════════════════════════════════════════════════════════
def strategy_orderflow_imbalance(tick_candles: np.ndarray,
                                  imbalance_threshold: float = 0.65,
                                  lookback_periods: int = 5,
                                  confirm_bars: int = 2) -> np.ndarray:
    """
    Uses tick direction (uptick vs downtick per candle) as a proxy for
    buy/sell orderflow pressure.

    tick_candles: [ts_ms, open, high, low, close, tick_count, upticks, downticks]

    Signal logic:
    - Rolling sum of uptick ratios over lookback_periods
    - Imbalance > threshold → BUY; < (1-threshold) → SELL
    - Require confirm_bars of consecutive imbalance before entry
    """
    if len(tick_candles) < lookback_periods + confirm_bars + 5:
        return np.zeros(len(tick_candles) if len(tick_candles) > 0 else 1)

    total_ticks = tick_candles[:, 5]
    upticks     = tick_candles[:, 6]
    downticks   = tick_candles[:, 7]

    # Per-bar uptick ratio
    total_dir = upticks + downticks
    uptick_ratio = np.where(total_dir > 0, upticks / total_dir, 0.5)

    signals = np.zeros(len(tick_candles))

    for i in range(lookback_periods + confirm_bars, len(tick_candles)):
        # Rolling uptick pressure
        window = uptick_ratio[i - lookback_periods:i]
        avg_ratio = np.mean(window)

        # Confirm: check last confirm_bars are consistent
        confirm_window = uptick_ratio[i - confirm_bars:i]

        if avg_ratio >= imbalance_threshold and np.all(confirm_window >= 0.55):
            signals[i] = 1   # Strong buy pressure
        elif avg_ratio <= (1 - imbalance_threshold) and np.all(confirm_window <= 0.45):
            signals[i] = -1  # Strong sell pressure

    return signals


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 5: ENSEMBLE TOP-3
# Vote: vwap_deviation + zscore_mean_reversion + bb_mean_reversion
# Enter only when 2/3 agree
# ═══════════════════════════════════════════════════════════════════════
def _sub_vwap_deviation(close, volume, period=20, dev_pct=1.0):
    result = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        pv = close[i - period + 1:i + 1] * volume[i - period + 1:i + 1]
        v = volume[i - period + 1:i + 1]
        tv = v.sum()
        if tv > 0:
            result[i] = pv.sum() / tv
    sigs = np.zeros(len(close))
    for i in range(period, len(close)):
        if np.isnan(result[i]) or result[i] <= 0:
            continue
        dev = (close[i] - result[i]) / result[i] * 100
        if dev <= -dev_pct:
            sigs[i] = 1
        elif dev >= dev_pct:
            sigs[i] = -1
    return sigs


def _sub_zscore_mean_reversion(close, period=20, z_thresh=2.0):
    sigs = np.zeros(len(close))
    for i in range(period, len(close)):
        window = close[i - period:i]
        mean = np.mean(window)
        std  = np.std(window)
        if std <= 0:
            continue
        z = (close[i] - mean) / std
        if z <= -z_thresh:
            sigs[i] = 1
        elif z >= z_thresh:
            sigs[i] = -1
    return sigs


def _sub_bb_mean_reversion(close, period=20, std_mult=2.0):
    upper, mid, lower = bollinger_bands(close, period, std_mult)
    sigs = np.zeros(len(close))
    for i in range(period, len(close)):
        if np.isnan(lower[i]):
            continue
        if close[i] <= lower[i] and close[i - 1] > lower[i - 1]:
            sigs[i] = 1
        elif close[i] >= upper[i] and close[i - 1] < upper[i - 1]:
            sigs[i] = -1
    return sigs


def strategy_ensemble_top3(candles: np.ndarray) -> np.ndarray:
    """
    Ensemble vote: signal only when ≥ 2 of 3 sub-strategies agree on direction.
    Sub-strategies: vwap_deviation, zscore_mean_reversion, bb_mean_reversion
    (these are top performers from strategy_library_summary.json)
    """
    if len(candles) < 40:
        return np.zeros(len(candles))

    close  = candles[:, 4]
    volume = candles[:, 5]

    s1 = _sub_vwap_deviation(close, volume, period=20, dev_pct=1.0)
    s2 = _sub_zscore_mean_reversion(close, period=20, z_thresh=2.0)
    s3 = _sub_bb_mean_reversion(close, period=20, std_mult=2.0)

    vote = s1 + s2 + s3  # Range: -3 to +3
    signals = np.zeros(len(candles))
    signals[vote >= 2]  =  1   # At least 2 agree → BUY
    signals[vote <= -2] = -1   # At least 2 agree → SELL

    return signals


# ─────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────
def run_strategy_on_symbols(strat_name: str, strat_fn,
                             symbols: List[str], timeframe: int = 5,
                             hold_bars: int = 6) -> List[Dict]:
    results = []
    eth_candles = load_ohlcv('ETH-USDT', timeframe)

    for symbol in symbols:
        try:
            if strat_name == 'orderflow_imbalance':
                candles = load_tick_directions(symbol, timeframe_minutes=1)
                # Re-aggregate to 5m for backtest close prices
                ohlcv5 = load_ohlcv(symbol, timeframe)
                if len(candles) < 50 or len(ohlcv5) < 50:
                    continue
                signals = strat_fn(candles)
                # Align signals to 5m candles (signals are on 1m)
                # Use every 5th signal as approximation for the hold_bars backtest
                # For simplicity, run backtest on 1m data
                metrics = run_backtest(signals, candles[:, 4], hold_bars=5, fee_pct=0.05)
            elif strat_name == 'cross_asset_correlation':
                if symbol == 'ETH-USDT':
                    continue  # Skip ETH itself
                candles = load_ohlcv(symbol, timeframe)
                if len(candles) < 50 or len(eth_candles) < 50:
                    continue
                signals = strat_fn(candles, eth_candles)
                metrics = run_backtest(signals, candles[:, 4], hold_bars=hold_bars, fee_pct=0.05)
            elif strat_name == 'ml_random_forest_15m':
                candles = load_ohlcv(symbol, timeframe_minutes=15)  # 15m candles
                if len(candles) < 150:
                    continue
                signals = strat_fn(candles)
                metrics = run_backtest(signals, candles[:, 4], hold_bars=4, fee_pct=0.05)
            else:
                candles = load_ohlcv(symbol, timeframe)
                if len(candles) < 50:
                    continue
                signals = strat_fn(candles)
                metrics = run_backtest(signals, candles[:, 4], hold_bars=hold_bars, fee_pct=0.05)

            if metrics['n_trades'] >= 3:
                results.append({'symbol': symbol, **metrics})

        except Exception as e:
            print(f"    [{symbol}] Error in {strat_name}: {e}")

    return results


def aggregate_results(results: List[Dict]) -> Dict:
    if not results:
        return {}
    trades = [r['n_trades'] for r in results]
    wrs    = [r['win_rate'] for r in results]
    sharps = [r['sharpe'] for r in results]
    pnls   = [r['total_pnl'] for r in results]
    return {
        'total_trades':  sum(trades),
        'avg_trades':    np.mean(trades),
        'avg_win_rate':  np.mean(wrs),
        'avg_sharpe':    np.mean(sharps),
        'total_pnl':     np.sum(pnls),
        'avg_pnl':       np.mean(pnls),
        'n_symbols':     len(results),
        'per_symbol':    results,
    }


def check_system():
    import subprocess
    try:
        cpu_out = subprocess.run(['sensors'], capture_output=True, text=True).stdout
        for line in cpu_out.splitlines():
            if 'Package id 0' in line:
                temp_str = line.split('+')[1].split('°')[0].strip()
                cpu_temp = float(temp_str)
                return cpu_temp
    except Exception:
        pass
    return 0.0


def fmt_row(name, agg, cpu_temp, status):
    n      = agg.get('total_trades', 0)
    wr     = agg.get('avg_win_rate', 0) * 100
    sharpe = agg.get('avg_sharpe', 0)
    pnl    = agg.get('total_pnl', 0)
    return f"{name:30s}  {n:6d}  {wr:5.1f}%  {sharpe:6.3f}  {pnl:+8.2f}%  {cpu_temp:4.0f}°C  {status}"


if __name__ == '__main__':
    print("=" * 80)
    print("NEW STRATEGY BATCH BACKTESTER — 5 Novel Strategies")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    STRATEGIES_TO_RUN = [
        ('cross_asset_correlation',  None),    # special multi-symbol ETH-lead
        ('volatility_regime_switch', strategy_volatility_regime_switch),
        ('ml_random_forest_15m',     strategy_ml_random_forest_15m),
        ('orderflow_imbalance',      strategy_orderflow_imbalance),
        ('ensemble_top3',            strategy_ensemble_top3),
    ]

    all_agg = {}
    report_rows = []

    for strat_name, strat_fn in STRATEGIES_TO_RUN:
        print(f"\n── {strat_name.upper()} ──")

        # Choose symbols
        if strat_name == 'cross_asset_correlation':
            symbols = ALTCOINS
            fn = lambda c, eth: strategy_cross_asset_correlation(c, eth)
        elif strat_name == 'ml_random_forest_15m':
            symbols = ALL_SYMBOLS
            fn = strategy_ml_random_forest_15m
        elif strat_name == 'orderflow_imbalance':
            symbols = ALL_SYMBOLS
            fn = strategy_orderflow_imbalance
        else:
            symbols = ALL_SYMBOLS
            fn = strat_fn

        results = run_strategy_on_symbols(strat_name, fn, symbols)
        agg = aggregate_results(results)
        all_agg[strat_name] = agg

        cpu = check_system()
        import subprocess
        mem_out = subprocess.run(['free', '-h'], capture_output=True, text=True).stdout.splitlines()

        status = "✅"
        if cpu > 80:
            status = "🔥 HOT"
        elif not results:
            status = "❌ NO DATA"
        elif agg.get('total_trades', 0) < 5:
            status = "⚠️ FEW TRADES"

        row = fmt_row(strat_name, agg, cpu, status)
        report_rows.append(row)

        print(f"  Symbols traded: {agg.get('n_symbols', 0)}/{len(symbols)}")
        print(f"  Total trades:   {agg.get('total_trades', 0)}")
        print(f"  Avg Win Rate:   {agg.get('avg_win_rate', 0):.1%}")
        print(f"  Avg Sharpe:     {agg.get('avg_sharpe', 0):.3f}")
        print(f"  Total PnL:      {agg.get('total_pnl', 0):+.2f}%")
        print(f"  CPU:            {cpu:.0f}°C")
        print(f"  Memory:         {mem_out[1] if len(mem_out) > 1 else 'N/A'}")

        if cpu > 80:
            print("\n⚠️  CPU > 80°C — STOPPING for thermal safety!")
            break

    # ── Final Summary Table ──
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':30s}  {'Trades':6s}  {'WR%':5s}  {'Sharpe':6s}  {'PnL%':8s}  {'CPU°C':5s}  Status")
    print("-" * 80)
    for row in report_rows:
        print(row)

    # Save results
    out = {
        'timestamp': datetime.now().isoformat(),
        'strategies': {k: {**v, 'per_symbol': v.get('per_symbol', [])}
                       for k, v in all_agg.items()},
    }
    with open('data/new_strategies_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nResults saved to data/new_strategies_results.json")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
