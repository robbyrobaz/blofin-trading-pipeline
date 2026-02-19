#!/usr/bin/env python3
"""
Aggressive Multi-Timeframe Strategy Optimizer
Runs 10 iterations of backtesting + parameter tuning to find winners.
Goal: Find strategies with >52% win rate and positive Sharpe ratio.
"""

import sqlite3
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from itertools import product
import warnings
warnings.filterwarnings('ignore')

DB_PATH = 'data/blofin_monitor.db'
INITIAL_CAPITAL = 10000.0

# Focus on high-volume, liquid symbols
SYMBOLS = [
    'BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'DOGE-USDT',
    'NOT-USDT', 'JUP-USDT', 'PEPE-USDT', 'SHIB-USDT',
    'ARB-USDT', 'OP-USDT', 'LINK-USDT', 'AVAX-USDT'
]

# ============================================================
# DATA LOADING & AGGREGATION
# ============================================================

def load_ticks_for_symbol(symbol: str, days_back: int = 12) -> np.ndarray:
    """Load ticks as numpy array [ts_ms, price]."""
    con = sqlite3.connect(DB_PATH)
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    cur = con.execute(
        'SELECT ts_ms, price FROM ticks WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms',
        (symbol, start_ts, end_ts)
    )
    rows = cur.fetchall()
    con.close()
    if not rows:
        return np.array([])
    return np.array(rows, dtype=np.float64)

def ticks_to_ohlcv(ticks: np.ndarray, period_minutes: int) -> np.ndarray:
    """Convert raw ticks to OHLCV candles. Returns [ts_ms, open, high, low, close, volume].
    Fast O(n) implementation using sorted split indices."""
    if len(ticks) == 0:
        return np.array([])
    period_ms = period_minutes * 60 * 1000
    # Assign each tick to a period bucket
    periods = (ticks[:, 0] // period_ms).astype(np.int64)
    prices = ticks[:, 1]
    
    # Use np.unique with return_index/return_counts for O(n log n) split
    unique_periods, first_idx, counts = np.unique(periods, return_index=True, return_counts=True)
    
    n = len(unique_periods)
    result = np.empty((n, 6), dtype=np.float64)
    
    for i in range(n):
        start = first_idx[i]
        end = start + counts[i]
        chunk_prices = prices[start:end]
        result[i, 0] = ticks[start, 0]       # ts_ms (first tick of period)
        result[i, 1] = chunk_prices[0]         # open
        result[i, 2] = chunk_prices.max()      # high
        result[i, 3] = chunk_prices.min()      # low
        result[i, 4] = chunk_prices[-1]        # close
        result[i, 5] = float(counts[i])        # volume (tick count)
    
    return result

def build_mtf_library(symbols: List[str], days_back: int = 12) -> Dict[str, Dict[str, np.ndarray]]:
    """Build multi-timeframe OHLCV library for all symbols."""
    library = {}
    timeframes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240}
    for sym in symbols:
        print(f"  Loading {sym}...", end='', flush=True)
        ticks = load_ticks_for_symbol(sym, days_back)
        if len(ticks) < 100:
            print(f" SKIP (only {len(ticks)} ticks)")
            continue
        library[sym] = {}
        for tf, mins in timeframes.items():
            ohlcv = ticks_to_ohlcv(ticks, mins)
            library[sym][tf] = ohlcv
        cnt_1m = len(library[sym]['1m'])
        print(f" {len(ticks):,} ticks → {cnt_1m} 1m candles")
    return library

# ============================================================
# TECHNICAL INDICATORS (vectorized)
# ============================================================

def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < period + 1:
        return np.full(len(closes), 50.0)
    deltas = np.diff(closes)
    rsi = np.full(len(closes), 50.0)
    for i in range(period, len(closes)):
        window = deltas[i-period:i]
        gains = window[window > 0].sum() / period
        losses = -window[window < 0].sum() / period
        if losses == 0:
            rsi[i] = 100.0
        else:
            rs = gains / losses
            rsi[i] = 100 - 100 / (1 + rs)
    return rsi

def calc_ema(closes: np.ndarray, period: int) -> np.ndarray:
    if len(closes) == 0:
        return np.array([])
    k = 2.0 / (period + 1)
    ema = np.full(len(closes), closes[0])
    for i in range(1, len(closes)):
        ema[i] = closes[i] * k + ema[i-1] * (1 - k)
    return ema

def calc_sma(closes: np.ndarray, period: int) -> np.ndarray:
    if len(closes) < period:
        return np.full(len(closes), closes.mean() if len(closes) > 0 else 0.0)
    result = np.full(len(closes), np.nan)
    for i in range(period-1, len(closes)):
        result[i] = closes[i-period+1:i+1].mean()
    return result

def calc_bb(closes: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Returns (upper, mid, lower) bands."""
    mid = calc_sma(closes, period)
    rolling_std = np.full(len(closes), np.nan)
    for i in range(period-1, len(closes)):
        rolling_std[i] = closes[i-period+1:i+1].std()
    upper = mid + std_mult * rolling_std
    lower = mid - std_mult * rolling_std
    return upper, mid, lower

def calc_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_atr(ohlcv: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    if len(ohlcv) < 2:
        return np.zeros(len(ohlcv))
    highs = ohlcv[:, 2]
    lows = ohlcv[:, 3]
    closes = ohlcv[:, 4]
    tr = np.maximum(highs - lows, 
         np.maximum(np.abs(highs - np.roll(closes, 1)),
                    np.abs(lows - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    atr = calc_ema(tr, period)
    return atr

def calc_vwap(ohlcv: np.ndarray, period: int = 20) -> np.ndarray:
    """VWAP with rolling window."""
    closes = ohlcv[:, 4]
    volumes = ohlcv[:, 5]
    typical = (ohlcv[:, 2] + ohlcv[:, 3] + closes) / 3
    vwap = np.full(len(ohlcv), np.nan)
    for i in range(period-1, len(ohlcv)):
        pv = typical[i-period+1:i+1] * volumes[i-period+1:i+1]
        v = volumes[i-period+1:i+1]
        vwap[i] = pv.sum() / v.sum() if v.sum() > 0 else closes[i]
    return vwap

def calc_stoch_rsi(closes: np.ndarray, rsi_period: int = 14, stoch_period: int = 14) -> np.ndarray:
    """Stochastic RSI oscillator (0-100)."""
    rsi = calc_rsi(closes, rsi_period)
    stoch = np.full(len(closes), 50.0)
    for i in range(stoch_period-1, len(closes)):
        window = rsi[i-stoch_period+1:i+1]
        lo, hi = window.min(), window.max()
        if hi - lo > 0:
            stoch[i] = (rsi[i] - lo) / (hi - lo) * 100
        else:
            stoch[i] = 50.0
    return stoch

# ============================================================
# BACKTEST ENGINE (vectorized, fast)
# ============================================================

def backtest_signals(
    ohlcv: np.ndarray,
    signals: np.ndarray,  # +1=BUY, -1=SELL, 0=HOLD
    sl_pct: float = 2.0,
    tp_pct: float = 4.0,
    hold_periods: int = 10,
    fee_pct: float = 0.06  # 0.06% per trade (taker)
) -> Dict[str, float]:
    """Fast backtest: given signal array, simulate trades with SL/TP."""
    if len(ohlcv) < 10:
        return {'num_trades': 0, 'win_rate': 0, 'sharpe_ratio': -99, 'total_pnl_pct': 0, 'avg_pnl_pct': 0, 'max_drawdown_pct': 0}
    
    closes = ohlcv[:, 4]
    highs = ohlcv[:, 2]
    lows = ohlcv[:, 3]
    
    trades = []
    in_position = False
    entry_idx = 0
    entry_price = 0.0
    side = 0
    
    for i in range(len(signals)):
        if not in_position:
            if signals[i] != 0:
                in_position = True
                entry_idx = i
                entry_price = closes[i]
                side = signals[i]
        else:
            # Check SL/TP
            current_close = closes[i]
            days_held = i - entry_idx
            
            if side == 1:  # LONG
                sl_price = entry_price * (1 - sl_pct / 100)
                tp_price = entry_price * (1 + tp_pct / 100)
                if lows[i] <= sl_price:
                    pnl = (sl_price / entry_price - 1) * 100 - fee_pct * 2
                    trades.append(pnl)
                    in_position = False
                elif highs[i] >= tp_price:
                    pnl = (tp_price / entry_price - 1) * 100 - fee_pct * 2
                    trades.append(pnl)
                    in_position = False
                elif days_held >= hold_periods:
                    pnl = (current_close / entry_price - 1) * 100 - fee_pct * 2
                    trades.append(pnl)
                    in_position = False
            else:  # SHORT
                sl_price = entry_price * (1 + sl_pct / 100)
                tp_price = entry_price * (1 - tp_pct / 100)
                if highs[i] >= sl_price:
                    pnl = (entry_price / sl_price - 1) * 100 - fee_pct * 2
                    trades.append(pnl)
                    in_position = False
                elif lows[i] <= tp_price:
                    pnl = (entry_price / tp_price - 1) * 100 - fee_pct * 2
                    trades.append(pnl)
                    in_position = False
                elif days_held >= hold_periods:
                    pnl = (entry_price / current_close - 1) * 100 - fee_pct * 2
                    trades.append(pnl)
                    in_position = False
    
    if not trades:
        return {'num_trades': 0, 'win_rate': 0, 'sharpe_ratio': -99, 'total_pnl_pct': 0, 'avg_pnl_pct': 0, 'max_drawdown_pct': 0}
    
    trades_arr = np.array(trades)
    wins = (trades_arr > 0).sum()
    num = len(trades_arr)
    win_rate = wins / num
    avg_pnl = trades_arr.mean()
    total_pnl = trades_arr.sum()
    
    # Equity curve
    equity = np.cumprod(1 + trades_arr / 100) * INITIAL_CAPITAL
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max * 100
    max_dd = drawdown.max()
    
    # Sharpe (trade-by-trade returns)
    # GUARD: Sharpe is statistically meaningless below 30 samples — suppress noisy values.
    MIN_TRADES_FOR_SHARPE = 30
    returns = trades_arr / 100
    if len(returns) >= MIN_TRADES_FOR_SHARPE and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0  # Insufficient samples — return 0 rather than noisy estimate

    # Anomaly detection: flag impossible metric combinations
    if win_rate < 0.05 and sharpe > 3.0:
        print(f"  ⚠️  ANOMALY: win_rate={win_rate:.1%} < 5% but sharpe={sharpe:.2f} > 3 — clamping sharpe to 0")
        sharpe = 0.0  # Physically impossible combination; discard Sharpe value

    return {
        'num_trades': num,
        'win_rate': win_rate,
        'sharpe_ratio': float(sharpe),
        'total_pnl_pct': float(total_pnl),
        'avg_pnl_pct': float(avg_pnl),
        'max_drawdown_pct': float(max_dd)
    }

# ============================================================
# STRATEGY SIGNAL GENERATORS
# ============================================================

def strategy_rsi_mean_revert(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """RSI mean reversion: buy oversold, sell overbought."""
    closes = ohlcv[:, 4]
    period = params.get('period', 14)
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)
    rsi = calc_rsi(closes, period)
    signals = np.zeros(len(ohlcv))
    for i in range(period + 1, len(ohlcv)):
        if rsi[i-1] < oversold and rsi[i] >= oversold:
            signals[i] = 1  # BUY (RSI bouncing up from oversold)
        elif rsi[i-1] > overbought and rsi[i] <= overbought:
            signals[i] = -1  # SELL (RSI dropping from overbought)
    return signals

def strategy_rsi_momentum(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """RSI momentum: buy when RSI crosses above mid, sell below."""
    closes = ohlcv[:, 4]
    period = params.get('period', 14)
    mid_up = params.get('mid_up', 55)
    mid_dn = params.get('mid_dn', 45)
    rsi = calc_rsi(closes, period)
    signals = np.zeros(len(ohlcv))
    for i in range(period + 1, len(ohlcv)):
        if rsi[i-1] < mid_up and rsi[i] >= mid_up:
            signals[i] = 1
        elif rsi[i-1] > mid_dn and rsi[i] <= mid_dn:
            signals[i] = -1
    return signals

def strategy_ema_cross(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """EMA crossover."""
    closes = ohlcv[:, 4]
    fast = params.get('fast', 9)
    slow = params.get('slow', 21)
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    signals = np.zeros(len(ohlcv))
    for i in range(slow + 1, len(ohlcv)):
        prev_diff = ema_fast[i-1] - ema_slow[i-1]
        curr_diff = ema_fast[i] - ema_slow[i]
        if prev_diff < 0 and curr_diff >= 0:
            signals[i] = 1
        elif prev_diff > 0 and curr_diff <= 0:
            signals[i] = -1
    return signals

def strategy_macd_signal(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """MACD histogram crossover."""
    closes = ohlcv[:, 4]
    fast = params.get('fast', 12)
    slow = params.get('slow', 26)
    signal = params.get('signal', 9)
    _, _, hist = calc_macd(closes, fast, slow, signal)
    signals = np.zeros(len(ohlcv))
    for i in range(slow + signal + 1, len(ohlcv)):
        if hist[i-1] < 0 and hist[i] >= 0:
            signals[i] = 1
        elif hist[i-1] > 0 and hist[i] <= 0:
            signals[i] = -1
    return signals

def strategy_bb_reversion(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """Bollinger Band mean reversion: buy below lower band, sell above upper."""
    closes = ohlcv[:, 4]
    period = params.get('period', 20)
    std_mult = params.get('std_mult', 2.0)
    upper, mid, lower = calc_bb(closes, period, std_mult)
    signals = np.zeros(len(ohlcv))
    for i in range(period + 1, len(ohlcv)):
        if not np.isnan(lower[i]) and not np.isnan(upper[i]):
            if closes[i-1] < lower[i-1] and closes[i] >= lower[i]:
                signals[i] = 1
            elif closes[i-1] > upper[i-1] and closes[i] <= upper[i]:
                signals[i] = -1
    return signals

def strategy_bb_breakout(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """BB breakout: buy when price breaks above upper band."""
    closes = ohlcv[:, 4]
    period = params.get('period', 20)
    std_mult = params.get('std_mult', 2.0)
    upper, mid, lower = calc_bb(closes, period, std_mult)
    signals = np.zeros(len(ohlcv))
    for i in range(period + 1, len(ohlcv)):
        if not np.isnan(lower[i]) and not np.isnan(upper[i]):
            if closes[i-1] <= upper[i-1] and closes[i] > upper[i]:
                signals[i] = 1
            elif closes[i-1] >= lower[i-1] and closes[i] < lower[i]:
                signals[i] = -1
    return signals

def strategy_vwap_reversion(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """VWAP mean reversion."""
    closes = ohlcv[:, 4]
    period = params.get('period', 20)
    dev_pct = params.get('dev_pct', 1.5)
    vwap = calc_vwap(ohlcv, period)
    signals = np.zeros(len(ohlcv))
    for i in range(period + 1, len(ohlcv)):
        if not np.isnan(vwap[i]):
            dev = (closes[i] - vwap[i]) / vwap[i] * 100
            prev_dev = (closes[i-1] - vwap[i-1]) / vwap[i-1] * 100 if not np.isnan(vwap[i-1]) else 0
            if prev_dev <= -dev_pct and dev > -dev_pct:
                signals[i] = 1
            elif prev_dev >= dev_pct and dev < dev_pct:
                signals[i] = -1
    return signals

def strategy_momentum_breakout(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """Price momentum breakout over N periods."""
    closes = ohlcv[:, 4]
    lookback = params.get('lookback', 10)
    threshold_pct = params.get('threshold_pct', 1.5)
    signals = np.zeros(len(ohlcv))
    for i in range(lookback + 1, len(ohlcv)):
        change = (closes[i] / closes[i - lookback] - 1) * 100
        if change >= threshold_pct:
            signals[i] = 1
        elif change <= -threshold_pct:
            signals[i] = -1
    return signals

def strategy_stoch_rsi(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """Stochastic RSI signals."""
    closes = ohlcv[:, 4]
    rsi_p = params.get('rsi_period', 14)
    stoch_p = params.get('stoch_period', 14)
    oversold = params.get('oversold', 20)
    overbought = params.get('overbought', 80)
    srsi = calc_stoch_rsi(closes, rsi_p, stoch_p)
    signals = np.zeros(len(ohlcv))
    for i in range(rsi_p + stoch_p + 1, len(ohlcv)):
        if srsi[i-1] < oversold and srsi[i] >= oversold:
            signals[i] = 1
        elif srsi[i-1] > overbought and srsi[i] <= overbought:
            signals[i] = -1
    return signals

def strategy_triple_ema(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """Triple EMA trend filter (fast>mid>slow=BUY, fast<mid<slow=SELL)."""
    closes = ohlcv[:, 4]
    fast = params.get('fast', 5)
    mid = params.get('mid', 13)
    slow = params.get('slow', 34)
    e_fast = calc_ema(closes, fast)
    e_mid = calc_ema(closes, mid)
    e_slow = calc_ema(closes, slow)
    signals = np.zeros(len(ohlcv))
    for i in range(slow + 1, len(ohlcv)):
        prev_bull = e_fast[i-1] > e_mid[i-1] > e_slow[i-1]
        curr_bull = e_fast[i] > e_mid[i] > e_slow[i]
        prev_bear = e_fast[i-1] < e_mid[i-1] < e_slow[i-1]
        curr_bear = e_fast[i] < e_mid[i] < e_slow[i]
        if not prev_bull and curr_bull:
            signals[i] = 1
        elif not prev_bear and curr_bear:
            signals[i] = -1
    return signals

def strategy_rsi_bb_combo(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """RSI + BB combo: buy when RSI oversold AND price at lower band."""
    closes = ohlcv[:, 4]
    rsi_period = params.get('rsi_period', 14)
    bb_period = params.get('bb_period', 20)
    std_mult = params.get('std_mult', 2.0)
    rsi_oversold = params.get('rsi_oversold', 35)
    rsi_overbought = params.get('rsi_overbought', 65)
    rsi = calc_rsi(closes, rsi_period)
    upper, mid, lower = calc_bb(closes, bb_period, std_mult)
    signals = np.zeros(len(ohlcv))
    for i in range(max(rsi_period, bb_period) + 1, len(ohlcv)):
        if not np.isnan(lower[i]):
            if rsi[i] < rsi_oversold and closes[i] < lower[i]:
                signals[i] = 1
            elif rsi[i] > rsi_overbought and closes[i] > upper[i]:
                signals[i] = -1
    return signals

def strategy_macd_rsi_confirm(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """MACD crossover confirmed by RSI direction."""
    closes = ohlcv[:, 4]
    macd_fast = params.get('macd_fast', 12)
    macd_slow = params.get('macd_slow', 26)
    macd_sig = params.get('macd_signal', 9)
    rsi_period = params.get('rsi_period', 14)
    rsi_min = params.get('rsi_min', 40)
    rsi_max = params.get('rsi_max', 60)
    _, _, hist = calc_macd(closes, macd_fast, macd_slow, macd_sig)
    rsi = calc_rsi(closes, rsi_period)
    signals = np.zeros(len(ohlcv))
    start = max(macd_slow + macd_sig, rsi_period) + 1
    for i in range(start, len(ohlcv)):
        if hist[i-1] < 0 and hist[i] >= 0 and rsi[i] > rsi_min:
            signals[i] = 1
        elif hist[i-1] > 0 and hist[i] <= 0 and rsi[i] < rsi_max:
            signals[i] = -1
    return signals

def strategy_ema_vwap_bounce(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """Price bouncing off EMA & VWAP confluence."""
    closes = ohlcv[:, 4]
    ema_period = params.get('ema_period', 21)
    vwap_period = params.get('vwap_period', 20)
    band_pct = params.get('band_pct', 0.5)
    ema = calc_ema(closes, ema_period)
    vwap = calc_vwap(ohlcv, vwap_period)
    signals = np.zeros(len(ohlcv))
    start = max(ema_period, vwap_period) + 1
    for i in range(start, len(ohlcv)):
        if np.isnan(vwap[i]):
            continue
        support = (ema[i] + vwap[i]) / 2
        band = support * band_pct / 100
        if closes[i-1] < support - band and closes[i] >= support - band:
            signals[i] = 1
        elif closes[i-1] > support + band and closes[i] <= support + band:
            signals[i] = -1
    return signals

def strategy_volume_breakout(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """High volume + price breakout to new highs/lows."""
    closes = ohlcv[:, 4]
    highs = ohlcv[:, 2]
    lows = ohlcv[:, 3]
    volumes = ohlcv[:, 5]
    lookback = params.get('lookback', 20)
    vol_mult = params.get('vol_mult', 2.0)
    signals = np.zeros(len(ohlcv))
    for i in range(lookback + 1, len(ohlcv)):
        avg_vol = volumes[i-lookback:i].mean()
        high_n = highs[i-lookback:i].max()
        low_n = lows[i-lookback:i].min()
        if volumes[i] > avg_vol * vol_mult:
            if closes[i] > high_n:
                signals[i] = 1
            elif closes[i] < low_n:
                signals[i] = -1
    return signals

def strategy_mean_revert_atr(ohlcv: np.ndarray, params: Dict) -> np.ndarray:
    """ATR-based mean reversion: price far from EMA in ATR units."""
    closes = ohlcv[:, 4]
    ema_period = params.get('ema_period', 20)
    atr_period = params.get('atr_period', 14)
    atr_mult = params.get('atr_mult', 2.0)
    ema = calc_ema(closes, ema_period)
    atr = calc_atr(ohlcv, atr_period)
    signals = np.zeros(len(ohlcv))
    start = max(ema_period, atr_period) + 1
    for i in range(start, len(ohlcv)):
        dist = closes[i] - ema[i]
        prev_dist = closes[i-1] - ema[i-1]
        threshold = atr[i] * atr_mult
        if prev_dist < -threshold and dist >= -threshold:
            signals[i] = 1
        elif prev_dist > threshold and dist <= threshold:
            signals[i] = -1
    return signals

# ============================================================
# MULTI-TIMEFRAME STRATEGIES
# ============================================================

def strategy_mtf_trend_align(ohlcv_tf1: np.ndarray, ohlcv_tf2: np.ndarray, params: Dict) -> np.ndarray:
    """
    Multi-TF: Higher TF trend filter + lower TF entry.
    tf1=lower (signal), tf2=higher (filter).
    Returns signals aligned to tf1 timestamps.
    """
    # Higher TF trend: EMA direction
    closes_h = ohlcv_tf2[:, 4]
    ema_h = calc_ema(closes_h, params.get('ema_h', 20))
    # Low TF RSI signal
    closes_l = ohlcv_tf1[:, 4]
    rsi_l = calc_rsi(closes_l, params.get('rsi_period', 14))
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)
    
    signals = np.zeros(len(ohlcv_tf1))
    
    for i in range(params.get('rsi_period', 14) + 1, len(ohlcv_tf1)):
        ts = ohlcv_tf1[i, 0]
        # Find corresponding higher TF candle
        h_idx = np.searchsorted(ohlcv_tf2[:, 0], ts, side='right') - 1
        if h_idx < params.get('ema_h', 20) or h_idx >= len(ohlcv_tf2):
            continue
        # Higher TF trend direction
        uptrend = closes_h[h_idx] > ema_h[h_idx]
        downtrend = closes_h[h_idx] < ema_h[h_idx]
        # Signal only in trend direction
        if uptrend and rsi_l[i-1] < oversold and rsi_l[i] >= oversold:
            signals[i] = 1
        elif downtrend and rsi_l[i-1] > overbought and rsi_l[i] <= overbought:
            signals[i] = -1
    return signals

def strategy_mtf_momentum_confirm(ohlcv_tf1: np.ndarray, ohlcv_tf2: np.ndarray, params: Dict) -> np.ndarray:
    """MTF: MACD signal on lower TF confirmed by higher TF momentum."""
    closes_l = ohlcv_tf1[:, 4]
    _, _, hist_l = calc_macd(closes_l, params.get('fast', 12), params.get('slow', 26), params.get('signal', 9))
    
    closes_h = ohlcv_tf2[:, 4]
    mom_period = params.get('mom_period', 5)
    
    signals = np.zeros(len(ohlcv_tf1))
    start = params.get('slow', 26) + params.get('signal', 9) + 1
    
    for i in range(start, len(ohlcv_tf1)):
        ts = ohlcv_tf1[i, 0]
        h_idx = np.searchsorted(ohlcv_tf2[:, 0], ts, side='right') - 1
        if h_idx < mom_period or h_idx >= len(ohlcv_tf2):
            continue
        h_mom = (closes_h[h_idx] / closes_h[h_idx - mom_period] - 1) * 100
        if hist_l[i-1] < 0 and hist_l[i] >= 0 and h_mom > 0:
            signals[i] = 1
        elif hist_l[i-1] > 0 and hist_l[i] <= 0 and h_mom < 0:
            signals[i] = -1
    return signals

def strategy_mtf_bb_squeeze_confirm(ohlcv_tf1: np.ndarray, ohlcv_tf2: np.ndarray, params: Dict) -> np.ndarray:
    """MTF: BB squeeze break on lower TF, confirmed by higher TF trend."""
    closes_l = ohlcv_tf1[:, 4]
    bb_p = params.get('bb_period', 20)
    std_mult = params.get('std_mult', 2.0)
    upper_l, mid_l, lower_l = calc_bb(closes_l, bb_p, std_mult)
    
    closes_h = ohlcv_tf2[:, 4]
    ema_h = calc_ema(closes_h, params.get('ema_h', 50))
    
    signals = np.zeros(len(ohlcv_tf1))
    
    for i in range(bb_p + 1, len(ohlcv_tf1)):
        if np.isnan(upper_l[i]) or np.isnan(lower_l[i]):
            continue
        ts = ohlcv_tf1[i, 0]
        h_idx = np.searchsorted(ohlcv_tf2[:, 0], ts, side='right') - 1
        if h_idx < params.get('ema_h', 50) or h_idx >= len(ohlcv_tf2):
            continue
        h_up = closes_h[h_idx] > ema_h[h_idx]
        h_dn = closes_h[h_idx] < ema_h[h_idx]
        if h_up and closes_l[i-1] <= upper_l[i-1] and closes_l[i] > upper_l[i]:
            signals[i] = 1
        elif h_dn and closes_l[i-1] >= lower_l[i-1] and closes_l[i] < lower_l[i]:
            signals[i] = -1
    return signals

# ============================================================
# STRATEGY REGISTRY
# ============================================================

SINGLE_TF_STRATEGIES = {
    'rsi_mean_revert': strategy_rsi_mean_revert,
    'rsi_momentum': strategy_rsi_momentum,
    'ema_cross': strategy_ema_cross,
    'macd_signal': strategy_macd_signal,
    'bb_reversion': strategy_bb_reversion,
    'bb_breakout': strategy_bb_breakout,
    'vwap_reversion': strategy_vwap_reversion,
    'momentum_breakout': strategy_momentum_breakout,
    'stoch_rsi': strategy_stoch_rsi,
    'triple_ema': strategy_triple_ema,
    'rsi_bb_combo': strategy_rsi_bb_combo,
    'macd_rsi_confirm': strategy_macd_rsi_confirm,
    'ema_vwap_bounce': strategy_ema_vwap_bounce,
    'volume_breakout': strategy_volume_breakout,
    'mean_revert_atr': strategy_mean_revert_atr,
}

MTF_STRATEGIES = {
    'mtf_trend_align': strategy_mtf_trend_align,
    'mtf_momentum_confirm': strategy_mtf_momentum_confirm,
    'mtf_bb_squeeze_confirm': strategy_mtf_bb_squeeze_confirm,
}

# Default parameter sets for first iteration
DEFAULT_PARAMS = {
    'rsi_mean_revert': {'period': 14, 'oversold': 30, 'overbought': 70},
    'rsi_momentum': {'period': 14, 'mid_up': 55, 'mid_dn': 45},
    'ema_cross': {'fast': 9, 'slow': 21},
    'macd_signal': {'fast': 12, 'slow': 26, 'signal': 9},
    'bb_reversion': {'period': 20, 'std_mult': 2.0},
    'bb_breakout': {'period': 20, 'std_mult': 2.0},
    'vwap_reversion': {'period': 20, 'dev_pct': 1.5},
    'momentum_breakout': {'lookback': 10, 'threshold_pct': 1.5},
    'stoch_rsi': {'rsi_period': 14, 'stoch_period': 14, 'oversold': 20, 'overbought': 80},
    'triple_ema': {'fast': 5, 'mid': 13, 'slow': 34},
    'rsi_bb_combo': {'rsi_period': 14, 'bb_period': 20, 'std_mult': 2.0, 'rsi_oversold': 35, 'rsi_overbought': 65},
    'macd_rsi_confirm': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'rsi_period': 14, 'rsi_min': 40, 'rsi_max': 60},
    'ema_vwap_bounce': {'ema_period': 21, 'vwap_period': 20, 'band_pct': 0.5},
    'volume_breakout': {'lookback': 20, 'vol_mult': 2.0},
    'mean_revert_atr': {'ema_period': 20, 'atr_period': 14, 'atr_mult': 2.0},
    # MTF strategies
    'mtf_trend_align': {'ema_h': 20, 'rsi_period': 14, 'oversold': 30, 'overbought': 70},
    'mtf_momentum_confirm': {'fast': 12, 'slow': 26, 'signal': 9, 'mom_period': 5},
    'mtf_bb_squeeze_confirm': {'bb_period': 20, 'std_mult': 2.0, 'ema_h': 50},
}

# SL/TP/hold combos to test
RISK_PARAMS = [
    {'sl_pct': 1.5, 'tp_pct': 3.0, 'hold_periods': 8},
    {'sl_pct': 2.0, 'tp_pct': 4.0, 'hold_periods': 10},
    {'sl_pct': 2.5, 'tp_pct': 5.0, 'hold_periods': 12},
    {'sl_pct': 1.0, 'tp_pct': 2.5, 'hold_periods': 6},
    {'sl_pct': 3.0, 'tp_pct': 6.0, 'hold_periods': 15},
]

# TF combos to test per strategy type
SINGLE_TF_COMBOS = ['1m', '5m', '15m', '1h']
MTF_COMBOS = [('5m', '1h'), ('15m', '4h'), ('1m', '15m'), ('5m', '4h')]

# ============================================================
# CORE OPTIMIZATION FUNCTIONS
# ============================================================

def run_single_strategy_sweep(
    library: Dict[str, Dict[str, np.ndarray]],
    strategy_name: str,
    strategy_fn,
    params: Dict,
    timeframes: List[str],
    risk_params: List[Dict],
    symbols: List[str]
) -> List[Dict]:
    """Run a strategy across all symbols, timeframes, and risk params."""
    results = []
    for sym in symbols:
        if sym not in library:
            continue
        for tf in timeframes:
            if tf not in library[sym]:
                continue
            ohlcv = library[sym][tf]
            if len(ohlcv) < 50:
                continue
            try:
                signals = strategy_fn(ohlcv, params)
                for rp in risk_params:
                    metrics = backtest_signals(ohlcv, signals, **rp)
                    if metrics['num_trades'] >= 5:
                        results.append({
                            'strategy': strategy_name,
                            'symbol': sym,
                            'timeframe': tf,
                            'params': params.copy(),
                            'risk_params': rp.copy(),
                            'metrics': metrics,
                            'sharpe': metrics['sharpe_ratio'],
                            'win_rate': metrics['win_rate'],
                        })
            except Exception as e:
                pass
    return results

def run_mtf_strategy_sweep(
    library: Dict[str, Dict[str, np.ndarray]],
    strategy_name: str,
    strategy_fn,
    params: Dict,
    tf_combos: List[Tuple],
    risk_params: List[Dict],
    symbols: List[str]
) -> List[Dict]:
    """Run MTF strategy across all combinations."""
    results = []
    for sym in symbols:
        if sym not in library:
            continue
        for tf_low, tf_high in tf_combos:
            if tf_low not in library[sym] or tf_high not in library[sym]:
                continue
            ohlcv_l = library[sym][tf_low]
            ohlcv_h = library[sym][tf_high]
            if len(ohlcv_l) < 50 or len(ohlcv_h) < 20:
                continue
            try:
                signals = strategy_fn(ohlcv_l, ohlcv_h, params)
                for rp in risk_params:
                    metrics = backtest_signals(ohlcv_l, signals, **rp)
                    if metrics['num_trades'] >= 5:
                        results.append({
                            'strategy': strategy_name,
                            'symbol': sym,
                            'timeframe': f"{tf_low}/{tf_high}",
                            'params': params.copy(),
                            'risk_params': rp.copy(),
                            'metrics': metrics,
                            'sharpe': metrics['sharpe_ratio'],
                            'win_rate': metrics['win_rate'],
                        })
            except Exception as e:
                pass
    return results

def rank_results(results: List[Dict], top_n: int = 20) -> List[Dict]:
    """Rank results by EEP score. Falls back to Sharpe composite if eep_scorer unavailable."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from eep_scorer import compute_eep_from_metrics
        _has_eep = True
    except ImportError:
        _has_eep = False

    for r in results:
        m = r['metrics']
        if _has_eep:
            # Build strategy dict compatible with compute_eep_from_metrics
            s = {
                "win_rate":          m.get("win_rate", 0),
                "sharpe_ratio":      m.get("sharpe_ratio", 0),
                "total_pnl_pct":     m.get("total_pnl_pct", 0),
                "avg_pnl_pct":       m.get("avg_pnl_pct", 0),
                "max_drawdown_pct":  m.get("max_drawdown_pct", 0),
                "num_trades":        m.get("num_trades", 0),
                "risk_params":       r.get("risk_params", {}),
                "strategy":          r.get("strategy", ""),
            }
            eep = compute_eep_from_metrics(s)
            r['eep_score']    = eep['eep_score']
            r['eep_rank']     = None  # filled after sort
            r['gate_pass']    = eep['gate_pass']
            r['gate_fails']   = eep['gate_fails']
            r['composite_score'] = eep['eep_score']
        else:
            # Legacy composite (fallback)
            wr_bonus = max(0, (m['win_rate'] - 0.5) * 2)
            trade_penalty = 1 / (1 + np.exp(-(m['num_trades'] - 15) / 5))
            r['composite_score'] = m['sharpe_ratio'] * (1 + wr_bonus) * trade_penalty
            r['eep_score'] = r['composite_score']
            r['gate_pass'] = True
            r['gate_fails'] = []

    # Gate-passing strategies ranked first, then by EEP descending
    results.sort(key=lambda x: (0 if x.get('gate_pass', True) else 1, -x['composite_score']))

    # Assign EEP ranks
    for i, r in enumerate(results, 1):
        r['eep_rank'] = i

    return results[:top_n]

def generate_param_variations(base_params: Dict, variation_pct: float = 0.2) -> List[Dict]:
    """Generate ±variation_pct variations for each numeric parameter."""
    variations = [base_params.copy()]
    for key, val in base_params.items():
        if isinstance(val, (int, float)):
            lo_mult = 1 - variation_pct
            hi_mult = 1 + variation_pct
            if isinstance(val, int):
                lo = max(2, int(val * lo_mult))
                hi = max(lo + 1, int(val * hi_mult))
                mid = val
                for v in sorted(set([lo, mid, hi])):
                    p = base_params.copy()
                    p[key] = v
                    variations.append(p)
            else:
                for mult in [lo_mult, 1.0, hi_mult]:
                    p = base_params.copy()
                    p[key] = round(val * mult, 4)
                    variations.append(p)
    return variations

# ============================================================
# MAIN OPTIMIZATION LOOP
# ============================================================

def print_separator(char='=', width=70):
    print(char * width)

def print_result(r: Dict, rank: int):
    m = r['metrics']
    print(f"  #{rank:2d} {r['strategy']:25s} {r['symbol']:12s} {r['timeframe']:8s} | "
          f"WR={m['win_rate']:.1%} Sharpe={m['sharpe_ratio']:+.3f} "
          f"PnL={m['total_pnl_pct']:+.1f}% Trades={m['num_trades']:3d} DD={m['max_drawdown_pct']:.1f}%")

def run_full_iteration(
    library: Dict,
    current_best_params: Dict,
    iteration: int,
    variation_pct: float = 0.2,
    days_back_train: int = 10,
    days_back_validate: int = 3,
) -> Tuple[List[Dict], Dict]:
    """Run one optimization iteration. Returns top results + updated params."""
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}/10  |  variation=±{variation_pct:.0%}  |  {datetime.now().strftime('%H:%M:%S')}")
    print_separator()
    
    all_results = []
    
    # 1) Single-TF strategies
    for name, fn in SINGLE_TF_STRATEGIES.items():
        params_list = generate_param_variations(current_best_params.get(name, DEFAULT_PARAMS[name]), variation_pct)
        for params in params_list:
            res = run_single_strategy_sweep(library, name, fn, params, SINGLE_TF_COMBOS, RISK_PARAMS, SYMBOLS)
            all_results.extend(res)
    
    # 2) MTF strategies
    for name, fn in MTF_STRATEGIES.items():
        params_list = generate_param_variations(current_best_params.get(name, DEFAULT_PARAMS[name]), variation_pct)
        for params in params_list:
            res = run_mtf_strategy_sweep(library, name, fn, params, MTF_COMBOS, RISK_PARAMS, SYMBOLS)
            all_results.extend(res)
    
    print(f"  Total configurations tested: {len(all_results):,}")
    
    # 3) Rank
    top = rank_results(all_results, top_n=30)
    
    print(f"\n  TOP 15 by Composite Score:")
    for i, r in enumerate(top[:15], 1):
        print_result(r, i)
    
    # 4) Update best params from top results
    new_best_params = current_best_params.copy()
    seen_strategies = set()
    for r in top:
        strat = r['strategy']
        if strat not in seen_strategies:
            new_best_params[strat] = r['params']
            seen_strategies.add(strat)
    
    return top, new_best_params

# ============================================================
# ML FEATURE ENGINEERING (lightweight)
# ============================================================

def extract_features(ohlcv: np.ndarray, idx: int, lookback: int = 20) -> Optional[np.ndarray]:
    """Extract ML features for a single candle."""
    if idx < lookback:
        return None
    window = ohlcv[idx-lookback:idx+1]
    closes = window[:, 4]
    highs = window[:, 2]
    lows = window[:, 3]
    vols = window[:, 5]
    
    # Price-based features
    returns = np.diff(closes) / closes[:-1]
    features = [
        returns[-1],                          # last return
        returns[-5:].mean() if len(returns) >= 5 else 0,  # 5-period mean return
        returns.std() if len(returns) > 1 else 0,         # volatility
        (closes[-1] / closes[-lookback] - 1), # total period return
        # RSI-like features
        calc_rsi(closes, min(14, lookback-1))[-1] / 100,
        # BB position
        (closes[-1] - closes.mean()) / closes.std() if closes.std() > 0 else 0,
        # Volume momentum
        vols[-1] / vols.mean() if vols.mean() > 0 else 1,
        # High-Low range
        (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0,
        # Relative position in range
        (closes[-1] - lows.min()) / (highs.max() - lows.min()) if (highs.max() - lows.min()) > 0 else 0.5,
        # EMA deviation
        (closes[-1] / calc_ema(closes, min(9, lookback-1))[-1] - 1) if len(closes) > 9 else 0,
    ]
    return np.array(features)

def train_ml_classifier(ohlcv: np.ndarray, future_periods: int = 5) -> Optional[Dict]:
    """Train a simple gradient boosting classifier on price features."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import f1_score
        
        lookback = 20
        X, y = [], []
        
        for i in range(lookback, len(ohlcv) - future_periods):
            feat = extract_features(ohlcv, i, lookback)
            if feat is None:
                continue
            # Target: is future return positive enough to trade?
            future_return = (ohlcv[i + future_periods, 4] / ohlcv[i, 4] - 1) * 100
            label = 1 if future_return > 0.3 else (2 if future_return < -0.3 else 0)  # up/down/neutral
            X.append(feat)
            y.append(label)
        
        if len(X) < 100:
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        # Time series split: train on first 70%, validate on last 30%
        split_idx = int(len(X) * 0.7)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Train classifier
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_val)
        # Win rate: when we predict UP (label=1), what % of time is it correct?
        up_mask = preds == 1
        dn_mask = preds == 2
        
        up_correct = (y_val[up_mask] == 1).sum() if up_mask.sum() > 0 else 0
        dn_correct = (y_val[dn_mask] == 2).sum() if dn_mask.sum() > 0 else 0
        
        up_pred = up_mask.sum()
        dn_pred = dn_mask.sum()
        
        precision_up = up_correct / up_pred if up_pred > 0 else 0
        precision_dn = dn_correct / dn_pred if dn_pred > 0 else 0
        
        # Generate signals from model
        signals = np.zeros(len(ohlcv))
        for i in range(lookback, len(ohlcv) - future_periods):
            feat = extract_features(ohlcv, i, lookback)
            if feat is None:
                continue
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            pred = clf.predict(feat_scaled)[0]
            proba = clf.predict_proba(feat_scaled)[0]
            max_proba = proba.max()
            if pred == 1 and max_proba > 0.45:  # UP with confidence
                signals[i] = 1
            elif pred == 2 and max_proba > 0.45:  # DOWN with confidence
                signals[i] = -1
        
        return {
            'clf': clf,
            'scaler': scaler,
            'signals': signals,
            'precision_up': precision_up,
            'precision_dn': precision_dn,
            'up_pred_count': int(up_pred),
            'dn_pred_count': int(dn_pred),
        }
    except ImportError:
        return None
    except Exception as e:
        return None

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    start_time = time.time()
    print_separator('=')
    print("AGGRESSIVE STRATEGY OPTIMIZER — Multi-Timeframe, 10 Iterations")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator('=')
    
    # Build OHLCV library
    print("\nPhase 1: Building Multi-Timeframe OHLCV Library...")
    library = build_mtf_library(SYMBOLS, days_back=12)
    
    print(f"\nLibrary ready: {len(library)} symbols")
    for sym in library:
        counts = {tf: len(library[sym][tf]) for tf in library[sym]}
        print(f"  {sym}: {counts}")
    
    # Global tracking
    all_iteration_results = []
    current_best_params = DEFAULT_PARAMS.copy()
    global_best = []
    
    # Optimization loop
    for iteration in range(1, 11):
        iter_start = time.time()
        variation_pct = max(0.05, 0.25 - (iteration - 1) * 0.02)  # Tighten variations each iter
        
        top, current_best_params = run_full_iteration(
            library, current_best_params, iteration, variation_pct
        )
        
        all_iteration_results.extend(top[:10])
        
        # Track global best
        for r in top[:5]:
            if r['metrics']['sharpe_ratio'] > 0.1 and r['metrics']['win_rate'] > 0.50:
                global_best.append(r)
        
        iter_time = time.time() - iter_start
        print(f"\n  Iteration {iteration} completed in {iter_time:.1f}s")
        
        # Quick ML augmentation on top symbols/timeframes for iterations 3,6,9
        if iteration in (3, 6, 9) and top:
            print(f"\n  [ML] Training classifiers on top performers...")
            ml_results = []
            top_combos = [(r['symbol'], r['timeframe'].split('/')[0]) for r in top[:5]]
            seen = set()
            for sym, tf in top_combos:
                key = (sym, tf)
                if key in seen or sym not in library or tf not in library[sym]:
                    continue
                seen.add(key)
                ohlcv = library[sym][tf]
                ml = train_ml_classifier(ohlcv, future_periods=5)
                if ml and ml['signals'].any():
                    metrics = backtest_signals(ohlcv, ml['signals'], sl_pct=2.0, tp_pct=4.0, hold_periods=10)
                    if metrics['num_trades'] >= 5:
                        ml_results.append({
                            'strategy': f'ml_gbt_{tf}',
                            'symbol': sym,
                            'timeframe': tf,
                            'params': {'future_periods': 5},
                            'risk_params': {'sl_pct': 2.0, 'tp_pct': 4.0, 'hold_periods': 10},
                            'metrics': metrics,
                            'sharpe': metrics['sharpe_ratio'],
                            'win_rate': metrics['win_rate'],
                            'ml_precision_up': ml['precision_up'],
                            'ml_precision_dn': ml['precision_dn'],
                        })
                        print(f"    ML {sym}/{tf}: WR={metrics['win_rate']:.1%} Sharpe={metrics['sharpe_ratio']:.3f} Trades={metrics['num_trades']}")
            
            if ml_results:
                global_best.extend(ml_results)
    
    # ============================================================
    # FINAL REPORT
    # ============================================================
    total_time = time.time() - start_time
    
    print_separator('=')
    print(f"\nOPTIMIZATION COMPLETE — {total_time/60:.1f} minutes elapsed")
    print_separator('=')
    
    # Deduplicate and re-rank global best
    seen_keys = set()
    deduped = []
    for r in global_best:
        k = (r['strategy'], r['symbol'], r['timeframe'])
        if k not in seen_keys:
            seen_keys.add(k)
            deduped.append(r)
    
    # Re-rank global best
    final_ranked = rank_results(deduped, top_n=50)
    
    print("\n" + "="*70)
    print("TOP 20 STRATEGIES BY COMPOSITE SCORE (Sharpe + WR Bonus)")
    print("="*70)
    for i, r in enumerate(final_ranked[:20], 1):
        m = r['metrics']
        print(f"  #{i:2d} {r['strategy']:28s} {r['symbol']:12s} TF={r['timeframe']:8s} | "
              f"WR={m['win_rate']:.1%} Sharpe={m['sharpe_ratio']:+.3f} "
              f"PnL={m['total_pnl_pct']:+.1f}% Trades={m['num_trades']:3d} "
              f"DD={m['max_drawdown_pct']:.1f}%")
    
    # TOP 5 with full details
    print("\n" + "="*70)
    print("TOP 5 DETAILED BREAKDOWN (Ready for Deployment)")
    print("="*70)
    for i, r in enumerate(final_ranked[:5], 1):
        m = r['metrics']
        print(f"\n#{i} — {r['strategy'].upper()} on {r['symbol']} [{r['timeframe']}]")
        print(f"  Win Rate:     {m['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {m['sharpe_ratio']:.4f}")
        print(f"  Total PnL:    {m['total_pnl_pct']:+.2f}%")
        print(f"  Avg PnL/trade:{m['avg_pnl_pct']:+.4f}%")
        print(f"  Max Drawdown: {m['max_drawdown_pct']:.2f}%")
        print(f"  Num Trades:   {m['num_trades']}")
        print(f"  Strategy Params: {json.dumps(r['params'])}")
        print(f"  Risk Params:     SL={r['risk_params']['sl_pct']}% / TP={r['risk_params']['tp_pct']}% / hold={r['risk_params']['hold_periods']}bars")
    
    # Summary: strategies meeting goal criteria
    winners = [r for r in final_ranked if r['metrics']['win_rate'] > 0.52 and r['metrics']['sharpe_ratio'] > 0]
    print(f"\n{'='*70}")
    print(f"GOAL CHECK: Strategies with >52% WR AND positive Sharpe:")
    print(f"  Found: {len(winners)} qualifying strategies")
    print(f"{'='*70}")
    for i, r in enumerate(winners[:10], 1):
        m = r['metrics']
        print(f"  ✓ #{i}: {r['strategy']} | {r['symbol']} [{r['timeframe']}] "
              f"WR={m['win_rate']:.1%} Sharpe={m['sharpe_ratio']:.3f} PnL={m['total_pnl_pct']:+.1f}%")
    
    # Portfolio simulation: top 3 deployed together
    print(f"\n{'='*70}")
    print("PORTFOLIO SIMULATION: Top 3 Strategies Combined")
    print(f"{'='*70}")
    top3 = final_ranked[:3]
    all_trades = []
    for r in top3:
        m = r['metrics']
        avg_trades_per_day = m['num_trades'] / 12  # 12 days
        print(f"  Strategy: {r['strategy']} / {r['symbol']} / {r['timeframe']}")
        print(f"    WR={m['win_rate']:.1%} Sharpe={m['sharpe_ratio']:.3f} "
              f"AvgPnL={m['avg_pnl_pct']:+.4f}% {avg_trades_per_day:.1f} trades/day")
    
    if top3:
        # Combined expected daily PnL estimate
        daily_pnls = []
        for r in top3:
            m = r['metrics']
            daily_trades = m['num_trades'] / 12
            daily_pnl_est = daily_trades * m['avg_pnl_pct']
            daily_pnls.append(daily_pnl_est)
        total_daily = sum(daily_pnls)
        annual_proj = total_daily * 365
        print(f"\n  Estimated combined daily PnL: {total_daily:+.2f}%")
        print(f"  Projected annual return (gross): {annual_proj:+.1f}%")
        print(f"  Note: Actual returns will be lower due to slippage, fees, and market impact.")
    
    # Best MTF combinations
    mtf_results = [r for r in final_ranked if '/' in r['timeframe']]
    if mtf_results:
        print(f"\n{'='*70}")
        print("BEST MULTI-TIMEFRAME COMBINATIONS")
        print(f"{'='*70}")
        for i, r in enumerate(mtf_results[:5], 1):
            m = r['metrics']
            print(f"  #{i}: {r['strategy']:30s} {r['symbol']:12s} {r['timeframe']:10s} "
                  f"WR={m['win_rate']:.1%} Sharpe={m['sharpe_ratio']:.3f}")
    
    # Save results to file
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_time_min': round(total_time / 60, 1),
        'top_strategies': [
            {
                'rank': i,
                'strategy': r['strategy'],
                'symbol': r['symbol'],
                'timeframe': r['timeframe'],
                'win_rate': round(r['metrics']['win_rate'], 4),
                'sharpe_ratio': round(r['metrics']['sharpe_ratio'], 4),
                'total_pnl_pct': round(r['metrics']['total_pnl_pct'], 2),
                'avg_pnl_pct': round(r['metrics']['avg_pnl_pct'], 4),
                'max_drawdown_pct': round(r['metrics']['max_drawdown_pct'], 2),
                'num_trades': r['metrics']['num_trades'],
                'params': r['params'],
                'risk_params': r['risk_params'],
            }
            for i, r in enumerate(final_ranked[:20], 1)
        ],
        'winners_52pct_wr': len(winners),
        'best_params': {k: v for k, v in current_best_params.items()},
    }
    
    report_path = f"data/reports/optimizer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import os
    os.makedirs('data/reports', exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to: {report_path}")
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")

    # Auto-sync results to DB so the dashboard shows up-to-date data immediately
    try:
        from pathlib import Path as _Path
        import subprocess as _subprocess
        _sync_script = _Path(__file__).parent / 'sync_optimizer_to_db.py'
        if _sync_script.exists():
            print("\nSyncing results to database...")
            _subprocess.run(
                ['python3', str(_sync_script), '--file', report_path],
                timeout=30,
                check=False,
            )
        else:
            print(f"\n⚠  sync_optimizer_to_db.py not found — run it manually to update dashboard")
    except Exception as _e:
        print(f"\n⚠  DB sync failed: {_e}")

    return final_ranked

if __name__ == '__main__':
    results = main()
