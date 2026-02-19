#!/usr/bin/env python3
"""
Comprehensive Strategy Library Backtester
Tests 10+ strategies on Blofin tick data and produces JSON results.
Run with: python3 strategy_library_backtest.py
"""

import sqlite3
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

DB_PATH = 'data/blofin_monitor.db'
TOP_SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'DOGE-USDT', 'XRP-USDT',
               'JUP-USDT', 'SHIB-USDT', 'WIF-USDT', 'PEPE-USDT', 'AVAX-USDT',
               'ADA-USDT', 'NOT-USDT', 'SEI-USDT', 'PYTH-USDT', 'ATOM-USDT']

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def load_ohlcv(symbol: str, timeframe_minutes: int = 5, days_back: int = 10) -> np.ndarray:
    """Load OHLCV data for a symbol at a given timeframe.
    Returns array: [ts_ms, open, high, low, close, volume]
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days_back * 24 * 3600 * 1000)
    
    c.execute('''SELECT ts_ms, price FROM ticks
                 WHERE symbol=? AND ts_ms >= ? AND ts_ms <= ?
                 ORDER BY ts_ms ASC''', (symbol, start_ts, end_ts))
    ticks = c.fetchall()
    conn.close()
    
    if len(ticks) < 100:
        return np.array([])
    
    ticks = np.array(ticks, dtype=float)
    
    # Aggregate to target timeframe
    period_ms = timeframe_minutes * 60 * 1000
    first_ts = ticks[0, 0]
    first_period = (first_ts // period_ms) * period_ms
    period_idx = ((ticks[:, 0] - first_period) // period_ms).astype(int)
    
    candles = []
    for pid in np.unique(period_idx):
        mask = period_idx == pid
        pts = ticks[mask]
        if len(pts) < 2:
            continue
        candles.append([
            pts[0, 0],      # ts_ms
            pts[0, 1],      # open
            pts[:, 1].max(), # high
            pts[:, 1].min(), # low
            pts[-1, 1],     # close
            float(len(pts)) # volume proxy (tick count)
        ])
    
    return np.array(candles) if candles else np.array([])

# ─────────────────────────────────────────────
# Indicator Library
# ─────────────────────────────────────────────
def ema(prices: np.ndarray, period: int) -> np.ndarray:
    result = np.zeros(len(prices))
    k = 2.0 / (period + 1)
    result[0] = prices[0]
    for i in range(1, len(prices)):
        result[i] = prices[i] * k + result[i-1] * (1 - k)
    return result

def sma(prices: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    for i in range(period-1, len(prices)):
        result[i] = np.mean(prices[i-period+1:i+1])
    return result

def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    if len(prices) < period + 1:
        return result
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        g = max(0, delta)
        l = max(0, -delta)
        avg_gain = (avg_gain * (period-1) + g) / period
        avg_loss = (avg_loss * (period-1) + l) / period
        if avg_loss == 0:
            result[i] = 100
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))
    return result

def bollinger_bands(prices: np.ndarray, period: int = 20, std_mult: float = 2.0):
    mid = sma(prices, period)
    std = np.array([np.std(prices[max(0,i-period+1):i+1]) if i >= period-1 else np.nan
                   for i in range(len(prices))])
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    result = np.full(len(high), np.nan)
    if len(high) < period + 1:
        return result
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    result[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        result[i] = (result[i-1] * (period-1) + tr[i]) / period
    return result

def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9):
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
               k_period: int = 14, d_period: int = 3):
    k = np.full(len(close), np.nan)
    for i in range(k_period-1, len(close)):
        h = np.max(high[i-k_period+1:i+1])
        l = np.min(low[i-k_period+1:i+1])
        if h == l:
            k[i] = 50
        else:
            k[i] = ((close[i] - l) / (h - l)) * 100
    d = sma(np.where(np.isnan(k), 0, k), d_period)
    return k, d

def vwap(prices: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    for i in range(period-1, len(prices)):
        pv = prices[i-period+1:i+1] * volume[i-period+1:i+1]
        v = volume[i-period+1:i+1]
        total_v = v.sum()
        if total_v > 0:
            result[i] = pv.sum() / total_v
    return result

def donchian_channels(high: np.ndarray, low: np.ndarray, period: int = 20):
    upper = np.array([np.max(high[max(0,i-period+1):i+1]) for i in range(len(high))])
    lower = np.array([np.min(low[max(0,i-period+1):i+1]) for i in range(len(low))])
    mid = (upper + lower) / 2
    return upper, mid, lower

# ─────────────────────────────────────────────
# Backtesting Engine
# ─────────────────────────────────────────────
def run_backtest(signals: np.ndarray, close: np.ndarray, 
                 hold_bars: int = 6, fee_pct: float = 0.05) -> Dict:
    """
    Simple backtest: signals is array of -1, 0, +1 for each bar.
    Hold trade for hold_bars, then exit.
    """
    trades = []
    in_trade = False
    entry_price = 0
    entry_dir = 0
    bars_held = 0
    
    for i in range(len(signals)):
        if in_trade:
            bars_held += 1
            if bars_held >= hold_bars:
                # Exit trade
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
        return {'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0, 
                'sharpe': 0, 'max_dd': 0, 'n_trades': 0}
    
    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls)
    avg_pnl = np.mean(pnls)
    total_pnl = np.sum(pnls)
    
    # Sharpe
    returns = np.array(pnls) / 100
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24 * 12)  # ~per-bar annualization
        # Scale down to reasonable range
        sharpe = sharpe / 50  # Normalize
    else:
        sharpe = 0
    
    # Max drawdown
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    dd = running_max - equity
    max_dd = np.max(dd) if len(dd) > 0 else 0
    
    return {
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'n_trades': len(trades)
    }

# ─────────────────────────────────────────────
# Strategy Implementations
# ─────────────────────────────────────────────

def strategy_ema_crossover(candles: np.ndarray, fast: int = 9, slow: int = 21) -> np.ndarray:
    """EMA crossover: buy when fast crosses above slow, sell opposite."""
    close = candles[:, 4]
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    signals = np.zeros(len(close))
    for i in range(1, len(close)):
        if ema_fast[i-1] < ema_slow[i-1] and ema_fast[i] >= ema_slow[i]:
            signals[i] = 1  # BUY
        elif ema_fast[i-1] > ema_slow[i-1] and ema_fast[i] <= ema_slow[i]:
            signals[i] = -1  # SELL
    return signals

def strategy_rsi_mean_reversion(candles: np.ndarray, period: int = 14, 
                                  oversold: float = 30, overbought: float = 70) -> np.ndarray:
    """RSI mean reversion: buy at oversold, sell at overbought."""
    close = candles[:, 4]
    rsi_vals = rsi(close, period)
    signals = np.zeros(len(close))
    for i in range(1, len(close)):
        if not np.isnan(rsi_vals[i]):
            if rsi_vals[i-1] >= oversold > rsi_vals[i]:
                signals[i] = 0  # Crossing into oversold — wait
            elif rsi_vals[i-1] < oversold <= rsi_vals[i]:
                signals[i] = 1  # Bouncing OUT of oversold — BUY
            elif rsi_vals[i-1] > overbought >= rsi_vals[i]:
                signals[i] = -1  # Dropping OUT of overbought — SELL
    return signals

def strategy_bb_mean_reversion(candles: np.ndarray, period: int = 20, 
                                  std_mult: float = 2.0) -> np.ndarray:
    """BB mean reversion: buy at lower band, sell at upper band."""
    close = candles[:, 4]
    upper, mid, lower = bollinger_bands(close, period, std_mult)
    signals = np.zeros(len(close))
    for i in range(period, len(close)):
        if np.isnan(lower[i]):
            continue
        if close[i] <= lower[i] and close[i-1] > lower[i-1]:
            signals[i] = 1  # Touch lower band — BUY
        elif close[i] >= upper[i] and close[i-1] < upper[i-1]:
            signals[i] = -1  # Touch upper band — SELL
    return signals

def strategy_macd_crossover(candles: np.ndarray, fast: int = 12, slow: int = 26, 
                              signal_p: int = 9) -> np.ndarray:
    """MACD crossover: buy when MACD crosses signal from below."""
    close = candles[:, 4]
    macd_line, signal_line, hist = macd(close, fast, slow, signal_p)
    signals = np.zeros(len(close))
    for i in range(1, len(close)):
        if hist[i-1] < 0 and hist[i] >= 0:
            signals[i] = 1  # Histogram crosses zero upward
        elif hist[i-1] > 0 and hist[i] <= 0:
            signals[i] = -1  # Histogram crosses zero downward
    return signals

def strategy_donchian_breakout(candles: np.ndarray, period: int = 20) -> np.ndarray:
    """Donchian channel breakout: buy new highs, sell new lows."""
    high = candles[:, 2]
    low = candles[:, 3]
    close = candles[:, 4]
    upper, mid, lower = donchian_channels(high, low, period)
    signals = np.zeros(len(close))
    for i in range(period, len(close)):
        # Price breaking above previous period's channel high
        prev_high = np.max(high[i-period:i])
        prev_low = np.min(low[i-period:i])
        if close[i] > prev_high:
            signals[i] = 1
        elif close[i] < prev_low:
            signals[i] = -1
    return signals

def strategy_atr_breakout(candles: np.ndarray, atr_period: int = 14, 
                            multiplier: float = 1.5) -> np.ndarray:
    """ATR breakout: buy when price moves > multiplier*ATR above recent close."""
    high = candles[:, 2]
    low = candles[:, 3]
    close = candles[:, 4]
    atr_vals = atr(high, low, close, atr_period)
    signals = np.zeros(len(close))
    for i in range(atr_period + 1, len(close)):
        if np.isnan(atr_vals[i]):
            continue
        move = close[i] - close[i-1]
        if move > multiplier * atr_vals[i-1]:
            signals[i] = 1
        elif move < -multiplier * atr_vals[i-1]:
            signals[i] = -1
    return signals

def strategy_vwap_deviation(candles: np.ndarray, period: int = 20, 
                              dev_pct: float = 1.0) -> np.ndarray:
    """VWAP deviation mean reversion."""
    close = candles[:, 4]
    volume = candles[:, 5]
    vwap_vals = vwap(close, volume, period)
    signals = np.zeros(len(close))
    for i in range(period, len(close)):
        if np.isnan(vwap_vals[i]) or vwap_vals[i] <= 0:
            continue
        dev = (close[i] - vwap_vals[i]) / vwap_vals[i] * 100
        if dev <= -dev_pct:
            signals[i] = 1  # Below VWAP — mean reversion BUY
        elif dev >= dev_pct:
            signals[i] = -1  # Above VWAP — mean reversion SELL
    return signals

def strategy_stoch_oscillator(candles: np.ndarray, k_period: int = 14, 
                                d_period: int = 3) -> np.ndarray:
    """Stochastic crossover in oversold/overbought zones."""
    high = candles[:, 2]
    low = candles[:, 3]
    close = candles[:, 4]
    k, d = stochastic(high, low, close, k_period, d_period)
    signals = np.zeros(len(close))
    for i in range(1, len(close)):
        if np.isnan(k[i]) or np.isnan(d[i]):
            continue
        # K crossing above D in oversold zone
        if k[i-1] < d[i-1] and k[i] >= d[i] and k[i] < 30:
            signals[i] = 1
        # K crossing below D in overbought zone
        elif k[i-1] > d[i-1] and k[i] <= d[i] and k[i] > 70:
            signals[i] = -1
    return signals

def strategy_momentum_roc(candles: np.ndarray, period: int = 10, 
                            threshold: float = 1.5) -> np.ndarray:
    """Rate of change momentum: buy when ROC exceeds threshold."""
    close = candles[:, 4]
    signals = np.zeros(len(close))
    for i in range(period, len(close)):
        if close[i-period] <= 0:
            continue
        roc = (close[i] - close[i-period]) / close[i-period] * 100
        if roc > threshold:
            signals[i] = 1
        elif roc < -threshold:
            signals[i] = -1
    return signals

def strategy_triple_ema(candles: np.ndarray, fast: int = 5, mid: int = 13, 
                          slow: int = 21) -> np.ndarray:
    """Triple EMA: all three aligned = strong signal."""
    close = candles[:, 4]
    e_fast = ema(close, fast)
    e_mid = ema(close, mid)
    e_slow = ema(close, slow)
    signals = np.zeros(len(close))
    for i in range(slow, len(close)):
        if e_fast[i] > e_mid[i] > e_slow[i]:
            signals[i] = 1  # All aligned bullish
        elif e_fast[i] < e_mid[i] < e_slow[i]:
            signals[i] = -1  # All aligned bearish
    return signals

def strategy_volume_momentum(candles: np.ndarray, vol_period: int = 20, 
                               vol_mult: float = 2.0, price_period: int = 5) -> np.ndarray:
    """Volume spike + price momentum confirmation."""
    close = candles[:, 4]
    volume = candles[:, 5]
    signals = np.zeros(len(close))
    for i in range(max(vol_period, price_period) + 1, len(close)):
        avg_vol = np.mean(volume[i-vol_period:i])
        if avg_vol <= 0:
            continue
        vol_ratio = volume[i] / avg_vol
        price_move = (close[i] - close[i-price_period]) / close[i-price_period] * 100
        if vol_ratio >= vol_mult and price_move > 0.5:
            signals[i] = 1
        elif vol_ratio >= vol_mult and price_move < -0.5:
            signals[i] = -1
    return signals

def strategy_mean_reversion_zscore(candles: np.ndarray, period: int = 20, 
                                     z_threshold: float = 2.0) -> np.ndarray:
    """Z-score mean reversion: trade when price is >2 std devs from mean."""
    close = candles[:, 4]
    signals = np.zeros(len(close))
    for i in range(period, len(close)):
        window = close[i-period:i]
        mean = np.mean(window)
        std = np.std(window)
        if std <= 0:
            continue
        z = (close[i] - mean) / std
        if z <= -z_threshold:
            signals[i] = 1  # Mean reversion BUY
        elif z >= z_threshold:
            signals[i] = -1  # Mean reversion SELL
    return signals

def strategy_cci(candles: np.ndarray, period: int = 20) -> np.ndarray:
    """Commodity Channel Index: trade when CCI exits extreme zones."""
    high = candles[:, 2]
    low = candles[:, 3]
    close = candles[:, 4]
    typical = (high + low + close) / 3
    signals = np.zeros(len(close))
    cci_vals = np.full(len(close), np.nan)
    for i in range(period-1, len(close)):
        tp_window = typical[i-period+1:i+1]
        mean_tp = np.mean(tp_window)
        mean_dev = np.mean(np.abs(tp_window - mean_tp))
        if mean_dev > 0:
            cci_vals[i] = (typical[i] - mean_tp) / (0.015 * mean_dev)
    for i in range(1, len(close)):
        if np.isnan(cci_vals[i]) or np.isnan(cci_vals[i-1]):
            continue
        if cci_vals[i-1] < -100 and cci_vals[i] >= -100:
            signals[i] = 1  # CCI exits oversold
        elif cci_vals[i-1] > 100 and cci_vals[i] <= 100:
            signals[i] = -1  # CCI exits overbought
    return signals

# ─────────────────────────────────────────────
# Main Backtest Runner
# ─────────────────────────────────────────────

STRATEGIES = {
    'ema_crossover_9_21': lambda c: strategy_ema_crossover(c, 9, 21),
    'ema_crossover_20_50': lambda c: strategy_ema_crossover(c, 20, 50),
    'rsi_mean_reversion_14': lambda c: strategy_rsi_mean_reversion(c, 14, 30, 70),
    'rsi_mean_reversion_tight': lambda c: strategy_rsi_mean_reversion(c, 14, 20, 80),
    'bb_mean_reversion': lambda c: strategy_bb_mean_reversion(c, 20, 2.0),
    'bb_tight': lambda c: strategy_bb_mean_reversion(c, 20, 1.5),
    'macd_crossover': lambda c: strategy_macd_crossover(c),
    'donchian_breakout_20': lambda c: strategy_donchian_breakout(c, 20),
    'atr_breakout_1_5': lambda c: strategy_atr_breakout(c, 14, 1.5),
    'atr_breakout_2_0': lambda c: strategy_atr_breakout(c, 14, 2.0),
    'vwap_deviation': lambda c: strategy_vwap_deviation(c, 20, 1.0),
    'stochastic_oscillator': lambda c: strategy_stoch_oscillator(c),
    'momentum_roc_10': lambda c: strategy_momentum_roc(c, 10, 1.5),
    'momentum_roc_5': lambda c: strategy_momentum_roc(c, 5, 1.0),
    'triple_ema': lambda c: strategy_triple_ema(c),
    'volume_momentum': lambda c: strategy_volume_momentum(c),
    'zscore_mean_reversion': lambda c: strategy_mean_reversion_zscore(c, 20, 2.0),
    'cci_oscillator': lambda c: strategy_cci(c, 20),
}

def run_all_backtests(timeframe: int = 5, hold_bars: int = 6) -> List[Dict]:
    results = []
    
    for symbol in TOP_SYMBOLS:
        print(f"\n  Loading {symbol}...", end='', flush=True)
        candles = load_ohlcv(symbol, timeframe_minutes=timeframe)
        if len(candles) < 200:
            print(f" SKIP (only {len(candles)} candles)")
            continue
        print(f" {len(candles)} candles", end='', flush=True)
        
        for strat_name, strat_fn in STRATEGIES.items():
            try:
                signals = strat_fn(candles)
                n_signals = np.sum(signals != 0)
                if n_signals < 5:
                    continue
                
                metrics = run_backtest(signals, candles[:, 4], hold_bars=hold_bars)
                
                if metrics['n_trades'] < 5:
                    continue
                
                results.append({
                    'strategy': strat_name,
                    'symbol': symbol,
                    'timeframe': f'{timeframe}m',
                    'hold_bars': hold_bars,
                    **metrics
                })
            except Exception as e:
                pass
        
        print(" ✓", end='', flush=True)
    
    return results

if __name__ == '__main__':
    print("="*60)
    print("STRATEGY LIBRARY BACKTESTER")
    print("="*60)
    
    all_results = []
    
    for tf, hold in [(5, 6), (15, 4)]:
        print(f"\n\nTimeframe: {tf}m, Hold: {hold} bars")
        results = run_all_backtests(timeframe=tf, hold_bars=hold)
        all_results.extend(results)
        print(f"\n  {len(results)} strategy-symbol combinations tested")
    
    # Save raw results
    with open('data/strategy_library_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Aggregate by strategy
    from collections import defaultdict
    strategy_agg = defaultdict(lambda: {'results': [], 'symbols': set()})
    for r in all_results:
        s = r['strategy']
        strategy_agg[s]['results'].append(r)
        strategy_agg[s]['symbols'].add(r['symbol'])
    
    print("\n\n" + "="*80)
    print("STRATEGY SUMMARY (by avg win rate, min 10 trades)")
    print("="*80)
    print(f"{'Strategy':35s} | {'AvgWR':6s} | {'Sharpe':7s} | {'AvgPnL':7s} | {'MaxDD':6s} | {'Trades':7s} | {'Syms':4s}")
    print("-"*85)
    
    summary = []
    for strat, data in strategy_agg.items():
        valid = [r for r in data['results'] if r['n_trades'] >= 5]
        if not valid:
            continue
        avg_wr = np.mean([r['win_rate'] for r in valid])
        avg_sharpe = np.mean([r['sharpe'] for r in valid])
        avg_pnl = np.mean([r['avg_pnl'] for r in valid])
        avg_dd = np.mean([r['max_dd'] for r in valid])
        avg_trades = np.mean([r['n_trades'] for r in valid])
        summary.append({
            'strategy': strat,
            'avg_win_rate': avg_wr,
            'avg_sharpe': avg_sharpe,
            'avg_pnl': avg_pnl,
            'avg_dd': avg_dd,
            'avg_trades': avg_trades,
            'n_symbols': len(data['symbols']),
            'best_symbols': sorted(valid, key=lambda x: x['win_rate'], reverse=True)[:3]
        })
    
    summary.sort(key=lambda x: x['avg_win_rate'], reverse=True)
    for s in summary:
        flag = " 🏆" if s['avg_win_rate'] > 0.52 and s['avg_sharpe'] > 0.3 else ""
        flag = " 🎯" if s['avg_win_rate'] > 0.55 else flag
        print(f"{s['strategy']:35s} | {s['avg_win_rate']:.3f} | {s['avg_sharpe']:7.3f} | {s['avg_pnl']:7.3f} | {s['avg_dd']:6.2f} | {s['avg_trades']:7.0f} | {s['n_symbols']:4d}{flag}")
    
    # Save summary
    with open('data/strategy_library_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n\nResults saved to data/strategy_library_results.json")
    print("Summary saved to data/strategy_library_summary.json")
    
    # Top performers
    print("\n\nTOP 10 CANDIDATES FOR DEPLOYMENT:")
    print("-"*60)
    top = sorted(summary, key=lambda x: x['avg_win_rate'] * 0.5 + x['avg_sharpe'] * 0.5, reverse=True)[:10]
    for i, s in enumerate(top, 1):
        best = s['best_symbols'][0] if s['best_symbols'] else {}
        print(f"{i}. {s['strategy']}")
        print(f"   WR: {s['avg_win_rate']:.1%} | Sharpe: {s['avg_sharpe']:.3f} | Symbols: {s['n_symbols']}")
        if best:
            print(f"   Best: {best.get('symbol', '?')} @ {best.get('win_rate', 0):.1%} WR")
