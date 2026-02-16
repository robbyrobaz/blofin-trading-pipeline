#!/usr/bin/env python3
import asyncio
import json
import os
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import websockets
from dotenv import load_dotenv

from db import connect, init_db, insert_signal, insert_tick, upsert_heartbeat

# Try to import strategy manager (new plugin system)
try:
    from strategy_manager import StrategyManager
    import knowledge_base
    USE_PLUGIN_SYSTEM = True
    print("[ingestor] Using plugin-based strategy system")
except ImportError as e:
    print(f"[ingestor] Plugin system not available ({e}), using legacy strategies")
    USE_PLUGIN_SYSTEM = False

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

WS_URL = os.getenv("BLOFIN_WS_URL", "wss://openapi.blofin.com/ws/public")
SYMBOLS = [s.strip() for s in os.getenv("BLOFIN_SYMBOLS", "BTC-USDT,ETH-USDT").split(",") if s.strip()]
DB_PATH = os.getenv("BLOFIN_DB_PATH", str(ROOT / "data" / "blofin_monitor.db"))
DATA_DIR = Path(os.getenv("BLOFIN_DATA_DIR", str(ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MOMENTUM_WINDOW = int(os.getenv("MOMENTUM_WINDOW_SECONDS", "180"))
MOMENTUM_UP_PCT = float(os.getenv("MOMENTUM_UP_PCT", "0.60"))
MOMENTUM_DOWN_PCT = float(os.getenv("MOMENTUM_DOWN_PCT", "-0.60"))
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK_SECONDS", "900"))
BREAKOUT_BUFFER_PCT = float(os.getenv("BREAKOUT_BUFFER_PCT", "0.18"))
REVERSAL_LOOKBACK = int(os.getenv("REVERSAL_LOOKBACK_SECONDS", "600"))
REVERSAL_BOUNCE_PCT = float(os.getenv("REVERSAL_BOUNCE_PCT", "0.35"))
VWAP_LOOKBACK = int(os.getenv("VWAP_LOOKBACK_SECONDS", "1200"))
VWAP_DEVIATION_PCT = float(os.getenv("VWAP_DEVIATION_PCT", "0.40"))
RSI_WINDOW = int(os.getenv("RSI_WINDOW_SECONDS", "840"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
BB_LOOKBACK = int(os.getenv("BB_LOOKBACK_SECONDS", "1200"))
BB_STD_MULT = float(os.getenv("BB_STD_MULT", "2.0"))
BB_SQUEEZE_THRESHOLD = float(os.getenv("BB_SQUEEZE_THRESHOLD", "0.3"))
SIGNAL_COOLDOWN = int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "240"))
RECONNECT_MIN = int(os.getenv("RECONNECT_MIN_SECONDS", "2"))
RECONNECT_MAX = int(os.getenv("RECONNECT_MAX_SECONDS", "20"))

RAW_FILE = DATA_DIR / f"raw_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"

price_windows: Dict[str, Deque[Tuple[int, float]]] = defaultdict(deque)
volume_windows: Dict[str, Deque[Tuple[int, float]]] = defaultdict(deque)
last_signal_at: Dict[Tuple[str, str, str], int] = {}


def now_ms() -> int:
    return int(time.time() * 1000)


def iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def parse_price(msg: dict) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    data = msg.get("data")
    if not data:
        return None, None, None

    row = None
    if isinstance(data, list) and data:
        row = data[0]
    elif isinstance(data, dict):
        row = data

    if not isinstance(row, dict):
        return None, None, None

    symbol = row.get("instId") or row.get("symbol") or row.get("s")
    price_val = row.get("last") or row.get("lastPrice") or row.get("price") or row.get("p")
    if symbol is None or price_val is None:
        return None, None, None

    # Extract volume (Blofin provides vol24h or volCcy24h)
    volume_val = row.get("vol24h") or row.get("volCcy24h") or row.get("volume")
    
    try:
        price = float(price_val)
        volume = float(volume_val) if volume_val is not None else 0.0
        return symbol, price, volume
    except Exception:
        return None, None, None


def _trim(symbol: str, ts_ms: int) -> None:
    q = price_windows[symbol]
    v = volume_windows[symbol]
    keep_ms = max(MOMENTUM_WINDOW, BREAKOUT_LOOKBACK, REVERSAL_LOOKBACK, VWAP_LOOKBACK, RSI_WINDOW, BB_LOOKBACK) * 1000
    cutoff = ts_ms - keep_ms
    while q and q[0][0] < cutoff:
        q.popleft()
    while v and v[0][0] < cutoff:
        v.popleft()


def _slice_window(symbol: str, ts_ms: int, lookback_seconds: int) -> List[Tuple[int, float]]:
    cutoff = ts_ms - lookback_seconds * 1000
    return [(t, p) for (t, p) in price_windows[symbol] if t >= cutoff]


def _cooldown_ok(symbol: str, signal: str, strategy: str, ts_ms: int) -> bool:
    key = (symbol, signal, strategy)
    last = last_signal_at.get(key)
    if last is None or ts_ms - last >= SIGNAL_COOLDOWN * 1000:
        last_signal_at[key] = ts_ms
        return True
    return False


def detect_signals(symbol: str, price: float, ts_ms: int, volume: float = 0.0) -> List[dict]:
    out = []
    q = price_windows[symbol]
    v = volume_windows[symbol]
    q.append((ts_ms, price))
    v.append((ts_ms, volume))
    _trim(symbol, ts_ms)

    # Momentum detector
    m_window = _slice_window(symbol, ts_ms, MOMENTUM_WINDOW)
    if len(m_window) >= 2 and m_window[0][1] > 0:
        pct = ((price - m_window[0][1]) / m_window[0][1]) * 100.0
        if pct >= MOMENTUM_UP_PCT and _cooldown_ok(symbol, "BUY", "momentum", ts_ms):
            out.append({
                "signal": "BUY",
                "strategy": "momentum",
                "confidence": min(0.99, max(0.50, pct / max(MOMENTUM_UP_PCT, 0.01))),
                "details": {"window_s": MOMENTUM_WINDOW, "change_pct": round(pct, 4)},
            })
        elif pct <= MOMENTUM_DOWN_PCT and _cooldown_ok(symbol, "SELL", "momentum", ts_ms):
            out.append({
                "signal": "SELL",
                "strategy": "momentum",
                "confidence": min(0.99, max(0.50, abs(pct) / max(abs(MOMENTUM_DOWN_PCT), 0.01))),
                "details": {"window_s": MOMENTUM_WINDOW, "change_pct": round(pct, 4)},
            })

    # Simple breakout detector
    b_window = _slice_window(symbol, ts_ms, BREAKOUT_LOOKBACK)
    if len(b_window) >= 10:
        prev_prices = [p for _, p in b_window[:-1]]
        hi = max(prev_prices)
        lo = min(prev_prices)
        up_thr = hi * (1 + BREAKOUT_BUFFER_PCT / 100.0)
        dn_thr = lo * (1 - BREAKOUT_BUFFER_PCT / 100.0)
        if price >= up_thr and _cooldown_ok(symbol, "BUY", "breakout", ts_ms):
            out.append({
                "signal": "BUY",
                "strategy": "breakout",
                "confidence": 0.70,
                "details": {"lookback_s": BREAKOUT_LOOKBACK, "prev_high": hi, "threshold": up_thr},
            })
        elif price <= dn_thr and _cooldown_ok(symbol, "SELL", "breakout", ts_ms):
            out.append({
                "signal": "SELL",
                "strategy": "breakout",
                "confidence": 0.70,
                "details": {"lookback_s": BREAKOUT_LOOKBACK, "prev_low": lo, "threshold": dn_thr},
            })

    # Reversal heuristic: bounce from local low / reject local high
    r_window = _slice_window(symbol, ts_ms, REVERSAL_LOOKBACK)
    if len(r_window) >= 10:
        prices = [p for _, p in r_window]
        low = min(prices)
        high = max(prices)
        if low > 0:
            bounce_pct = ((price - low) / low) * 100.0
            if bounce_pct >= REVERSAL_BOUNCE_PCT and _cooldown_ok(symbol, "BUY", "reversal", ts_ms):
                out.append({
                    "signal": "BUY",
                    "strategy": "reversal",
                    "confidence": 0.65,
                    "details": {"lookback_s": REVERSAL_LOOKBACK, "low": low, "bounce_pct": round(bounce_pct, 4)},
                })
        if high > 0:
            reject_pct = ((high - price) / high) * 100.0
            if reject_pct >= REVERSAL_BOUNCE_PCT and _cooldown_ok(symbol, "SELL", "reversal", ts_ms):
                out.append({
                    "signal": "SELL",
                    "strategy": "reversal",
                    "confidence": 0.65,
                    "details": {"lookback_s": REVERSAL_LOOKBACK, "high": high, "reject_pct": round(reject_pct, 4)},
                })

    # VWAP Mean Reversion
    vwap_window = _slice_window(symbol, ts_ms, VWAP_LOOKBACK)
    if len(vwap_window) >= 10:
        cutoff = ts_ms - VWAP_LOOKBACK * 1000
        vwap_volumes = [(t, vol) for (t, vol) in v if t >= cutoff]
        if len(vwap_volumes) == len(vwap_window) and len(vwap_volumes) > 0:
            total_pv = sum(p * vol for (_, p), (_, vol) in zip(vwap_window, vwap_volumes))
            total_v = sum(vol for _, vol in vwap_volumes)
            if total_v > 0:
                vwap = total_pv / total_v
                if vwap > 0:
                    deviation_pct = ((price - vwap) / vwap) * 100.0
                    if deviation_pct <= -VWAP_DEVIATION_PCT and _cooldown_ok(symbol, "BUY", "vwap_reversion", ts_ms):
                        out.append({
                            "signal": "BUY",
                            "strategy": "vwap_reversion",
                            "confidence": min(0.85, max(0.60, abs(deviation_pct) / max(VWAP_DEVIATION_PCT, 0.01))),
                            "details": {"lookback_s": VWAP_LOOKBACK, "vwap": round(vwap, 6), "deviation_pct": round(deviation_pct, 4)},
                        })
                    elif deviation_pct >= VWAP_DEVIATION_PCT and _cooldown_ok(symbol, "SELL", "vwap_reversion", ts_ms):
                        out.append({
                            "signal": "SELL",
                            "strategy": "vwap_reversion",
                            "confidence": min(0.85, max(0.60, abs(deviation_pct) / max(VWAP_DEVIATION_PCT, 0.01))),
                            "details": {"lookback_s": VWAP_LOOKBACK, "vwap": round(vwap, 6), "deviation_pct": round(deviation_pct, 4)},
                        })

    # RSI Divergence
    rsi_window = _slice_window(symbol, ts_ms, RSI_WINDOW)
    if len(rsi_window) >= 14:
        # Calculate simple RSI from price changes
        prices = [p for _, p in rsi_window]
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        elif avg_gain > 0:
            rsi = 100
        else:
            rsi = 50
        
        if rsi <= RSI_OVERSOLD and _cooldown_ok(symbol, "BUY", "rsi_divergence", ts_ms):
            out.append({
                "signal": "BUY",
                "strategy": "rsi_divergence",
                "confidence": min(0.80, max(0.55, (RSI_OVERSOLD - rsi) / RSI_OVERSOLD)),
                "details": {"window_s": RSI_WINDOW, "rsi": round(rsi, 2), "threshold": RSI_OVERSOLD},
            })
        elif rsi >= RSI_OVERBOUGHT and _cooldown_ok(symbol, "SELL", "rsi_divergence", ts_ms):
            out.append({
                "signal": "SELL",
                "strategy": "rsi_divergence",
                "confidence": min(0.80, max(0.55, (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT))),
                "details": {"window_s": RSI_WINDOW, "rsi": round(rsi, 2), "threshold": RSI_OVERBOUGHT},
            })

    # Bollinger Band Squeeze
    bb_window = _slice_window(symbol, ts_ms, BB_LOOKBACK)
    if len(bb_window) >= 20:
        prices = [p for _, p in bb_window]
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = variance ** 0.5
        
        if mean > 0 and std > 0:
            upper_band = mean + (BB_STD_MULT * std)
            lower_band = mean - (BB_STD_MULT * std)
            band_width_pct = ((upper_band - lower_band) / mean) * 100.0
            
            # Check if bands are tight (squeeze)
            is_squeeze = band_width_pct <= (BB_SQUEEZE_THRESHOLD * 100)
            
            if is_squeeze:
                # Check for breakout direction
                if price > upper_band and _cooldown_ok(symbol, "BUY", "bb_squeeze", ts_ms):
                    out.append({
                        "signal": "BUY",
                        "strategy": "bb_squeeze",
                        "confidence": 0.75,
                        "details": {
                            "lookback_s": BB_LOOKBACK,
                            "mean": round(mean, 6),
                            "upper": round(upper_band, 6),
                            "band_width_pct": round(band_width_pct, 4),
                        },
                    })
                elif price < lower_band and _cooldown_ok(symbol, "SELL", "bb_squeeze", ts_ms):
                    out.append({
                        "signal": "SELL",
                        "strategy": "bb_squeeze",
                        "confidence": 0.75,
                        "details": {
                            "lookback_s": BB_LOOKBACK,
                            "mean": round(mean, 6),
                            "lower": round(lower_band, 6),
                            "band_width_pct": round(band_width_pct, 4),
                        },
                    })

    return out


async def subscribe(ws) -> None:
    # Blofin may reject oversized subscribe payloads (code 60012).
    # Send in small batches to keep requests valid.
    batch_size = 8
    for i in range(0, len(SYMBOLS), batch_size):
        args = [{"channel": "tickers", "instId": s} for s in SYMBOLS[i:i + batch_size]]
        await ws.send(json.dumps({"op": "subscribe", "args": args}))
        await asyncio.sleep(0.15)


async def run() -> None:
    con = connect(DB_PATH)
    init_db(con)
    backoff = RECONNECT_MIN
    
    # Initialize strategy manager if using plugin system
    strategy_manager = None
    if USE_PLUGIN_SYSTEM:
        try:
            strategy_manager = StrategyManager()
            print(f"[ingestor] Strategy manager initialized with {len(strategy_manager.strategies)} strategies")
        except Exception as e:
            print(f"[ingestor] Failed to initialize strategy manager: {e}")
            print("[ingestor] Falling back to legacy strategies")
    
    # Periodic maintenance counters
    last_score_update = now_ms()
    last_auto_manage = now_ms()
    SCORE_UPDATE_INTERVAL_MS = 5 * 60 * 1000  # 5 minutes
    AUTO_MANAGE_INTERVAL_MS = 60 * 60 * 1000  # 1 hour

    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20, close_timeout=10) as ws:
                await subscribe(ws)
                backoff = RECONNECT_MIN
                async for message in ws:
                    ts = now_ms()
                    ts_iso = iso_utc(ts)
                    try:
                        msg = json.loads(message)
                    except Exception:
                        continue

                    with RAW_FILE.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"ts_ms": ts, "payload": msg}, ensure_ascii=False) + "\n")

                    symbol, price, volume = parse_price(msg)
                    if not symbol or price is None:
                        continue

                    # Update in-memory windows used by strategy detectors
                    price_windows[symbol].append((ts, price))
                    volume_windows[symbol].append((ts, volume))
                    _trim(symbol, ts)

                    insert_tick(con, {
                        "ts_ms": ts,
                        "ts_iso": ts_iso,
                        "symbol": symbol,
                        "price": price,
                        "raw_json": json.dumps(msg, ensure_ascii=False),
                    })

                    # Detect signals using plugin system or legacy code
                    if USE_PLUGIN_SYSTEM and strategy_manager:
                        try:
                            # Use new plugin-based detection
                            signal_objects = strategy_manager.detect_all(
                                symbol=symbol,
                                price=price,
                                volume=volume,
                                ts_ms=ts,
                                prices=list(price_windows[symbol]),
                                volumes=list(volume_windows[symbol])
                            )
                            
                            for sig in signal_objects:
                                row = {
                                    "ts_ms": ts,
                                    "ts_iso": ts_iso,
                                    "symbol": symbol,
                                    "signal": sig.signal,
                                    "strategy": sig.strategy,
                                    "confidence": sig.confidence,
                                    "price": price,
                                    "details_json": json.dumps(sig.details, ensure_ascii=False),
                                }
                                insert_signal(con, row)
                                print({k: row[k] for k in ["ts_iso", "symbol", "signal", "strategy", "price"]})
                        except Exception as e:
                            print(f"[ingestor] Strategy manager error: {e}, falling back to legacy")
                            # Fall back to legacy on error
                            signals = detect_signals(symbol, price, ts, volume)
                            for s in signals:
                                row = {
                                    "ts_ms": ts,
                                    "ts_iso": ts_iso,
                                    "symbol": symbol,
                                    "signal": s["signal"],
                                    "strategy": s["strategy"],
                                    "confidence": s["confidence"],
                                    "price": price,
                                    "details_json": json.dumps(s["details"], ensure_ascii=False),
                                }
                                insert_signal(con, row)
                                print({k: row[k] for k in ["ts_iso", "symbol", "signal", "strategy", "price"]})
                    else:
                        # Use legacy strategy detection
                        signals = detect_signals(symbol, price, ts, volume)
                        for s in signals:
                            row = {
                                "ts_ms": ts,
                                "ts_iso": ts_iso,
                                "symbol": symbol,
                                "signal": s["signal"],
                                "strategy": s["strategy"],
                                "confidence": s["confidence"],
                                "price": price,
                                "details_json": json.dumps(s["details"], ensure_ascii=False),
                            }
                            insert_signal(con, row)
                            print({k: row[k] for k in ["ts_iso", "symbol", "signal", "strategy", "price"]})

                    upsert_heartbeat(
                        con,
                        service="blofin-ingestor",
                        ts_ms=ts,
                        ts_iso=ts_iso,
                        details_json=json.dumps({"symbols": len(SYMBOLS)}),
                    )
                    
                    # Periodic maintenance for plugin system
                    if USE_PLUGIN_SYSTEM and strategy_manager:
                        # Update performance scores every 5 minutes
                        if ts - last_score_update >= SCORE_UPDATE_INTERVAL_MS:
                            try:
                                count = knowledge_base.update_all_scores(con)
                                print(f"[ingestor] Updated {count} performance scores")
                                last_score_update = ts
                            except Exception as e:
                                print(f"[ingestor] Error updating scores: {e}")
                        
                        # Auto-manage strategies every hour
                        if ts - last_auto_manage >= AUTO_MANAGE_INTERVAL_MS:
                            try:
                                changes = knowledge_base.auto_manage_strategies(con, strategy_manager)
                                if changes['disabled'] or changes['enabled']:
                                    print(f"[ingestor] Auto-manage: disabled={changes['disabled']}, enabled={changes['enabled']}")
                                last_auto_manage = ts
                            except Exception as e:
                                print(f"[ingestor] Error in auto-manage: {e}")
                    
                    con.commit()
        except Exception as e:
            print(f"[reconnect] {e}")
            await asyncio.sleep(backoff + random.random())
            backoff = min(RECONNECT_MAX, max(RECONNECT_MIN, backoff * 2))


if __name__ == "__main__":
    asyncio.run(run())
