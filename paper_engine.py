#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from db import connect, init_db, upsert_heartbeat

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')

DB_PATH = os.getenv('BLOFIN_DB_PATH', str(ROOT / 'data' / 'blofin_monitor.db'))
CONFIRM_WINDOW_MIN = int(os.getenv('CONFIRM_WINDOW_MINUTES', '30'))
CONFIRM_MIN_STRATEGIES = int(os.getenv('CONFIRM_MIN_STRATEGIES', '2'))
PAPER_TP_PCT = float(os.getenv('PAPER_TP_PCT', '1.5'))
PAPER_SL_PCT = float(os.getenv('PAPER_SL_PCT', '1.0'))
PAPER_MAX_HOLD_MIN = int(os.getenv('PAPER_MAX_HOLD_MINUTES', '180'))
LOOP_SECONDS = int(os.getenv('PAPER_LOOP_SECONDS', '15'))

# Trading friction model (percent values, e.g., 0.04 = 0.04%)
# Defaults include small crypto-style fees + slippage for more realistic paper PnL.
PAPER_FEE_PCT_PER_SIDE = float(os.getenv('PAPER_FEE_PCT_PER_SIDE', '0.04'))
PAPER_SLIPPAGE_PCT_PER_SIDE = float(os.getenv('PAPER_SLIPPAGE_PCT_PER_SIDE', '0.02'))


def now_ms():
    return int(time.time() * 1000)


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def maybe_confirm_signals(con):
    t = now_ms()
    cutoff = t - CONFIRM_WINDOW_MIN * 60_000
    rows = con.execute(
        'SELECT id,ts_ms,ts_iso,symbol,signal,strategy,confidence,price FROM signals WHERE ts_ms >= ? ORDER BY ts_ms DESC',
        (cutoff,),
    ).fetchall()

    grouped = {}
    for r in rows:
        key = (r['symbol'], r['signal'])
        grouped.setdefault(key, []).append(r)

    inserted = 0
    for (symbol, side), arr in grouped.items():
        strategies = {a['strategy'] for a in arr}
        if len(strategies) < CONFIRM_MIN_STRATEGIES:
            continue
        top = arr[0]
        exists = con.execute('SELECT 1 FROM confirmed_signals WHERE signal_id=?', (top['id'],)).fetchone()
        if exists:
            continue
        score = min(0.99, 0.5 + (len(strategies) * 0.15))
        con.execute(
            'INSERT INTO confirmed_signals(signal_id,ts_ms,ts_iso,symbol,signal,score,rationale) VALUES(?,?,?,?,?,?,?)',
            (top['id'], top['ts_ms'], top['ts_iso'], symbol, side, score, f'{len(strategies)} strategies agreed in {CONFIRM_WINDOW_MIN}m'),
        )
        inserted += 1
    return inserted


def latest_price(con, symbol):
    r = con.execute('SELECT price FROM ticks WHERE symbol=? ORDER BY ts_ms DESC LIMIT 1', (symbol,)).fetchone()
    return float(r['price']) if r else None


def open_paper_trades(con):
    c = con.execute(
        "SELECT id,ts_ms,ts_iso,symbol,signal,score,rationale FROM confirmed_signals WHERE id NOT IN (SELECT confirmed_signal_id FROM paper_trades)")
    rows = c.fetchall()
    opened = 0
    for r in rows:
        px = latest_price(con, r['symbol'])
        if px is None:
            continue
        entry_reason = (r['rationale'] or '').strip() or 'No rationale provided'
        opened_ts = now_ms()
        con.execute(
            'INSERT INTO paper_trades(confirmed_signal_id,opened_ts_ms,opened_ts_iso,symbol,side,entry_price,qty,status,reason) VALUES(?,?,?,?,?,?,?,?,?)',
            (r['id'], opened_ts, iso(opened_ts), r['symbol'], r['signal'], px, 1.0, 'OPEN', f'ENTRY: {entry_reason}'),
        )
        opened += 1
    return opened


def calc_pnl_pct_with_friction(entry_price: float, mark_price: float, side: str) -> float:
    """Calculate PnL% with per-side fee + slippage applied at entry and exit."""
    fee = PAPER_FEE_PCT_PER_SIDE / 100.0
    slip = PAPER_SLIPPAGE_PCT_PER_SIDE / 100.0

    if side == 'SELL':
        # Short: worse fill on sell entry, worse fill on buy-to-cover exit.
        effective_entry = entry_price * (1.0 - slip - fee)
        effective_exit = mark_price * (1.0 + slip + fee)
        return ((effective_entry - effective_exit) / effective_entry) * 100.0

    # BUY long: worse fill on buy entry, worse fill on sell exit.
    effective_entry = entry_price * (1.0 + slip + fee)
    effective_exit = mark_price * (1.0 - slip - fee)
    return ((effective_exit - effective_entry) / effective_entry) * 100.0


def close_paper_trades(con):
    rows = con.execute('SELECT * FROM paper_trades WHERE status="OPEN"').fetchall()
    closed = 0
    t = now_ms()
    for r in rows:
        px = latest_price(con, r['symbol'])
        if px is None:
            continue
        entry = float(r['entry_price'])
        side = r['side']
        pnl_pct = calc_pnl_pct_with_friction(entry, px, side)

        # Keep OPEN trades marked-to-market with live unrealized PnL.
        con.execute('UPDATE paper_trades SET pnl_pct=? WHERE id=?', (pnl_pct, r['id']))

        age_min = (t - int(r['opened_ts_ms'])) / 60000.0
        reason = None
        if pnl_pct >= PAPER_TP_PCT:
            reason = 'TP'
        elif pnl_pct <= -PAPER_SL_PCT:
            reason = 'SL'
        elif age_min >= PAPER_MAX_HOLD_MIN:
            reason = 'TIME'

        if reason:
            existing_reason = (r['reason'] or '').strip()
            close_reason = f'EXIT: {reason}'
            if existing_reason:
                combined_reason = f'{existing_reason} | {close_reason}'
            else:
                combined_reason = close_reason
            con.execute(
                'UPDATE paper_trades SET status="CLOSED",closed_ts_ms=?,closed_ts_iso=?,exit_price=?,pnl_pct=?,reason=? WHERE id=?',
                (t, iso(t), px, pnl_pct, combined_reason, r['id']),
            )
            closed += 1
    return closed


def main():
    con = connect(DB_PATH)
    init_db(con)
    while True:
        confirmed = maybe_confirm_signals(con)
        opened = open_paper_trades(con)
        closed = close_paper_trades(con)
        t = now_ms()
        upsert_heartbeat(con, 'blofin-paper-engine', t, iso(t), json.dumps({'confirmed': confirmed, 'opened': opened, 'closed': closed}))
        con.commit()
        time.sleep(LOOP_SECONDS)


if __name__ == '__main__':
    main()
