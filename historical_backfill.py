#!/usr/bin/env python3
import os
import time
import sqlite3
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from db import connect, init_db

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')

DB_PATH = os.getenv('BLOFIN_DB_PATH', str(ROOT / 'data' / 'blofin_monitor.db'))
SYMBOLS = [s.strip() for s in os.getenv('BLOFIN_SYMBOLS', 'BTC-USDT,ETH-USDT').split(',') if s.strip()]
TIMEFRAME = os.getenv('BACKFILL_TIMEFRAME', '1m')
LOOKBACK_HOURS = int(os.getenv('BACKFILL_LOOKBACK_HOURS', '168'))
BATCH_LIMIT = min(300, max(1, int(os.getenv('BACKFILL_BATCH_LIMIT', '300'))))
LOOP_SECONDS = int(os.getenv('BACKFILL_LOOP_SECONDS', '0'))
REQUEST_SLEEP_MS = max(0, int(os.getenv('BACKFILL_REQUEST_SLEEP_MS', '250')))
SYMBOL_SLEEP_MS = max(0, int(os.getenv('BACKFILL_SYMBOL_SLEEP_MS', '500')))

TF_MS = 60_000 if TIMEFRAME == '1m' else 60_000
BLOFIN_API_BASE = 'https://openapi.blofin.com/api/v1'


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def floor_tf(ms: int) -> int:
    return (ms // TF_MS) * TF_MS


def get_existing_minute_set(con: sqlite3.Connection, symbol: str, since_ms: int, until_ms: int) -> set[int]:
    """Get all minute timestamps that already exist in the DB for this symbol."""
    cur = con.execute(
        '''
        SELECT DISTINCT ((ts_ms / ?) * ?) AS minute_ts
        FROM ticks
        WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
        ''',
        (TF_MS, TF_MS, symbol, since_ms, until_ms),
    )
    return {int(r[0]) for r in cur.fetchall() if r[0] is not None}


def fetch_blofin_candles(inst_id: str, limit: int = 300, after: Optional[str] = None) -> list[dict]:
    """
    Fetch candles from Blofin REST API.
    
    Returns candles in DESCENDING order (newest first).
    Each candle: [ts_ms_str, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
    
    Args:
        inst_id: Blofin instrument ID (e.g., 'BTC-USDT')
        limit: Max candles to fetch (max 300)
        after: Pagination cursor - returns candles OLDER than this timestamp (ms)
    """
    url = f'{BLOFIN_API_BASE}/market/candles'
    params = {
        'instId': inst_id,
        'bar': TIMEFRAME,
        'limit': str(limit),
    }
    if after:
        params['after'] = str(after)
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get('code') != '0':
            print(f"    API error: {data.get('msg', 'unknown error')}")
            return []
        
        candles = data.get('data', [])
        return candles
    except Exception as e:
        print(f"    Request error: {e}")
        return []


def backfill_symbol(con: sqlite3.Connection, symbol: str, window_start_ms: int, window_end_ms: int, expected_points: int, now_ms: int) -> tuple[int, int]:
    """
    Backfill a single symbol using backward pagination.
    
    Returns (gaps_found, rows_inserted).
    """
    # Check existing coverage
    existing = get_existing_minute_set(con, symbol, window_start_ms, window_end_ms)
    coverage_before = len(existing) / expected_points if expected_points > 0 else 0.0
    
    # Skip if already well-covered
    if coverage_before >= 0.995:
        print(f'{symbol}: ✓ Already {coverage_before*100:.1f}% covered ({len(existing)}/{expected_points}), skipping')
        con.execute(
            'INSERT INTO gap_fill_runs(ts_ms, ts_iso, symbol, gaps_found, rows_inserted, note) VALUES(?,?,?,?,?,?)',
            (now_ms, iso(now_ms), symbol, 0, 0, f'already {coverage_before*100:.1f}% covered'),
        )
        con.commit()
        return 0, 0
    
    print(f'{symbol}: Coverage {coverage_before*100:.1f}% ({len(existing)}/{expected_points}), backfilling...')
    
    rows_inserted = 0
    total_fetched = 0
    batch_count = 0
    after_cursor = None  # Start with most recent candles
    oldest_fetched_ms = window_end_ms
    
    # Paginate backward until we reach window_start_ms
    while oldest_fetched_ms >= window_start_ms:
        batch_count += 1
        candles = fetch_blofin_candles(symbol, limit=BATCH_LIMIT, after=after_cursor)
        
        if not candles:
            print(f'  Batch {batch_count}: No more data (empty response)')
            break
        
        # Process candles (they come in descending order)
        batch_inserted = 0
        batch_oldest_ms = None
        batch_newest_ms = None
        
        for candle in candles:
            # Candle format: [ts_ms_str, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
            ts_ms = int(candle[0])
            close_price = float(candle[4])
            
            # Track batch boundaries
            if batch_newest_ms is None or ts_ms > batch_newest_ms:
                batch_newest_ms = ts_ms
            if batch_oldest_ms is None or ts_ms < batch_oldest_ms:
                batch_oldest_ms = ts_ms
            
            # Skip if outside window
            if ts_ms < window_start_ms or ts_ms > window_end_ms:
                continue
            
            # Skip if already exists
            if ts_ms in existing:
                continue
            
            # Insert new candle
            con.execute(
                'INSERT INTO ticks(ts_ms, ts_iso, symbol, price, source, raw_json) VALUES(?,?,?,?,?,?)',
                (ts_ms, iso(ts_ms), symbol, close_price, 'historical_fill', None),
            )
            existing.add(ts_ms)
            rows_inserted += 1
            batch_inserted += 1
        
        total_fetched += len(candles)
        
        if batch_oldest_ms:
            oldest_fetched_ms = batch_oldest_ms
            after_cursor = str(batch_oldest_ms)  # Next batch: get candles older than this
        
        print(f'  Batch {batch_count}: Fetched {len(candles)} candles ({iso(batch_newest_ms or 0)} to {iso(batch_oldest_ms or 0)}), inserted {batch_inserted}')
        
        # Stop if we've reached far enough back
        if batch_oldest_ms and batch_oldest_ms < window_start_ms:
            print(f'  Reached window start, stopping')
            break
        
        # Rate limiting
        if REQUEST_SLEEP_MS > 0:
            time.sleep(REQUEST_SLEEP_MS / 1000.0)
    
    # Calculate final coverage
    coverage_after = len(existing) / expected_points if expected_points > 0 else 0.0
    coverage_pct = round(coverage_after * 100.0, 2)
    
    # Log to gap_fill_runs
    note = (
        f'batches={batch_count} fetched={total_fetched} '
        f'points={len(existing)}/{expected_points} ({coverage_pct}%) '
        f'window={LOOKBACK_HOURS}h'
    )
    
    # Find first and last gap timestamps
    first_gap_ts = None
    last_gap_ts = None
    if rows_inserted > 0:
        # Get min/max timestamps of inserted rows
        cur = con.execute(
            'SELECT MIN(ts_ms), MAX(ts_ms) FROM ticks WHERE symbol=? AND source=? AND ts_ms>=? AND ts_ms<=?',
            (symbol, 'historical_fill', window_start_ms, window_end_ms),
        )
        row = cur.fetchone()
        if row and row[0]:
            first_gap_ts = row[0]
            last_gap_ts = row[1]
    
    con.execute(
        'INSERT INTO gap_fill_runs(ts_ms, ts_iso, symbol, gaps_found, rows_inserted, first_gap_ts_ms, last_gap_ts_ms, note) VALUES(?,?,?,?,?,?,?,?)',
        (now_ms, iso(now_ms), symbol, batch_count, rows_inserted, first_gap_ts, last_gap_ts, note),
    )
    con.commit()
    
    print(f'{symbol}: ✓ Inserted {rows_inserted} rows in {batch_count} batches, coverage now {coverage_pct}% ({len(existing)}/{expected_points})\n')
    
    return batch_count, rows_inserted


def compute_window_bounds(now_ms: int, lookback_hours: int) -> tuple[int, int]:
    """Return inclusive [start,end] bounds for fully closed candles only."""
    window_end_ms = floor_tf(now_ms - TF_MS)
    window_start_ms = floor_tf(window_end_ms - lookback_hours * 60 * 60 * 1000)
    return window_start_ms, window_end_ms


def run_once(con: sqlite3.Connection) -> tuple[int, int]:
    """Run one backfill cycle for all symbols."""
    now_ms = int(time.time() * 1000)
    window_start_ms, window_end_ms = compute_window_bounds(now_ms, LOOKBACK_HOURS)
    expected_points = int((window_end_ms - window_start_ms) // TF_MS) + 1

    total_rows = 0
    total_gaps = 0

    print(f'\n=== Historical Backfill (Direct Blofin API) ===')
    print(f'Window: {iso(window_start_ms)} to {iso(window_end_ms)}')
    print(f'Lookback: {LOOKBACK_HOURS} hours ({expected_points} expected 1m candles)')
    print(f'Batch size: {BATCH_LIMIT}, Sleep: {REQUEST_SLEEP_MS}ms between requests, {SYMBOL_SLEEP_MS}ms between symbols')
    print(f'Symbols: {len(SYMBOLS)}\n')

    for sym in SYMBOLS:
        try:
            gaps, rows = backfill_symbol(con, sym, window_start_ms, window_end_ms, expected_points, now_ms)
            total_gaps += gaps
            total_rows += rows
        except Exception as e:
            print(f'{sym}: ✗ Error: {e}\n')
            con.execute(
                'INSERT INTO gap_fill_runs(ts_ms, ts_iso, symbol, gaps_found, rows_inserted, note) VALUES(?,?,?,?,?,?)',
                (now_ms, iso(now_ms), sym, 0, 0, f'error: {type(e).__name__}: {e}'),
            )
            con.commit()
        finally:
            if SYMBOL_SLEEP_MS > 0:
                time.sleep(SYMBOL_SLEEP_MS / 1000.0)

    print(f'\n=== Summary ===')
    print(f'Total API batches: {total_gaps}')
    print(f'Total rows inserted: {total_rows}')
    return total_gaps, total_rows


def main():
    con = connect(DB_PATH)
    init_db(con)
    
    print('Blofin Historical Backfill - Direct REST API')
    print(f'DB: {DB_PATH}')
    print(f'API: {BLOFIN_API_BASE}')

    if LOOP_SECONDS > 0:
        print(f'Loop mode: running every {LOOP_SECONDS} seconds\n')
        while True:
            run_once(con)
            time.sleep(max(60, LOOP_SECONDS))
    else:
        run_once(con)


if __name__ == '__main__':
    main()
