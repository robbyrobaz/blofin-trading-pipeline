# Blofin Data Coverage

## How Backfill Works

**Blofin's REST API supports full 7-day historical data** via proper backward pagination using the `after` parameter.

### API Details
- **Endpoint**: `GET https://openapi.blofin.com/api/v1/market/candles`
- **Params**: `instId`, `bar` (timeframe), `limit` (max 300), `after` (pagination cursor)
- **Pagination**: `after=<oldest_ts_ms>` returns candles OLDER than that timestamp (OKX-style)
- **Order**: Returns candles in DESCENDING order (newest first)
- **Rate**: ~300 candles/request, 34 requests per symbol for 7 days of 1m data

### Important: CCXT Limitation
CCXT's `fetch_ohlcv` does NOT correctly paginate Blofin. It ignores the `since` parameter (only uses it for client-side filtering) and caps at 100 candles per request. **The backfill script uses direct REST API calls** to bypass this limitation.

If you need to use CCXT, pass `params={'after': str(timestamp_ms)}` directly — but the batch limit is still 100 vs 300 with the direct API.

## Coverage Strategy

### 1. Live Websocket Feed (Primary)
- `blofin-stack-ingestor.service` — real-time tick ingestion
- Batched symbol subscriptions (8 symbols/batch)
- Continuous 1-minute resolution data

### 2. Historical Backfill (Secondary)
- `blofin-stack-gapfill.timer` — runs every 30 minutes
- `historical_backfill.py` — direct Blofin REST API with backward pagination
- **7-day lookback** (168 hours, ~10,080 expected 1m candles per symbol)
- Skips symbols already at ≥99.5% coverage
- ~10-15 seconds per symbol needing full backfill

### 3. Expected Coverage
- **100% for all configured symbols** within the 7-day window
- Fresh symbols reach 100% on first backfill run (~34 API requests)
- Subsequent runs skip already-covered symbols and only fill new gaps

## Files
- `ingestor.py` — Websocket → SQLite (primary, real-time)
- `historical_backfill.py` — REST gap filler (direct API, 7-day reach)
- `api_server.py` — Dashboard with coverage metrics
- `db.py` — SQLite schema
