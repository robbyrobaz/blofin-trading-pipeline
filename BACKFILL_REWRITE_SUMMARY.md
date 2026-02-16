# Historical Backfill Rewrite - Summary

**Date**: 2026-02-12  
**Task**: Rewrite historical_backfill.py to use Blofin REST API directly

## Critical Discovery

The previous assumption that **Blofin only returns 1-2 hours of historical data was WRONG**. 

The Blofin REST API actually provides **full 7-day (and potentially longer) historical OHLCV data**. The issue was that CCXT's `fetch_ohlcv()` method wasn't properly passing the pagination parameters to Blofin.

## Solution

Rewrote `historical_backfill.py` to:
- **Call Blofin REST API directly** using `requests` library
- **Use backward pagination** with the `after` parameter
- **Fetch 300 candles per request** (Blofin's max limit)
- **Walk backward from present** to 7 days ago
- **Insert only missing minutes** (efficient gap filling)

### API Details

**Endpoint**: `GET https://openapi.blofin.com/api/v1/market/candles`

**Key Parameters**:
- `instId`: Symbol (e.g., `BTC-USDT`)
- `bar`: Timeframe (`1m`)
- `limit`: Batch size (max 300)
- `after`: Pagination cursor (timestamp in ms) - returns candles **older than** this value

**Pagination Pattern**:
1. First request (no `after`): Get most recent 300 candles
2. Extract oldest timestamp from batch
3. Next request (`after=oldest_ts`): Get 300 candles older than that
4. Repeat until reaching 7-day lookback or empty response

## Results

### Before
- Symbols had ~20% coverage (only 1-2 hours of data)
- CCXT wasn't paginating correctly
- Gaps older than 2 hours couldn't be filled

### After
- Symbols reach **100% coverage** for 7-day window
- Successfully backfilled **8,000+ missing rows** per symbol
- ~34 API requests per symbol for full 7-day backfill
- Subsequent runs only fill small gaps (1-2 requests)

### Example Output

```
SHIB-USDT: Coverage 20.1% (2023/10081), backfilling...
  Batch 1: Fetched 300 candles (2026-02-13T00:19 to 2026-02-12T19:20), inserted 1
  Batch 2: Fetched 300 candles (2026-02-12T19:19 to 2026-02-12T14:20), inserted 0
  ...
  Batch 34: Fetched 300 candles (2026-02-06T03:19 to 2026-02-05T22:20), inserted 181
  Reached window start, stopping
SHIB-USDT: ✓ Inserted 8058 rows in 34 batches, coverage now 100.0% (10081/10081)
```

## Configuration Changes

Updated `.env` and `.env.example`:

```bash
BACKFILL_LOOKBACK_HOURS=168          # 7 days
BACKFILL_BATCH_LIMIT=300             # Blofin API max
BACKFILL_REQUEST_SLEEP_MS=250        # Rate limiting
BACKFILL_SYMBOL_SLEEP_MS=500         # Between symbols
```

**Removed**: `BACKFILL_MAX_GAP_MINUTES` (no longer needed with proper pagination)

## Documentation Updates

Updated `DATA_COVERAGE.md` to:
- Remove incorrect claims about Blofin's 1-2 hour limitation
- Document the correct API pagination approach
- Explain why CCXT failed (didn't pass `after` parameter correctly)
- Update coverage expectations (99-100% for 7-day window)

## Files Changed

1. **historical_backfill.py** - Complete rewrite
   - Removed CCXT dependency for Blofin
   - Added `fetch_blofin_candles()` function
   - Implemented backward pagination logic
   - Simplified gap filling (no chunking needed)

2. **DATA_COVERAGE.md** - Major update
   - Corrected historical data availability info
   - Documented direct API approach
   - Updated coverage expectations

3. **.env.example** - Parameter updates
   - Changed `BACKFILL_LOOKBACK_DAYS` → `BACKFILL_LOOKBACK_HOURS`
   - Updated batch limit: 60 → 300
   - Updated sleep timings
   - Added inline comments

## Verification

Tested successfully:
- ✅ Symbols with 100% coverage are skipped
- ✅ Symbols with low coverage (20%) backfill to 100%
- ✅ Backward pagination works correctly
- ✅ Missing minutes are identified and filled
- ✅ API rate limiting respected
- ✅ Progress logging works
- ✅ Database logging (gap_fill_runs table) works

## Known Issues

- Database lock errors when ingestor/API server are active
  - **Mitigation**: Already using WAL mode
  - **Future**: Add retry logic or run during quiet periods

## Deployment

Changes committed and pushed to `second-brain` repo:
- Commit: `94f22fa` - ".env.example updates"
- Previous: `e1537c5` - "auto code sync" (historical_backfill.py + DATA_COVERAGE.md)

The systemd timer service should continue working without changes:
- `blofin-stack-gapfill.timer` runs every 30 minutes
- Now fills gaps using direct Blofin API instead of external exchanges
- No external dependencies (OKX/Bybit) needed

## Performance

**Full backfill** (7 days, 32 symbols):
- ~34 requests × 32 symbols = ~1,088 API calls
- @ 250ms/request + 500ms/symbol = ~5-6 minutes total
- One-time cost for new symbols

**Maintenance mode** (subsequent runs):
- Most symbols: 0-1 requests (already covered)
- Recent gaps only: 1-2 requests
- Total: <1 minute

## Conclusion

**The Blofin REST API has been providing full 7-day historical data all along.** The limitation was in how we were calling it (via CCXT). By using the API directly with proper backward pagination, we now have:

- ✅ Full 7-day backfill capability
- ✅ No external exchange dependencies
- ✅ Efficient gap filling
- ✅ 100% coverage achievable for all symbols

The monitoring stack is now much more robust with complete historical data coverage.
