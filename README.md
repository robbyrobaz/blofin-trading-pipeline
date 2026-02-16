# Blofin Monitoring Stack

Local-first monitoring stack for Blofin public market data:

- WebSocket ingestion for ~25 symbols
- Durable SQLite storage (`ticks`, `signals`, `service_heartbeats`)
- Pattern detectors (momentum + breakout + reversal)
- Buy/sell signal recording
- Local dashboard/API endpoints
- systemd user services for auto-start

## Quick start

```bash
cd /home/rob/.openclaw/workspace/blofin-stack
cp .env.example .env
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python ingestor.py
# in another terminal:
python api_server.py
```

## API

- `GET /healthz`
- `GET /api/summary`
- `GET /api/timeseries?symbol=BTC-USDT&limit=300`
- `GET /api/gap-fills`
- `GET /` (HTML dashboard)

## Futures data probe (Tradovate/NinjaTrader credentials)

A minimal probe script is included to test whether Tradovate credentials can stream NQ/MNQ quote data:

```bash
cd /home/rob/.openclaw/workspace/blofin-stack
source .venv/bin/activate
export TRADOVATE_USERNAME='...'
export TRADOVATE_PASSWORD='...'
# Optional overrides if using live environment instead of demo:
# export TRADOVATE_AUTH_URL='https://live.tradovateapi.com/v1/auth/accesstokenrequest'
# export TRADOVATE_API_BASE_URL='https://live.tradovateapi.com/v1'
# export TRADOVATE_MD_WS_URL='wss://md.tradovateapi.com/v1/websocket'
python tradovate_nq_probe.py --symbol NQ --max-messages 20
```

Output is written to `data/tradovate_nq_probe.jsonl`.
