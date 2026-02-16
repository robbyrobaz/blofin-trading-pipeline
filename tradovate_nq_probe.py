#!/usr/bin/env python3
"""Probe Tradovate market-data stream for NQ/MNQ quotes.

Usage:
  source .venv/bin/activate
  export TRADOVATE_USERNAME=...
  export TRADOVATE_PASSWORD=...
  python tradovate_nq_probe.py --symbol NQH6 --max-messages 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets

from tradovate_client import TradovateClient, TradovateConfig, TradovateError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream Tradovate NQ/MNQ quotes")
    p.add_argument("--symbol", default="NQ", help="Symbol/root or contract code")
    p.add_argument("--max-messages", type=int, default=25, help="Stop after N messages")
    p.add_argument(
        "--out",
        default="data/tradovate_nq_probe.jsonl",
        help="Output JSONL path",
    )
    return p.parse_args()


async def run_probe(symbol: str, max_messages: int, out_path: Path) -> int:
    cfg = TradovateConfig.from_env()
    client = TradovateClient(cfg)
    token = client.access_token()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0

    async with websockets.connect(cfg.md_ws_url, ping_interval=20, ping_timeout=20) as ws:
        cmd = client.md_subscribe_quote_command(symbol)
        cmd["token"] = token
        await ws.send(json.dumps(cmd))

        started = time.time()
        with out_path.open("a", encoding="utf-8") as f:
            while wrote < max_messages:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                now = time.time()
                row = {
                    "ts_local": now,
                    "elapsed_s": round(now - started, 3),
                    "symbol": symbol,
                    "raw": raw,
                }
                f.write(json.dumps(row) + "\n")
                wrote += 1

    return wrote


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    try:
        wrote = asyncio.run(run_probe(args.symbol, args.max_messages, out_path))
    except (TradovateError, TimeoutError, OSError, asyncio.TimeoutError) as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Wrote {wrote} quote messages to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
