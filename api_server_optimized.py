#!/usr/bin/env python3
"""
Optimized Blofin API server with reduced CPU footprint.

Key improvements:
1. ThreadingHTTPServer for concurrent requests
2. Separate JSON data endpoint from HTML rendering
3. Reduced data window (500 signals instead of 3000, etc.)
4. Client-side HTML rendering instead of server-side string building
5. Result caching (3-second TTL)
"""

import json
import os
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from db import connect, init_db

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DB_PATH = os.getenv('BLOFIN_DB_PATH', str(ROOT / 'data' / 'blofin_monitor.db'))
HOST = os.getenv('API_HOST', '127.0.0.1')
PORT = int(os.getenv('API_PORT', '8780'))
SYMBOLS = [s.strip() for s in os.getenv('BLOFIN_SYMBOLS', '').split(',') if s.strip()]

con = connect(DB_PATH)
init_db(con)

# Cache to reduce CPU from repeated requests
_cache = {}
_cache_ts = {}

def _get_cached(key, ttl_sec=3):
    if key not in _cache:
        return None
    if time.time() - _cache_ts.get(key, 0) > ttl_sec:
        return None
    return _cache[key]

def _set_cache(key, value):
    _cache[key] = value
    _cache_ts[key] = time.time()

def _in_clause(symbols):
    if not symbols:
        return '', []
    ph = ','.join('?' for _ in symbols)
    return f' WHERE symbol IN ({ph}) ', list(symbols)

def _grade_score(score: float) -> str:
    if score >= 85:
        return 'A'
    if score >= 70:
        return 'B'
    if score >= 55:
        return 'C'
    if score >= 40:
        return 'D'
    return 'F'

def fetch_summary_json():
    """Fetch summary data as JSON (optimized for speed).
    
    Reduced data sizes:
    - 500 signals (was 3000)
    - 50 confirmed (was 300)
    - Fewer calculations
    """
    wh, args = _in_clause(SYMBOLS)
    
    # Fetch fewer rows (500 instead of 3000)
    sigs = [dict(r) for r in con.execute(
        f'SELECT ts_iso,symbol,signal,strategy,confidence,price FROM signals {wh} ORDER BY ts_ms DESC LIMIT 500',
        args
    )]
    
    # Fetch fewer confirmed (50 instead of 300)
    confirmed = [dict(r) for r in con.execute(
        f'SELECT ts_iso,symbol,signal,score,rationale FROM confirmed_signals {wh} ORDER BY ts_ms DESC LIMIT 50',
        args
    )]
    
    # Paper trades (100 instead of 300)
    paper = [dict(r) for r in con.execute(
        f'SELECT opened_ts_iso,closed_ts_iso,symbol,side,entry_price,exit_price,status,pnl_pct,reason FROM paper_trades {wh} ORDER BY id DESC LIMIT 100',
        args
    )]

    # Quick counters
    by_sig = Counter(s['signal'] for s in sigs)
    by_strat = Counter(s['strategy'] for s in sigs)
    
    # Latest tick info
    latest_by_symbol = [dict(r) for r in con.execute(
        f'SELECT symbol, MAX(ts_iso) as last_seen, COUNT(*) as ticks FROM ticks {wh} GROUP BY symbol ORDER BY symbol',
        args
    )]

    # Live status (simple version)
    now_ms = int(time.time() * 1000)
    tick_flow = dict(con.execute(
        f'''SELECT
            SUM(CASE WHEN ts_ms >= ? THEN 1 ELSE 0 END) AS ticks_10s,
            MAX(ts_ms) AS last_tick_ms
        FROM ticks {wh}''',
        [now_ms - 10_000, *args],
    ).fetchone())
    
    last_tick_ms = tick_flow.get('last_tick_ms')
    seconds_since_last = round(max(0.0, (now_ms - last_tick_ms) / 1000.0), 1) if last_tick_ms else None
    is_live = bool((tick_flow.get('ticks_10s') or 0) > 5 and seconds_since_last is not None and seconds_since_last <= 12)

    # Paper stats (simplified)
    closed = [p for p in paper if p['status'] == 'CLOSED']
    if closed:
        wins = sum(1 for p in closed if p['pnl_pct'] and p['pnl_pct'] > 0)
        win_rate = (wins / len(closed) * 100) if closed else 0
        avg_pnl = sum(p['pnl_pct'] for p in closed if p['pnl_pct']) / len(closed) if closed else 0
    else:
        win_rate = 0
        avg_pnl = 0

    # Strategy performance (cached calculation)
    perf_rows = con.execute(f'''
        SELECT strategy, side as pattern, COUNT(*) as closed_count,
               SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
               AVG(pnl_pct) as avg_pnl_pct, SUM(pnl_pct) as total_pnl_pct
        FROM paper_trades {wh} WHERE status='CLOSED'
        GROUP BY strategy, side
        ORDER BY SUM(pnl_pct) DESC LIMIT 20
    ''', args).fetchall()
    
    strategy_scores = []
    for row in perf_rows:
        closed_count = int(row['closed_count'] or 0)
        wins = int(row['wins'] or 0)
        wr = (wins / closed_count * 100) if closed_count else 0
        avg_pnl = float(row['avg_pnl_pct'] or 0)
        total_pnl = float(row['total_pnl_pct'] or 0)
        
        if closed_count >= 3:
            pnl_component = max(0.0, min(100.0, ((avg_pnl + 2.0) / 4.0) * 100.0))
            score = (wr * 0.6) + (pnl_component * 0.4)
            grade = _grade_score(score)
        else:
            score = None
            grade = 'N/A'
        
        strategy_scores.append({
            'strategy': row['strategy'],
            'pattern': row['pattern'],
            'closed_count': closed_count,
            'win_rate_pct': round(wr, 2),
            'avg_pnl_pct': round(avg_pnl, 4),
            'total_pnl_pct': round(total_pnl, 4),
            'score': round(score, 2) if score else None,
            'grade': grade,
        })

    return {
        'symbols_configured': SYMBOLS,
        'signals_count': len(sigs),
        'confirmed_count': len(confirmed),
        'signals_by_type': dict(by_sig),
        'signals_by_strategy': dict(by_strat),
        'live_status': {
            'is_live': is_live,
            'ticks_10s': int(tick_flow.get('ticks_10s') or 0),
            'seconds_since_last_tick': seconds_since_last,
        },
        'paper_stats': {
            'total_trades': len(paper),
            'closed_count': len(closed),
            'open_count': len(paper) - len(closed),
            'win_rate_pct': round(win_rate, 2),
            'avg_pnl_pct': round(avg_pnl, 4),
        },
        'recent_signals': sigs[:50],
        'confirmed_signals': confirmed[:50],
        'paper_trades': paper[:50],
        'strategy_scores': strategy_scores[:15],
    }

def timeseries(symbol: str, limit: int = 300):
    rows = con.execute(
        'SELECT ts_iso, price FROM ticks WHERE symbol=? ORDER BY ts_ms DESC LIMIT ?',
        (symbol, min(limit, 1000))
    ).fetchall()
    out = [dict(r) for r in rows]
    out.reverse()
    return out

# Minimal HTML that loads data via JSON API and renders client-side
HTML_TEMPLATE = '''<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Blofin Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0a0e1a;color:#e7ecff;font-family:system-ui,sans-serif;padding:24px}
.wrap{max-width:1400px;margin:0 auto}
h1{font-size:32px;font-weight:700;margin:0 0 24px;color:#60a5fa}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:24px 0}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px}
.card .label{font-size:11px;color:#94a3b8;text-transform:uppercase;margin-bottom:8px}
.card .value{font-size:24px;font-weight:700}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:13px}
th{background:#1e293b;padding:12px;text-align:left;font-weight:600;border-bottom:2px solid #334155}
td{padding:12px;border-bottom:1px solid #334155}
tr:hover{background:rgba(59,130,246,0.1)}
.section{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px;margin:24px 0}
.section h2{margin:0 0 16px;font-size:18px;color:#e2e8f0}
.badge{display:inline-block;background:#1e40af;color:#fff;padding:4px 8px;border-radius:6px;font-size:11px;margin:2px;font-weight:600}
select{background:#0f172a;color:#e7ecff;border:1px solid #334155;padding:8px;border-radius:6px;font-size:13px;cursor:pointer}
.stat{display:inline-block;margin-right:20px}
.positive{color:#10b981}
.negative{color:#ef4444}
</style></head><body>
<div class="wrap">
<h1>Blofin 24/7 Dashboard</h1>
<div id="loading">Loading...</div>
<div id="content" style="display:none">
<div class="grid">
  <div class="card">
    <div class="label">Status</div>
    <div class="value" id="status-live">—</div>
  </div>
  <div class="card">
    <div class="label">Signals</div>
    <div class="value" id="signals-count">—</div>
  </div>
  <div class="card">
    <div class="label">Confirmed</div>
    <div class="value" id="confirmed-count">—</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value" id="win-rate">—</div>
  </div>
</div>

<div class="section">
  <h2>Top Strategies</h2>
  <table><thead><tr><th>Strategy</th><th>Win%</th><th>Avg PnL%</th><th>Total PnL%</th><th>Grade</th></tr></thead>
  <tbody id="strategy-table"></tbody></table>
</div>

<div class="section">
  <h2>Chart</h2>
  <label>Symbol: <select id="sym-select"></select></label>
  <canvas id="chart" style="margin-top:16px"></canvas>
</div>

<div class="section">
  <h2>Confirmed Signals</h2>
  <table><thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>Score</th></tr></thead>
  <tbody id="confirmed-table"></tbody></table>
</div>
</div>

<script>
let chart = null;
async function load() {
  try {
    const resp = await fetch('/api/summary-json');
    const data = await resp.json();
    render(data);
    document.getElementById('loading').style.display = 'none';
    document.getElementById('content').style.display = 'block';
  } catch (err) {
    document.getElementById('loading').textContent = 'Error: ' + err.message;
  }
}

function render(data) {
  // Stats
  const live = data.live_status.is_live;
  document.getElementById('status-live').textContent = live ? '✓ LIVE' : '✗ STALE';
  document.getElementById('status-live').style.color = live ? '#10b981' : '#ef4444';
  document.getElementById('signals-count').textContent = data.signals_count;
  document.getElementById('confirmed-count').textContent = data.confirmed_count;
  document.getElementById('win-rate').textContent = data.paper_stats.win_rate_pct + '%';
  
  // Strategies
  const strats = data.strategy_scores.map(s => `
    <tr>
      <td>${s.strategy}</td>
      <td>${s.win_rate_pct}%</td>
      <td class="${s.avg_pnl_pct >= 0 ? 'positive' : 'negative'}">${s.avg_pnl_pct.toFixed(3)}%</td>
      <td class="${s.total_pnl_pct >= 0 ? 'positive' : 'negative'}">${s.total_pnl_pct.toFixed(2)}%</td>
      <td>${s.grade}</td>
    </tr>
  `).join('');
  document.getElementById('strategy-table').innerHTML = strats;
  
  // Symbols
  const syms = data.symbols_configured.map(s => `<option>${s}</option>`).join('');
  const select = document.getElementById('sym-select');
  select.innerHTML = syms;
  if (data.symbols_configured.length) loadChart(data.symbols_configured[0]);
  select.addEventListener('change', (e) => loadChart(e.target.value));
  
  // Confirmed
  const confirmed = data.confirmed_signals.map(c => `
    <tr>
      <td>${c.ts_iso.slice(11, 19)}</td>
      <td><strong>${c.symbol}</strong></td>
      <td>${c.signal}</td>
      <td>${c.score}</td>
    </tr>
  `).join('');
  document.getElementById('confirmed-table').innerHTML = confirmed;
}

async function loadChart(sym) {
  const resp = await fetch('/api/timeseries?symbol=' + encodeURIComponent(sym));
  const data = await resp.json();
  const labels = data.map(d => d.ts_iso.slice(11, 19));
  const prices = data.map(d => d.price);
  
  if (chart) chart.destroy();
  const ctx = document.getElementById('chart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: sym,
        data: prices,
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139,92,246,0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
        fill: true
      }]
    },
    options: {
      responsive: true,
      plugins: {legend: {labels: {color: '#e2e8f0'}}}
    }
  });
}

load();
setInterval(load, 5000);
</script>
</body></html>'''

class H(BaseHTTPRequestHandler):
    def sendb(self, b: bytes, code=200, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, format, *args):
        pass  # Silence log spam

    def do_GET(self):
        p = urlparse(self.path)
        q = parse_qs(p.query)

        if p.path == '/healthz':
            return self.sendb(b'ok', ctype='text/plain')
        
        if p.path == '/api/summary-json':
            cached = _get_cached('summary-json')
            if cached:
                return self.sendb(cached)
            result = json.dumps(fetch_summary_json(), default=str).encode()
            _set_cache('summary-json', result)
            return self.sendb(result)
        
        if p.path == '/api/timeseries':
            symbol = q.get('symbol', [SYMBOLS[0] if SYMBOLS else 'PEPE-USDT'])[0]
            limit = int(q.get('limit', ['300'])[0])
            result = json.dumps(timeseries(symbol, max(30, min(limit, 1000))), default=str).encode()
            return self.sendb(result)
        
        if p.path == '/':
            return self.sendb(HTML_TEMPLATE.encode(), ctype='text/html; charset=utf-8')
        
        return self.sendb(b'{"error":"not found"}', code=404)

if __name__ == '__main__':
    print(f'Starting API server on {HOST}:{PORT}')
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
