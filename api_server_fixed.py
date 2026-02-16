#!/usr/bin/env python3
"""
Blofin API with proper caching + stale-while-revalidate pattern.

Best practices implemented:
1. Serve cached data immediately (always responsive)
2. Fetch fresh data in background
3. Indicate data freshness with timestamps
4. Result caching with conditional updates
"""

import json
import os
import time
import threading
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

# Stale-while-revalidate cache
_cache_lock = threading.Lock()
_cache_data = {'summary': None, 'summary_ts': 0}

def _in_clause(symbols):
    """Build WHERE clause for symbol filter."""
    if not symbols:
        return '', []
    ph = ','.join('?' for _ in symbols)
    return f' WHERE symbol IN ({ph}) ', list(symbols)

def _grade_score(score: float) -> str:
    if score >= 85: return 'A'
    if score >= 70: return 'B'
    if score >= 55: return 'C'
    if score >= 40: return 'D'
    return 'F'

def fetch_summary_data():
    """Fetch summary data (may be slow, runs in background)."""
    try:
        wh, args = _in_clause(SYMBOLS)
        
        # Fetch signals
        sigs = [dict(r) for r in con.execute(
            f'SELECT ts_iso,symbol,signal,strategy,confidence,price FROM signals {wh} ORDER BY ts_ms DESC LIMIT 500',
            args
        )]
        
        # Fetch confirmed
        confirmed = [dict(r) for r in con.execute(
            f'SELECT ts_iso,symbol,signal,score,rationale FROM confirmed_signals {wh} ORDER BY ts_ms DESC LIMIT 50',
            args
        )]
        
        # Fetch paper trades
        paper = [dict(r) for r in con.execute(
            f'SELECT opened_ts_iso,closed_ts_iso,symbol,side,entry_price,exit_price,status,pnl_pct FROM paper_trades {wh} ORDER BY id DESC LIMIT 100',
            args
        )]
        
        # Counters
        by_sig = Counter(s['signal'] for s in sigs)
        by_strat = Counter(s['strategy'] for s in sigs)
        
        # Live status
        now_ms = int(time.time() * 1000)
        tick_flow = dict(con.execute(
            f'SELECT SUM(CASE WHEN ts_ms >= ? THEN 1 ELSE 0 END) AS ticks_10s, MAX(ts_ms) AS last_tick_ms FROM ticks {wh}',
            [now_ms - 10_000, *args]
        ).fetchone())
        
        last_tick_ms = tick_flow.get('last_tick_ms')
        seconds_since_last = round(max(0.0, (now_ms - last_tick_ms) / 1000.0), 1) if last_tick_ms else None
        is_live = bool((tick_flow.get('ticks_10s') or 0) > 5 and seconds_since_last is not None and seconds_since_last <= 12)
        
        # Paper stats
        closed = [p for p in paper if p['status'] == 'CLOSED']
        if closed:
            wins = sum(1 for p in closed if p['pnl_pct'] and p['pnl_pct'] > 0)
            win_rate = (wins / len(closed) * 100) if closed else 0
            avg_pnl = sum(p['pnl_pct'] for p in closed if p['pnl_pct']) / len(closed) if closed else 0
        else:
            win_rate = 0
            avg_pnl = 0
        
        # Strategy scores (simplified - by side since paper_trades doesn't have strategy column)
        strategy_scores = []
        for side in ['BUY', 'SELL']:
            closed_trades = [p for p in paper if p['status'] == 'CLOSED' and p['side'] == side]
            if closed_trades:
                wins = sum(1 for p in closed_trades if p['pnl_pct'] and p['pnl_pct'] > 0)
                wr = (wins / len(closed_trades) * 100) if closed_trades else 0
                avg_pnl = sum(p['pnl_pct'] for p in closed_trades if p['pnl_pct']) / len(closed_trades) if closed_trades else 0
                total_pnl = sum(p['pnl_pct'] for p in closed_trades if p['pnl_pct'])
                
                pnl_component = max(0.0, min(100.0, ((avg_pnl + 2.0) / 4.0) * 100.0))
                score = (wr * 0.6) + (pnl_component * 0.4)
                
                strategy_scores.append({
                    'strategy': side,
                    'pattern': side,
                    'closed_count': len(closed_trades),
                    'win_rate_pct': round(wr, 2),
                    'avg_pnl_pct': round(avg_pnl, 4),
                    'total_pnl_pct': round(total_pnl, 4),
                    'score': round(score, 2),
                    'grade': _grade_score(score),
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
            'fetched_at_ms': int(time.time() * 1000),
        }
    except Exception as e:
        print(f'Error fetching data: {e}')
        import traceback
        traceback.print_exc()
        return None

def timeseries(symbol: str, limit: int = 300):
    rows = con.execute(
        'SELECT ts_iso, price FROM ticks WHERE symbol=? ORDER BY ts_ms DESC LIMIT ?',
        (symbol, min(limit, 1000))
    ).fetchall()
    out = [dict(r) for r in rows]
    out.reverse()
    return out

def background_update():
    """Periodically fetch fresh data in background."""
    global _cache_data
    while True:
        time.sleep(3)  # Update every 3 seconds
        data = fetch_summary_data()
        if data:
            with _cache_lock:
                _cache_data['summary'] = data
                _cache_data['summary_ts'] = int(time.time() * 1000)

class H(BaseHTTPRequestHandler):
    def sendb(self, b: bytes, code=200, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, format, *args):
        pass  # Silence logs

    def do_GET(self):
        p = urlparse(self.path)
        q = parse_qs(p.query)

        if p.path == '/healthz':
            return self.sendb(b'ok', ctype='text/plain')
        
        if p.path == '/api/summary':
            # Return cached data immediately (stale-while-revalidate)
            with _cache_lock:
                data = _cache_data['summary'] or {'error': 'warming up...'}
            result = json.dumps(data, default=str).encode()
            return self.sendb(result)
        
        if p.path == '/api/timeseries':
            symbol = q.get('symbol', [SYMBOLS[0] if SYMBOLS else 'PEPE-USDT'])[0]
            limit = int(q.get('limit', ['300'])[0])
            result = json.dumps(timeseries(symbol, max(30, min(limit, 1000))), default=str).encode()
            return self.sendb(result)
        
        if p.path == '/':
            # Minimal HTML that auto-refreshes and shows staleness
            html = '''<!doctype html>
<html><head><meta charset="utf-8"><title>Blofin Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0a0e1a;color:#e7ecff;font-family:system-ui;padding:20px}
.wrap{max-width:1200px;margin:0 auto}
h1{font-size:28px;margin:0 0 8px;color:#60a5fa}
.freshness{font-size:11px;color:#94a3b8;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:16px 0}
.card{background:#1e293b;border:1px solid #334155;padding:14px;border-radius:6px}
.card .label{font-size:10px;color:#94a3b8;text-transform:uppercase;margin-bottom:4px}
.card .value{font-size:18px;font-weight:700}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:12px}
th{background:#0f172a;padding:8px;text-align:left;border-bottom:1px solid #334155;font-weight:600}
td{padding:8px;border-bottom:1px solid #334155}
.section{background:#1e293b;border:1px solid #334155;padding:16px;margin:12px 0;border-radius:6px}
.section h2{margin:0 0 12px;font-size:14px}
.positive{color:#10b981}.negative{color:#ef4444}
select{background:#0f172a;color:#e7ecff;border:1px solid #334155;padding:6px;border-radius:4px;font-size:12px}
#status{font-weight:700}
#loading{text-align:center;padding:20px;color:#94a3b8}
</style></head><body>
<div class="wrap">
<h1>Blofin 24/7</h1>
<div class="freshness">Last updated: <span id="age">—</span></div>
<div id="loading">Warming up dashboard...</div>
<div id="content" style="display:none">
<div class="grid">
<div class="card"><div class="label">Status</div><div class="value" id="status">—</div></div>
<div class="card"><div class="label">Signals</div><div class="value" id="signals">—</div></div>
<div class="card"><div class="label">Confirmed</div><div class="value" id="confirmed">—</div></div>
<div class="card"><div class="label">Win Rate</div><div class="value" id="wr">—</div></div>
</div>
<div class="section">
<h2>Top Strategies</h2>
<table><thead><tr><th>Strategy</th><th>Win%</th><th>Avg PnL%</th><th>Total%</th><th>Grade</th></tr></thead>
<tbody id="strats"></tbody></table>
</div>
<div class="section">
<h2>Price Chart</h2>
<label>Symbol: <select id="sym"></select></label>
<canvas id="chart" height="50"></canvas>
</div>
<div class="section">
<h2>Confirmed Signals</h2>
<table><thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>Score</th></tr></thead>
<tbody id="conf"></tbody></table>
</div>
</div>
</div>
<script>
let ch=null;
function formatAge(ms){
  const s=Math.round((Date.now()-ms)/1000);
  if(s<60)return s+'s ago';
  if(s<3600)return Math.round(s/60)+'m ago';
  return Math.round(s/3600)+'h ago';
}
async function render(){
  try{
    const r=await fetch('/api/summary');
    const d=await r.json();
    if(d.error){document.getElementById('loading').textContent=d.error; return;}
    document.getElementById('loading').style.display='none';
    document.getElementById('content').style.display='block';
    document.getElementById('age').textContent=formatAge(d.fetched_at_ms);
    document.getElementById('status').textContent=d.live_status.is_live?'✓ LIVE':'✗ STALE';
    document.getElementById('status').style.color=d.live_status.is_live?'#10b981':'#ef4444';
    document.getElementById('signals').textContent=d.signals_count;
    document.getElementById('confirmed').textContent=d.confirmed_count;
    document.getElementById('wr').textContent=d.paper_stats.win_rate_pct+'%';
    
    let h='';
    for(let s of d.strategy_scores){
      const c=s.avg_pnl_pct>=0?'positive':'negative';
      h+='<tr><td>'+s.strategy+'</td><td>'+s.win_rate_pct+'%</td><td class="'+c+'">'+s.avg_pnl_pct.toFixed(3)+'%</td><td class="'+c+'">'+s.total_pnl_pct.toFixed(1)+'%</td><td>'+s.grade+'</td></tr>';
    }
    document.getElementById('strats').innerHTML=h;
    
    let opts='';
    for(let s of d.symbols_configured)opts+='<option>'+s+'</option>';
    const sel=document.getElementById('sym');
    sel.innerHTML=opts;
    sel.onchange=e=>loadChart(e.target.value);
    if(d.symbols_configured.length)loadChart(d.symbols_configured[0]);
    
    h='';
    for(let c of d.confirmed_signals){
      h+='<tr><td>'+c.ts_iso.slice(11,19)+'</td><td><b>'+c.symbol+'</b></td><td>'+c.signal+'</td><td>'+c.score+'</td></tr>';
    }
    document.getElementById('conf').innerHTML=h;
  }catch(e){
    document.getElementById('loading').textContent='Error: '+e.message;
  }
}
async function loadChart(sym){
  const r=await fetch('/api/timeseries?symbol='+encodeURIComponent(sym));
  const d=await r.json();
  const ls=d.map(x=>x.ts_iso.slice(11,19));
  const ps=d.map(x=>x.price);
  if(ch)ch.destroy();
  ch=new Chart(document.getElementById('chart'),{type:'line',data:{labels:ls,datasets:[{label:sym,data:ps,borderColor:'#8b5cf6',backgroundColor:'rgba(139,92,246,0.1)',borderWidth:2,pointRadius:0,tension:0.1,fill:true}]},options:{responsive:true,plugins:{legend:{labels:{color:'#e2e8f0'}}}}});
}
render();
setInterval(render,2000);
</script>
</body></html>'''
            return self.sendb(html.encode('utf-8'), ctype='text/html; charset=utf-8')
        
        return self.sendb(b'{"error":"not found"}', code=404)

if __name__ == '__main__':
    # Start background update thread
    bg = threading.Thread(target=background_update, daemon=True)
    bg.start()
    
    # Initial warm-up
    print('Warming up cache...')
    data = fetch_summary_data()
    if data:
        with _cache_lock:
            _cache_data['summary'] = data
            _cache_data['summary_ts'] = int(time.time() * 1000)
    
    print(f'Blofin API: {HOST}:{PORT}')
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
