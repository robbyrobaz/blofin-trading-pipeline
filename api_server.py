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

def calculate_stats(trades):
    """Calculate advanced trading stats: profit factor, Sortino ratio, max drawdown."""
    if not trades:
        return {'profit_factor': 0, 'sortino': 0, 'max_dd': 0, 'return_pct': 0}

    pnls = [t.get('pnl_pct', 0) for t in trades if t.get('pnl_pct')]
    if not pnls:
        return {'profit_factor': 0, 'sortino': 0, 'max_dd': 0, 'return_pct': 0}

    # Profit factor: gross profit / gross loss (avoid div by zero)
    gains = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    profit_factor = (gains / losses) if losses > 0 else gains

    # Sortino ratio (downside deviation focus)
    mean_pnl = sum(pnls) / len(pnls)
    downside = [p for p in pnls if p < mean_pnl]
    downside_std = (sum((p - mean_pnl) ** 2 for p in downside) / len(downside)) ** 0.5 if downside else 0
    sortino = (mean_pnl / downside_std * (252 ** 0.5)) if downside_std > 0 else 0

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    return {
        'profit_factor': round(profit_factor, 2),
        'sortino': round(sortino, 2),
        'max_dd': round(max_dd, 2),
        'return_pct': round(sum(pnls), 2),
    }

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
        
        # Strategy scores (by signal type from recent signals)
        strategy_scores = []
        strategy_data = {}
        for sig in sigs:
            strat = sig.get('strategy', 'unknown')
            if strat not in strategy_data:
                strategy_data[strat] = {'count': 0, 'by_signal': {'BUY': 0, 'SELL': 0}}
            strategy_data[strat]['count'] += 1
            signal_type = sig.get('signal', 'UNKNOWN')
            if signal_type in strategy_data[strat]['by_signal']:
                strategy_data[strat]['by_signal'][signal_type] += 1
        
        # Fetch per-strategy closed trades (single query, grouped in Python)
        strategy_trades_map = {}
        for row in con.execute('''
            SELECT s.strategy, pt.pnl_pct
            FROM paper_trades pt
            JOIN confirmed_signals cs ON pt.confirmed_signal_id = cs.id
            JOIN signals s ON cs.signal_id = s.id
            WHERE pt.status = 'CLOSED'
        '''):
            row_dict = dict(row)
            strat = row_dict.get('strategy', 'unknown')
            if strat not in strategy_trades_map:
                strategy_trades_map[strat] = []
            strategy_trades_map[strat].append(row_dict)

        # Calculate scores for each strategy
        for strat_name, strat_info in sorted(strategy_data.items(), key=lambda x: x[1]['count'], reverse=True):
            strat_trades = strategy_trades_map.get(strat_name, [])

            if strat_trades:
                wins = sum(1 for p in strat_trades if p['pnl_pct'] and p['pnl_pct'] > 0)
                wr = (wins / len(strat_trades) * 100)
                avg_pnl_val = sum(p['pnl_pct'] for p in strat_trades if p['pnl_pct']) / len(strat_trades)
                total_pnl_val = sum(p['pnl_pct'] for p in strat_trades if p['pnl_pct'])

                stats = calculate_stats(strat_trades)

                # Composite scoring
                pnl_component = max(0.0, min(100.0, ((avg_pnl_val + 2.0) / 4.0) * 100.0))
                pf_component = max(0.0, min(100.0, ((stats['profit_factor'] - 0.8) / 1.2) * 100.0))
                sortino_component = max(0.0, min(100.0, ((stats['sortino'] + 1.0) / 3.0) * 100.0))
                dd_component = max(0.0, min(100.0, 100.0 - (stats['max_dd'] * 5.0)))
                score = (
                    (wr * 0.35) +
                    (pnl_component * 0.25) +
                    (pf_component * 0.20) +
                    (sortino_component * 0.10) +
                    (dd_component * 0.10)
                )

                strategy_scores.append({
                    'strategy': strat_name,
                    'signals': strat_info['count'],
                    'buy_count': strat_info['by_signal'].get('BUY', 0),
                    'sell_count': strat_info['by_signal'].get('SELL', 0),
                    'closed_count': len(strat_trades),
                    'win_rate_pct': round(wr, 2),
                    'avg_pnl_pct': round(avg_pnl_val, 4),
                    'total_pnl_pct': round(total_pnl_val, 4),
                    'profit_factor': stats['profit_factor'],
                    'sortino': stats['sortino'],
                    'max_dd': stats['max_dd'],
                    'score': round(score, 2),
                    'grade': _grade_score(score),
                })
            else:
                # Strategy has signals but no closed trades
                strategy_scores.append({
                    'strategy': strat_name,
                    'signals': strat_info['count'],
                    'buy_count': strat_info['by_signal'].get('BUY', 0),
                    'sell_count': strat_info['by_signal'].get('SELL', 0),
                    'closed_count': 0,
                    'win_rate_pct': 0.0,
                    'avg_pnl_pct': 0.0,
                    'total_pnl_pct': 0.0,
                    'profit_factor': 0.0,
                    'sortino': 0.0,
                    'max_dd': 0.0,
                    'score': 0.0,
                    'grade': 'F',
                })

        # Rank by profit factor first, then score and closed count
        strategy_scores.sort(key=lambda s: (s.get('profit_factor', 0), s.get('score', 0), s.get('closed_count', 0)), reverse=True)

        # Top 10 paper trades by PnL (SQL-side sort, no Python re-sort needed)
        top_trades = [dict(r) for r in con.execute(
            f"SELECT opened_ts_iso,closed_ts_iso,symbol,side,entry_price,exit_price,status,pnl_pct "
            f"FROM paper_trades {wh + (' AND ' if wh else ' WHERE ')} status='CLOSED' "
            f"ORDER BY pnl_pct DESC LIMIT 10",
            args
        )]
        
        # Top symbols by recent signal count
        symbol_counts = Counter(s['symbol'] for s in sigs)
        top_symbols = [{'symbol': sym, 'signal_count': count} for sym, count in symbol_counts.most_common(10)]
        
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
            'recent_signals': sigs[:10],
            'confirmed_signals': confirmed[:10],
            'top_trades': top_trades,
            'top_symbols': top_symbols,
            'strategy_scores': strategy_scores[:25],
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
        time.sleep(60)  # Update every 60 seconds (monitoring dashboard, not trading execution)
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
th{background:#0f172a;padding:8px;text-align:left;border-bottom:1px solid #334155;font-weight:600;cursor:pointer;user-select:none}
th:hover{background:#1e293b}
th.sorted{color:#60a5fa}
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
<h2>Top Strategies (25)</h2>
<table><thead><tr><th data-col="strategy">Strategy</th><th data-col="signals">Signals</th><th data-col="trades">Trades</th><th data-col="wr">Win%</th><th data-col="pnl">PnL%</th><th data-col="pf">Profit Factor</th><th data-col="sortino">Sortino</th><th data-col="maxdd">Max DD%</th><th data-col="grade" title="Grade: A (85+), B (70-84), C (55-69), D (40-54), F (<40). Combines win rate, profit factor, Sortino ratio, and max drawdown into a single score.">Grade</th></tr></thead>
<tbody id="strats"></tbody></table>
</div>
<div class="section">
<h2>Top 10 Paper Trades by PnL</h2>
<table><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>Paper PnL%</th><th>Status</th></tr></thead>
<tbody id="trades"></tbody></table>
</div>
<div class="section">
<h2>Top 10 Symbols (Recent)</h2>
<table><thead><tr><th>Symbol</th><th>Signal Count</th></tr></thead>
<tbody id="symbols"></tbody></table>
</div>
<div class="section">
<h2>Price Chart</h2>
<label>Symbol: <select id="sym"></select></label>
<canvas id="chart" height="50"></canvas>
</div>
<div class="section">
<h2>Top 10 Confirmed Signals</h2>
<table><thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>Score</th></tr></thead>
<tbody id="conf"></tbody></table>
</div>
<div class="section">
<h2>Recent 10 Signals</h2>
<table><thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>Strategy</th><th>Price</th></tr></thead>
<tbody id="recent"></tbody></table>
</div>
</div>
</div>
<script>
let ch=null;
let strategyData=[];
// Default to profitability-first sorting for quick operational triage.
let currentSort={col:'pf',asc:false};
function formatAge(ms){
  const s=Math.round((Date.now()-ms)/1000);
  if(s<60)return s+'s ago';
  if(s<3600)return Math.round(s/60)+'m ago';
  return Math.round(s/3600)+'h ago';
}
function renderStrategies(){
  const sorted=[...strategyData];
  const col=currentSort.col;
  const asc=currentSort.asc;
  sorted.sort((a,b)=>{
    let av,bv;
    if(col==='strategy')av=a.strategy,bv=b.strategy;
    else if(col==='signals')av=a.signals,bv=b.signals;
    else if(col==='trades')av=a.closed_count,bv=b.closed_count;
    else if(col==='wr')av=a.win_rate_pct,bv=b.win_rate_pct;
    else if(col==='pnl')av=a.total_pnl_pct,bv=b.total_pnl_pct;
    else if(col==='pf')av=a.profit_factor,bv=b.profit_factor;
    else if(col==='sortino')av=a.sortino,bv=b.sortino;
    else if(col==='maxdd')av=a.max_dd,bv=b.max_dd;
    else if(col==='grade'){const gv={A:4,B:3,C:2,D:1,F:0};av=gv[a.grade]||0;bv=gv[b.grade]||0;}
    if(typeof av==='string')return asc?av.localeCompare(bv):bv.localeCompare(av);
    return asc?av-bv:bv-av;
  });
  let h='';
  for(let s of sorted){
    const c=s.total_pnl_pct>=0?'positive':'negative';
    const pf=s.profit_factor>1.5?'positive':(s.profit_factor>1?'#999':'negative');
    h+='<tr><td>'+s.strategy+'</td><td>'+s.signals+'</td><td>'+s.closed_count+'</td><td>'+s.win_rate_pct+'%</td><td class="'+c+'">'+s.total_pnl_pct.toFixed(1)+'%</td><td style="color:'+pf+'">'+s.profit_factor+'</td><td>'+s.sortino.toFixed(2)+'</td><td>'+s.max_dd.toFixed(2)+'%</td><td>'+s.grade+'</td></tr>';
  }
  document.getElementById('strats').innerHTML=h;
  document.querySelectorAll('table thead th[data-col]').forEach(th=>{
    th.classList.remove('sorted');
    if(th.dataset.col===col)th.classList.add('sorted');
  });
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
    
    strategyData=d.strategy_scores||[];
    renderStrategies();
    document.querySelectorAll('table thead th[data-col]').forEach(th=>{
      th.onclick=e=>{
        const col=th.dataset.col;
        if(currentSort.col===col)currentSort.asc=!currentSort.asc;
        else{currentSort.col=col;currentSort.asc=col==='strategy';}
        renderStrategies();
      };
    });
    
    // Top 10 paper trades
    h='';
    for(let t of (d.top_trades||[])){
      const c=t.pnl_pct>=0?'positive':'negative';
      h+='<tr><td>'+t.symbol+'</td><td>'+t.side+'</td><td>'+t.entry_price.toFixed(4)+'</td><td>'+(t.exit_price?t.exit_price.toFixed(4):'-')+'</td><td class="'+c+'" title="PAPER TRADING ONLY">'+t.pnl_pct.toFixed(2)+'%</td><td>'+t.status+'</td></tr>';
    }
    document.getElementById('trades').innerHTML=h;
    
    // Top 10 symbols
    h='';
    for(let sym of (d.top_symbols||[])){
      h+='<tr><td><b>'+sym.symbol+'</b></td><td>'+sym.signal_count+'</td></tr>';
    }
    document.getElementById('symbols').innerHTML=h;
    
    let opts='';
    for(let s of d.symbols_configured)opts+='<option>'+s+'</option>';
    const sel=document.getElementById('sym');
    sel.innerHTML=opts;
    sel.onchange=e=>loadChart(e.target.value);
    if(d.symbols_configured.length)loadChart(d.symbols_configured[0]);
    
    h='';
    for(let c of (d.confirmed_signals||[])){
      h+='<tr><td>'+c.ts_iso.slice(11,19)+'</td><td><b>'+c.symbol+'</b></td><td>'+c.signal+'</td><td>'+c.score+'</td></tr>';
    }
    document.getElementById('conf').innerHTML=h;
    
    h='';
    for(let sig of (d.recent_signals||[])){
      h+='<tr><td>'+sig.ts_iso.slice(11,19)+'</td><td><b>'+sig.symbol+'</b></td><td>'+sig.signal+'</td><td>'+sig.strategy+'</td><td>'+sig.price.toFixed(6)+'</td></tr>';
    }
    document.getElementById('recent').innerHTML=h;
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
setInterval(render,60000);
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
