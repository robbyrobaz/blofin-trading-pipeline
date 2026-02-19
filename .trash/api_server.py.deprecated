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
        
        # Strategy scores — optimizer results first, then library backtest data
        strategy_scores = []
        optimizer_names = set()

        # Import EEP scorer for live ranking
        try:
            import sys as _sys
            _sys.path.insert(0, str(ROOT))
            from eep_scorer import compute_eep_from_metrics as _eep_score, rank_by_eep as _rank_by_eep
            _has_eep = True
        except Exception:
            _has_eep = False

        # 1. Pull from optimizer_runs (highest priority)
        try:
            opt_row = con.execute(
                'SELECT raw_json FROM optimizer_runs ORDER BY ts_ms DESC LIMIT 1'
            ).fetchone()
            if opt_row and opt_row[0]:
                import json as _json
                opt_data = _json.loads(opt_row[0])
                raw_strats = opt_data.get('top_strategies', [])
                # Score all with EEP and rank
                if _has_eep:
                    ranked_strats = _rank_by_eep(raw_strats)
                else:
                    ranked_strats = [dict(s, eep_rank=i+1, eep_score=0, gate_pass=True, gate_fails=[])
                                     for i, s in enumerate(raw_strats)]
                for s in ranked_strats:
                    strat_name = s.get('strategy', 'unknown')
                    optimizer_names.add(strat_name)
                    wr = float(s.get('win_rate', 0))
                    sharpe = float(s.get('sharpe_ratio', 0))
                    pnl = float(s.get('total_pnl_pct', 0))
                    dd = float(s.get('max_drawdown_pct', 0))
                    trades = int(s.get('num_trades', 0))
                    eep_score = float(s.get('eep_score', 0))
                    strategy_scores.append({
                        'strategy': f"{strat_name} ({s.get('symbol','')})",
                        'signals': 0,
                        'buy_count': 0,
                        'sell_count': 0,
                        'closed_count': trades,
                        'win_rate_pct': round(wr * 100, 2),
                        'avg_pnl_pct': round(float(s.get('avg_pnl_pct', pnl / max(trades, 1))), 4),
                        'total_pnl_pct': round(pnl, 2),
                        'profit_factor': round(s.get('eep_detail', {}).get('profit_factor', 0) if _has_eep else 0, 2),
                        'sortino': round(sharpe, 2),
                        'max_dd': round(dd, 2),
                        'score': round(eep_score, 2),
                        'eep_rank': s.get('eep_rank', 0),
                        'gate_pass': s.get('gate_pass', True),
                        'gate_fails': s.get('gate_fails', []),
                        'grade': _grade_score(eep_score),
                        'source': 'optimizer',
                    })
        except Exception as _e:
            print(f'optimizer_runs query error: {_e}')

        # 2. Library backtest from strategy_scores (only strategies not in optimizer)
        strategy_rows = con.execute(
            'SELECT DISTINCT strategy FROM strategy_scores ORDER BY strategy'
        ).fetchall()

        for (strat_name,) in strategy_rows:
            if strat_name in optimizer_names:
                continue  # already covered by optimizer
            latest = con.execute(
                '''SELECT 
                    COUNT(*) as eval_count,
                    AVG(win_rate) as avg_win_rate,
                    AVG(avg_pnl_pct) as avg_pnl,
                    AVG(sharpe_ratio) as avg_sharpe,
                    AVG(max_drawdown_pct) as avg_max_dd,
                    MAX(score) as max_score
                FROM strategy_scores 
                WHERE strategy = ?
                ''',
                (strat_name,)
            ).fetchone()

            if latest:
                eval_count, avg_wr, avg_pnl, avg_sharpe, avg_max_dd, max_score = latest
                win_pct = (avg_wr * 100) if avg_wr else 0
                # Compute EEP for library strategies too
                lib_s = {
                    "win_rate": avg_wr or 0,
                    "sharpe_ratio": avg_sharpe or 0,
                    "total_pnl_pct": avg_pnl * eval_count if avg_pnl else 0,
                    "avg_pnl_pct": avg_pnl or 0,
                    "max_drawdown_pct": avg_max_dd or 0,
                    "num_trades": eval_count or 0,
                    "risk_params": {},
                }
                if _has_eep:
                    lib_eep = _eep_score(lib_s)
                    final_score = lib_eep["eep_score"]
                    lib_gate_pass = lib_eep["gate_pass"]
                    lib_gate_fails = lib_eep["gate_fails"]
                else:
                    final_score = max_score if max_score else 0
                    lib_gate_pass = True
                    lib_gate_fails = []
                grade = _grade_score(final_score)
                strategy_scores.append({
                    'strategy': strat_name,
                    'signals': int(sum(1 for s in sigs if s.get('strategy') == strat_name)),
                    'buy_count': int(sum(1 for s in sigs if s.get('strategy') == strat_name and s.get('signal') == 'BUY')),
                    'sell_count': int(sum(1 for s in sigs if s.get('strategy') == strat_name and s.get('signal') == 'SELL')),
                    'closed_count': int(eval_count),
                    'win_rate_pct': round(win_pct, 2),
                    'avg_pnl_pct': round(avg_pnl, 4) if avg_pnl else 0,
                    'profit_factor': 0,
                    'sortino': round(avg_sharpe, 2) if avg_sharpe else 0,
                    'max_dd': round(avg_max_dd, 2) if avg_max_dd else 0,
                    'score': round(final_score, 2),
                    'eep_rank': None,  # filled below after ranking
                    'gate_pass': lib_gate_pass,
                    'gate_fails': lib_gate_fails,
                    'grade': grade,
                    'source': 'library_backtest',
                })

        # Optimizer strategies always lead, then library sorted by EEP score
        # Gate-passing strategies ranked before non-passing within each group
        opt_entries = [s for s in strategy_scores if s.get('source') == 'optimizer']
        lib_entries = [s for s in strategy_scores if s.get('source') != 'optimizer']
        opt_entries.sort(key=lambda s: (0 if s.get('gate_pass', True) else 1, -s.get('score', 0)))
        lib_entries.sort(key=lambda s: (0 if s.get('gate_pass', True) else 1, -s.get('score', 0)))
        strategy_scores = opt_entries + lib_entries

        # Assign EEP ranks to any entries without one (library backtest)
        global_rank = 1
        for entry in strategy_scores:
            if entry.get('eep_rank') is None:
                entry['eep_rank'] = global_rank
            global_rank += 1

        # Enrich with strategy_registry tier data
        tier_map = {}
        try:
            reg_rows = con.execute('SELECT strategy_name, tier, bt_eep_score, ft_eep_score FROM strategy_registry WHERE archived=0').fetchall()
            for rr in reg_rows:
                tier_map[rr[0]] = {'tier': rr[1], 'bt_eep': rr[2], 'ft_eep': rr[3]}
        except:
            pass  # table doesn't exist yet

        for entry in strategy_scores:
            strat_name = entry['strategy'].split(' (')[0]  # strip " (SYMBOL)" suffix
            reg = tier_map.get(strat_name, {})
            entry['tier'] = reg.get('tier')
            entry['bt_eep'] = reg.get('bt_eep')
            entry['ft_eep'] = reg.get('ft_eep')

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
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
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

        if p.path == '/api/strategies':
            # Return top strategies with EEP scores from strategy_scores + backtest tables
            try:
                limit = int(q.get('limit', ['50'])[0])
                source_filter = q.get('source', [None])[0]

                # Pull latest entry per strategy+symbol from strategy_scores
                query = """
                    SELECT ss.strategy, ss.symbol, ss.window, ss.trades, ss.wins, ss.losses,
                           ss.win_rate, ss.avg_pnl_pct, ss.total_pnl_pct, ss.sharpe_ratio,
                           ss.max_drawdown_pct, ss.score,
                           MAX(ss.ts_ms) as latest_ts, ss.ts_iso
                    FROM strategy_scores ss
                    WHERE ss.enabled = 1
                    GROUP BY ss.strategy, ss.symbol
                    ORDER BY ss.score DESC
                    LIMIT ?
                """
                rows = [dict(r) for r in con.execute(query, (limit,)).fetchall()]

                # Enrich with EEP metadata from strategy_backtest_results
                enriched = []
                for row in rows:
                    br = con.execute(
                        """SELECT metrics_json, config_json
                           FROM strategy_backtest_results
                           WHERE strategy=? AND symbol=?
                           ORDER BY ts_ms DESC LIMIT 1""",
                        (row["strategy"], row["symbol"] or "")
                    ).fetchone()
                    extra = {}
                    if br:
                        try:
                            extra = json.loads(br["metrics_json"] or "{}")
                        except Exception:
                            pass
                    row["eep_score"] = row.get("score", 0)
                    row["source"] = extra.get("source", "library_backtest")
                    if source_filter and row["source"] != source_filter:
                        continue
                    enriched.append(row)

                result = json.dumps({
                    "strategies": enriched,
                    "count": len(enriched),
                    "fetched_at_ms": int(time.time() * 1000),
                }, default=str).encode()
                return self.sendb(result)
            except Exception as e:
                return self.sendb(json.dumps({"error": str(e)}).encode(), code=500)

        if p.path == '/api/execution_calibration':
            # Serve the execution calibration config
            calib_path = ROOT / 'data' / 'execution_calibration.json'
            if calib_path.exists():
                with open(calib_path) as f:
                    result = f.read().encode()
                return self.sendb(result)
            else:
                return self.sendb(json.dumps({"error": "not generated yet"}).encode(), code=404)

        if p.path == '/api/registry':
            try:
                registry = []
                try:
                    rows = con.execute('''
                        SELECT strategy_name, tier, bt_win_rate, bt_sharpe, bt_pnl_pct, bt_max_dd,
                               bt_trades, bt_eep_score, ft_win_rate, ft_sharpe, ft_pnl_pct, ft_max_dd,
                               ft_trades, ft_eep_score, file_path, source, strategy_type,
                               archived, archive_reason
                        FROM strategy_registry
                        WHERE archived = 0
                        ORDER BY COALESCE(ft_eep_score, bt_eep_score, 0) DESC
                    ''').fetchall()
                    registry = [dict(r) for r in rows]
                except Exception:
                    pass  # table doesn't exist yet — graceful degradation
                result = json.dumps({"strategies": registry, "count": len(registry)}, default=str).encode()
                return self.sendb(result)
            except Exception as e:
                return self.sendb(json.dumps({"error": str(e)}).encode(), code=500)

        if p.path == '/api/ml_models':
            try:
                rows = con.execute('''
                    SELECT model_name, model_type, train_accuracy, test_accuracy,
                           f1_score, precision_score, recall_score, roc_auc, archived
                    FROM ml_model_results
                    WHERE archived = 0
                    ORDER BY COALESCE(test_accuracy, train_accuracy, 0) DESC
                ''').fetchall()
                models = []
                for r in rows:
                    m = dict(r)
                    # Flag leakage: train_accuracy >= 0.95 with no test_accuracy or test matches train
                    train_acc = m.get('train_accuracy') or 0
                    test_acc = m.get('test_accuracy')
                    m['leakage_suspected'] = train_acc >= 0.95 and (test_acc is None or abs(train_acc - (test_acc or 0)) < 0.05)
                    models.append(m)
                result = json.dumps({"models": models, "count": len(models)}, default=str).encode()
                return self.sendb(result)
            except Exception as e:
                return self.sendb(json.dumps({"error": str(e)}).encode(), code=500)

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
<div class="card"><div class="label">Best EEP</div><div class="value" id="wr">—</div></div>
</div>
<div class="section">
<h2>Top Strategies (25)</h2>
<table><thead><tr><th data-col="eep_rank" title="EEP Rank: Entry+Exit Package score (0-100). Gate-passing strategies ranked first.">EEP#</th><th data-col="tier" title="Strategy tier: T2=Forward Test, T1=Backtest, T0=Library">Tier</th><th data-col="strategy">Strategy</th><th data-col="signals">Signals</th><th data-col="trades">Trades</th><th data-col="wr">Win%</th><th data-col="pnl">PnL%</th><th data-col="pf">Profit Factor</th><th data-col="eep_score" title="EEP Score 0-100: 70% entry quality (PF 30%, Sharpe 25%, DD 20%, Sortino 15%, Expectancy 10%) + 30% exit quality (MPC 25%, RR 20%, SHF 15%, BEUR 10%)">EEP Score</th><th data-col="sortino">Sortino</th><th data-col="maxdd">Max DD%</th><th data-col="gate" title="Hard gates: PF≥1.3, Sharpe≥0.8, MDD≤35%, Trades≥30, positive expectancy">Gates</th></tr></thead>
<tbody id="strats"></tbody></table>
</div>
<div class="section">
<h2>ML Model Status</h2>
<table><thead><tr><th>Model</th><th>Type</th><th>Train Acc</th><th>Test Acc</th><th>F1</th><th>Status</th></tr></thead>
<tbody id="mlmodels"></tbody></table>
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
let currentSort={col:'eep_score',asc:false};
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
    else if(col==='eep_score')av=a.score||0,bv=b.score||0;
    else if(col==='eep_rank')av=a.eep_rank||999,bv=b.eep_rank||999;
    else if(col==='sortino')av=a.sortino,bv=b.sortino;
    else if(col==='maxdd')av=a.max_dd,bv=b.max_dd;
    else if(col==='tier')av=a.tier||0,bv=b.tier||0;
    else if(col==='gate')av=a.gate_pass?1:0,bv=b.gate_pass?1:0;
    if(typeof av==='string')return asc?av.localeCompare(bv):bv.localeCompare(av);
    return asc?av-bv:bv-av;
  });
  let h='';
  for(let i=0;i<sorted.length;i++){
    const s=sorted[i];
    const c=s.total_pnl_pct>=0?'positive':'negative';
    const pf=s.profit_factor>1.5?'positive':(s.profit_factor>1?'#999':'negative');
    const eepColor=s.score>=70?'#10b981':(s.score>=50?'#f59e0b':'#ef4444');
    const gateIcon=s.gate_pass===false?'<span title="'+(s.gate_fails||[]).join(', ')+'">⛔</span>':'✅';
    const rank=s.eep_rank||('#'+(i+1));
    const tierLabel=s.tier===2?'T2':s.tier===1?'T1':'T0';
    const tierColor=s.tier===2?'#10b981':s.tier===1?'#60a5fa':'#6b7280';
    const borderColor=s.tier===2?'#10b981':s.tier===1?'#60a5fa':'#334155';
    h+='<tr style="border-left:3px solid '+borderColor+'"><td style="color:#60a5fa;font-weight:700">#'+rank+'</td><td style="color:'+tierColor+';font-weight:600">'+tierLabel+'</td><td>'+s.strategy+'</td><td>'+s.signals+'</td><td>'+s.closed_count+'</td><td>'+s.win_rate_pct+'%</td><td class="'+c+'">'+s.total_pnl_pct.toFixed(1)+'%</td><td style="color:'+pf+'">'+s.profit_factor+'</td><td style="color:'+eepColor+';font-weight:600">'+(s.score||0).toFixed(1)+'</td><td>'+s.sortino.toFixed(2)+'</td><td>'+s.max_dd.toFixed(2)+'%</td><td>'+gateIcon+'</td></tr>';
  }
  document.getElementById('strats').innerHTML=h;
  document.querySelectorAll('table thead th[data-col]').forEach(th=>{
    th.classList.remove('sorted');
    if(th.dataset.col===col)th.classList.add('sorted');
  });
}
async function render(){
  try{
    document.getElementById('loading').textContent='Fetching data...';
    const r=await fetch('/api/summary');
    if(!r.ok){document.getElementById('loading').innerHTML='API error: '+r.status+' '+r.statusText; return;}
    const d=await r.json();
    if(!d || !d.strategy_scores){document.getElementById('loading').innerHTML='No data returned'; return;}
    if(d.error){document.getElementById('loading').textContent=d.error; return;}
    document.getElementById('loading').style.display='none';
    document.getElementById('content').style.display='block';
    document.getElementById('age').textContent=formatAge(d.fetched_at_ms);
    document.getElementById('status').textContent=d.live_status.is_live?'✓ LIVE':'✗ STALE';
    document.getElementById('status').style.color=d.live_status.is_live?'#10b981':'#ef4444';
    document.getElementById('signals').textContent=d.signals_count;
    document.getElementById('confirmed').textContent=d.confirmed_count;
    const bestEep=d.strategy_scores&&d.strategy_scores.length?Math.max(...d.strategy_scores.map(s=>s.score||0)):0;
    document.getElementById('wr').textContent=bestEep.toFixed(1);
    
    strategyData=d.strategy_scores||[];
    renderStrategies();
    // Fetch ML model status
    try{
      const mr=await fetch('/api/ml_models');
      if(mr.ok){const md=await mr.json();let mh='';for(let m of(md.models||[])){const ta=(m.train_accuracy||0).toFixed(3);const te=m.test_accuracy!=null?m.test_accuracy.toFixed(3):'-';const f1=m.f1_score!=null?m.f1_score.toFixed(3):'-';const leak=m.leakage_suspected;const st=leak?'<span style="background:#ef4444;color:#fff;padding:2px 6px;border-radius:3px;font-size:10px">LEAKAGE</span>':'<span style="color:#10b981">OK</span>';mh+='<tr><td>'+m.model_name+'</td><td>'+m.model_type+'</td><td>'+ta+'</td><td>'+te+'</td><td>'+f1+'</td><td>'+st+'</td></tr>';}document.getElementById('mlmodels').innerHTML=mh||'<tr><td colspan="6" style="color:#94a3b8">No ML models found</td></tr>';}
    }catch(e){}
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
