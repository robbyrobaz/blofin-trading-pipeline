#!/usr/bin/env python3
"""Quick diagnostic: feature manager, backtester, and one pipeline cycle."""
import sys, os, time, json, traceback
import sqlite3

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = "data/blofin_monitor.db"

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ── Step 0: Check data availability ──
section("STEP 0: Data check")
con = sqlite3.connect(DB_PATH)
row = con.execute("SELECT COUNT(*) as cnt, MIN(ts_ms) as mn, MAX(ts_ms) as mx FROM ticks WHERE symbol='BTC-USDT'").fetchone()
print(f"BTC-USDT ticks: {row[0]:,} rows, range {row[1]} → {row[2]}")
if row[0] == 0:
    print("FATAL: No BTC-USDT tick data. Aborting.")
    sys.exit(1)
# Check how many days of data
days = (row[2] - row[1]) / (86400 * 1000)
print(f"Data span: {days:.1f} days")
con.close()

# ── Step 1: Feature Manager ──
section("STEP 1: Feature Manager – load BTC-USDT, generate features")
try:
    from features.feature_manager import FeatureManager
    fm = FeatureManager(db_path=DB_PATH)
    df = fm.get_features('BTC-USDT', timeframe='5m', lookback_bars=200)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    # Check NaN in last 50 rows
    tail = df.tail(50)
    nan_counts = tail.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"⚠ NaN in last 50 rows: {dict(nan_cols)}")
    else:
        print("✓ No NaN in last 50 rows")
    print(f"✓ Feature manager OK – {len(df)} candles, {len(df.columns)} features")
except Exception as e:
    print(f"✗ Feature manager FAILED: {e}")
    traceback.print_exc()

# ── Step 2: Backtester with real strategy ──
section("STEP 2: Backtester – load 7 days, run strategy")
try:
    from backtester.backtest_engine import BacktestEngine
    
    # The backtester calls strategy.detect(context_candles, symbol)
    # But strategies expect detect(symbol, price, volume, ts_ms, prices, volumes)
    # We need an adapter
    class BacktestAdapter:
        """Wraps a BaseStrategy to work with BacktestEngine's calling convention."""
        def __init__(self, strategy):
            self.strategy = strategy
            self.name = strategy.name
        
        def detect(self, candles, symbol):
            """Adapt candle-list call to strategy's tick-based interface."""
            if not candles or len(candles) < 5:
                return None
            last = candles[-1]
            price = last['close']
            volume = last.get('volume', 0)
            ts_ms = last['ts_ms']
            prices = [(c['ts_ms'], c['close']) for c in candles]
            volumes = [(c['ts_ms'], c.get('volume', 0)) for c in candles]
            signal = self.strategy.detect(symbol, price, volume, ts_ms, prices, volumes)
            if signal:
                return {'signal': signal.signal, 'confidence': signal.confidence}
            return None
    
    from strategies.momentum import MomentumStrategy
    adapter = BacktestAdapter(MomentumStrategy())
    
    engine = BacktestEngine(symbol='BTC-USDT', days_back=7, db_path=DB_PATH)
    print(f"Loaded {len(engine.ticks):,} ticks → {len(engine.ohlcv_1m):,} 1m candles")
    
    result = engine.run_strategy(adapter, timeframe='5m', stop_loss_pct=3.0, take_profit_pct=5.0)
    print(f"Trades: {len(result['trades'])}, Final capital: ${result['final_capital']:.2f}, Candles: {result['num_candles']}")
    if result.get('metrics'):
        print(f"Metrics: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k,v in result['metrics'].items()}, indent=2)}")
    print("✓ Backtester OK")
except Exception as e:
    print(f"✗ Backtester FAILED: {e}")
    traceback.print_exc()

# ── Step 3: Full pipeline cycle ──
section("STEP 3: Pipeline cycle – design strategies, backtest, train ML, score, rank")

# 3a: "Design" 2-3 strategies (use existing ones as stand-ins)
print("\n--- 3a: Strategies ---")
strategies_to_test = []
try:
    from strategies.momentum import MomentumStrategy
    from strategies.ema_crossover import EMACrossoverStrategy
    strategies_to_test.append(BacktestAdapter(MomentumStrategy()))
    strategies_to_test.append(BacktestAdapter(EMACrossoverStrategy()))
    
    # Try loading a third
    try:
        from strategies.rsi_divergence import RSIDivergenceStrategy
        strategies_to_test.append(BacktestAdapter(RSIDivergenceStrategy()))
    except Exception:
        pass
    
    print(f"✓ Loaded {len(strategies_to_test)} strategies: {[s.name for s in strategies_to_test]}")
except Exception as e:
    print(f"✗ Strategy loading failed: {e}")
    traceback.print_exc()

# 3b: Backtest each
print("\n--- 3b: Backtest each strategy ---")
backtest_results = []
for strat in strategies_to_test:
    try:
        engine = BacktestEngine(symbol='BTC-USDT', days_back=7, db_path=DB_PATH)
        bt = engine.run_strategy(strat, timeframe='5m', stop_loss_pct=3.0, take_profit_pct=5.0)
        print(f"  {strat.name}: {len(bt['trades'])} trades, ${bt['final_capital']:.2f}")
        backtest_results.append((strat.name, bt))
    except Exception as e:
        print(f"  ✗ {strat.name} failed: {e}")
        traceback.print_exc()

# 3c: Train 1 ML model
print("\n--- 3c: Train ML model ---")
ml_result = None
try:
    from features.feature_manager import FeatureManager
    from ml_pipeline.target_generator import add_targets
    
    fm = FeatureManager(db_path=DB_PATH)
    features_df = fm.get_features('BTC-USDT', timeframe='1m', lookback_bars=1000)
    print(f"  Features shape: {features_df.shape}")
    
    # Add targets
    features_df = add_targets(features_df, lookback=5)
    print(f"  After targets: {features_df.shape}")
    
    # Remove non-numeric and target columns
    exclude = ['timestamp', 'target_direction', 'target_price', 'target_momentum', 'target_volatility', 'momentum']
    feature_cols = [c for c in features_df.columns if c not in exclude and features_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X = features_df[feature_cols].copy()
    y = features_df['target_direction'].copy()
    
    # Drop any remaining NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    print(f"  Clean samples: {len(X)}")
    
    if len(X) < 50:
        print("  ⚠ Too few samples, using synthetic fallback")
        raise ValueError("Too few samples")
    
    # Train a simple model (sklearn, no xgboost dependency issues)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    test_f1 = f1_score(y_test, model.predict(X_test), average='binary')
    
    ml_result = {
        'model_name': 'direction_gb_diag',
        'model_type': 'gradient_boosting',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': test_f1,
        'n_features': len(feature_cols),
        'n_samples': len(X),
    }
    print(f"  ✓ ML model trained: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, f1={test_f1:.4f}")
    
except Exception as e:
    print(f"  ✗ ML training failed: {e}")
    traceback.print_exc()

# 3d: Save results to database
print("\n--- 3d: Save to database ---")
try:
    con = sqlite3.connect(DB_PATH, timeout=30)
    
    # Ensure tables exist
    sys.path.insert(0, '.')
    import db
    db.init_db(con)
    
    ts_ms = int(time.time() * 1000)
    ts_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    
    # Save backtest results
    for strat_name, bt in backtest_results:
        metrics = bt.get('metrics', {})
        con.execute('''
            INSERT INTO strategy_backtest_results 
            (ts_ms, ts_iso, strategy, symbol, backtest_window_days, total_trades, 
             win_rate, sharpe_ratio, max_drawdown_pct, total_pnl_pct, score, metrics_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ts_ms, ts_iso, strat_name, 'BTC-USDT', 7,
            len(bt['trades']),
            metrics.get('win_rate', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('max_drawdown_pct', 0),
            metrics.get('total_pnl_pct', 0),
            metrics.get('sharpe_ratio', 0),  # use sharpe as score
            json.dumps(metrics),
            'active'
        ))
    print(f"  ✓ Saved {len(backtest_results)} backtest results")
    
    # Save ML result
    if ml_result:
        con.execute('''
            INSERT INTO ml_model_results
            (ts_ms, ts_iso, model_name, model_type, symbol, train_accuracy, test_accuracy, 
             f1_score, config_json, metrics_json, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ts_ms, ts_iso, ml_result['model_name'], ml_result['model_type'], 'BTC-USDT',
            ml_result['train_accuracy'], ml_result['test_accuracy'], ml_result['f1_score'],
            json.dumps({'n_features': ml_result['n_features']}),
            json.dumps(ml_result),
            0
        ))
        print(f"  ✓ Saved ML model result")
    
    con.commit()
    
    # 3e: Score & rank
    print("\n--- 3e: Score and rank ---")
    cur = con.execute('''
        SELECT strategy, symbol, win_rate, sharpe_ratio, total_pnl_pct, total_trades, score
        FROM strategy_backtest_results 
        WHERE status='active' AND symbol='BTC-USDT'
        ORDER BY score DESC LIMIT 10
    ''')
    rows = cur.fetchall()
    print(f"  Top strategies (BTC-USDT):")
    for r in rows:
        print(f"    {r[0]}: win_rate={r[2]:.2f}, sharpe={r[3]:.2f}, pnl={r[4]:.2f}%, trades={r[5]}, score={r[6]:.2f}")
    
    cur = con.execute('''
        SELECT model_name, model_type, test_accuracy, f1_score
        FROM ml_model_results WHERE archived=0
        ORDER BY test_accuracy DESC LIMIT 5
    ''')
    rows = cur.fetchall()
    print(f"  Top ML models:")
    for r in rows:
        print(f"    {r[0]} ({r[1]}): acc={r[2]:.4f}, f1={r[3]:.4f}")
    
    con.close()
    print("\n✓ All results saved to database")
    
except Exception as e:
    print(f"✗ Database save failed: {e}")
    traceback.print_exc()

section("DIAGNOSTIC COMPLETE")
