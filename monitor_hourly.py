#!/usr/bin/env python3
"""Monitor hourly pipeline runs and report status."""

import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
import re

WORK_DIR = Path("/home/rob/.openclaw/workspace/blofin-stack")
LOG_DIR = WORK_DIR / "data"
REPORTS_DIR = LOG_DIR / "reports"

def get_latest_runs(count=10):
    """Get latest pipeline runs."""
    logs = sorted(glob.glob(str(LOG_DIR / "hourly_run_*.log")), reverse=True)[:count]
    
    runs = []
    for log_file in logs:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract metrics
        models = re.search(r'"models_trained":\s*(\d+)', content)
        strategies = re.search(r'"strategies_designed":\s*(\d+)', content)
        duration = re.search(r'BLOFIN PIPELINE RUN - (.*)', content)
        
        run = {
            'file': Path(log_file).name,
            'timestamp': duration.group(1) if duration else 'unknown',
            'models': int(models.group(1)) if models else 0,
            'strategies': int(strategies.group(1)) if strategies else 0,
        }
        runs.append(run)
    
    return runs

def get_latest_report():
    """Get latest report."""
    reports = sorted(glob.glob(str(REPORTS_DIR / "*.json")), reverse=True)
    if not reports:
        return None
    
    with open(reports[0], 'r') as f:
        return json.load(f)

def print_status():
    """Print pipeline status."""
    print("\n" + "="*70)
    print("BLOFIN AI PIPELINE - HOURLY MONITORING STATUS")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S MST')}")
    
    # Latest runs
    runs = get_latest_runs(5)
    if runs:
        print("\n📊 Latest 5 Runs:")
        print("-" * 70)
        for i, run in enumerate(runs, 1):
            status = "✅" if run['models'] > 0 else "⚠️"
            print(f"  {i}. {run['timestamp']}")
            print(f"     Models: {run['models']} {status} | Strategies: {run['strategies']}")
    
    # Latest report
    report = get_latest_report()
    if report:
        print("\n📈 Latest Report:")
        print("-" * 70)
        print(f"  Date: {report.get('date', 'unknown')}")
        print(f"  Activity:")
        activity = report.get('activity', {})
        print(f"    - Strategies designed: {activity.get('strategies_designed', 0)}")
        print(f"    - Models trained: {activity.get('models_trained', 0)}")
        print(f"    - Ensembles tested: {activity.get('ensembles_tested', 0)}")
        
        # Top strategies
        top_strats = report.get('top_strategies', [])[:3]
        if top_strats:
            print(f"\n  Top 3 Strategies:")
            for strat in top_strats:
                print(f"    - {strat['strategy']}: score={strat['score']:.1f}, sharpe={strat['sharpe_ratio']:.2f}")
    
    # Summary
    print("\n" + "="*70)
    if runs and runs[0]['models'] > 0:
        print("✅ SYSTEM STATUS: ML PIPELINE WORKING!")
        print("   Next steps: Monitor performance, keep running hourly")
    else:
        print("⚠️  SYSTEM STATUS: Debugging in progress")
        print("   Pipeline running every hour at :02 mark")
    
    print("="*70 + "\n")

if __name__ == '__main__':
    print_status()
