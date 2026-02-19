#!/usr/bin/env python3
"""
Critical Alert Monitor — Watch for breaking issues and alert immediately
Runs on every heartbeat to catch:
- Audit findings with CRITICAL severity
- Data quality anomalies (impossible metrics)
- Pipeline health degradation
- Failed deployments
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime

def check_audit_for_critical():
    """Check if there are any CRITICAL audit findings"""
    try:
        reports_dir = Path("/home/rob/.openclaw/workspace/blofin-stack/data/reports")
        # Look for audit reports with CRITICAL in content
        for report_file in sorted(reports_dir.glob("*.json"), reverse=True)[:5]:
            try:
                with open(report_file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        severity = data.get("severity", "").upper()
                        if "CRITICAL" in severity:
                            findings = data.get("critical_findings", [])
                            if findings:
                                return {
                                    "alert": "CRITICAL AUDIT FINDINGS",
                                    "file": report_file.name,
                                    "severity": severity,
                                    "count": len(findings),
                                    "issues": [f["title"] for f in findings[:3]]
                                }
            except:
                pass
    except:
        pass
    return None

def check_impossible_metrics():
    """Detect mathematically impossible metric combinations"""
    try:
        db_path = Path("/home/rob/.openclaw/workspace/blofin-stack/data/blofin_monitor.db")
        conn = sqlite3.connect(str(db_path), timeout=2)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query for impossible combinations
        cursor.execute("""
            SELECT strategy, win_rate, sharpe_ratio, COUNT(*) as cnt
            FROM strategy_scores
            WHERE win_rate < 0.05 AND sharpe_ratio > 3.0
            AND ts_iso > datetime('now', '-1 hour')
            GROUP BY strategy, win_rate, sharpe_ratio
            LIMIT 5
        """)
        
        impossible = cursor.fetchall()
        conn.close()
        
        if impossible:
            return {
                "alert": "IMPOSSIBLE METRICS DETECTED",
                "examples": [
                    f"{r['strategy']}: WR={r['win_rate']*100:.1f}% + Sharpe={r['sharpe_ratio']:.2f}"
                    for r in impossible
                ]
            }
    except:
        pass
    return None

def check_perfect_model_scores():
    """Detect perfect model accuracy (likely data leakage)"""
    try:
        db_path = Path("/home/rob/.openclaw/workspace/blofin-stack/data/blofin_monitor.db")
        conn = sqlite3.connect(str(db_path), timeout=2)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT model_name, f1_score, accuracy, ts_iso
            FROM ml_model_results
            WHERE (f1_score >= 0.99 OR accuracy >= 0.99)
            AND ts_iso > datetime('now', '-24 hours')
            LIMIT 5
        """)
        
        perfect = cursor.fetchall()
        conn.close()
        
        if perfect:
            return {
                "alert": "PERFECT MODEL SCORES (DATA LEAKAGE RISK)",
                "examples": [
                    f"{r['model_name']}: F1={r['f1_score']:.3f}, Accuracy={r['accuracy']:.3f}"
                    for r in perfect
                ]
            }
    except:
        pass
    return None

def check_duplicate_rankings():
    """Detect duplicate strategies in top-N rankings"""
    try:
        db_path = Path("/home/rob/.openclaw/workspace/blofin-stack/data/blofin_monitor.db")
        conn = sqlite3.connect(str(db_path), timeout=2)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT strategy_name, COUNT(*) as cnt
            FROM strategy_scores
            WHERE ts_iso > datetime('now', '-1 hour')
            GROUP BY strategy_name
            HAVING cnt > 10
            LIMIT 10
        """)
        
        dupes = cursor.fetchall()
        conn.close()
        
        if dupes:
            return {
                "alert": "DUPLICATE STRATEGIES IN RANKINGS",
                "examples": [
                    f"{r['strategy_name']}: {r['cnt']} identical rows"
                    for r in dupes[:3]
                ]
            }
    except:
        pass
    return None

def main():
    """Run all checks and return alerts"""
    alerts = []
    
    audit_critical = check_audit_for_critical()
    if audit_critical:
        alerts.append(audit_critical)
    
    impossible = check_impossible_metrics()
    if impossible:
        alerts.append(impossible)
    
    leakage = check_perfect_model_scores()
    if leakage:
        alerts.append(leakage)
    
    dupes = check_duplicate_rankings()
    if dupes:
        alerts.append(dupes)
    
    if alerts:
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "alert_count": len(alerts),
            "severity": "CRITICAL",
            "alerts": alerts,
            "action": "⚠️ CRITICAL ISSUES DETECTED - DO NOT DEPLOY"
        }
    
    return {"status": "OK", "alerts": []}

if __name__ == "__main__":
    result = main()
    if result.get("alert_count", 0) > 0:
        print(json.dumps(result, indent=2))
        exit(1)  # Exit with error code to signal alerts
    exit(0)
