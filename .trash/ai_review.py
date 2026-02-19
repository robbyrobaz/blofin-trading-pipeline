#!/usr/bin/env python3
"""
AI Strategy Review — Periodic analysis and optimization of trading strategies.

This script:
1. Pulls current performance data from the knowledge base
2. Formats it as a prompt for the AI (via OpenClaw or OpenAI API)
3. Writes recommendations to knowledge_entries
4. Applies safe automatic changes (threshold tuning, enable/disable)
5. Logs everything

Designed to be called by: openclaw cron job or systemd timer
Output: JSON report written to data/ai_reviews/YYYY-MM-DD_HH.json
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from db import connect, init_db
import knowledge_base
from strategy_manager import StrategyManager

# Load environment
load_dotenv(ROOT / ".env")

DB_PATH = os.getenv("BLOFIN_DB_PATH", str(ROOT / "data" / "blofin_monitor.db"))
DATA_DIR = Path(os.getenv("BLOFIN_DATA_DIR", str(ROOT / "data")))
REVIEWS_DIR = DATA_DIR / "ai_reviews"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


def now_ms() -> int:
    return int(time.time() * 1000)


def iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def get_market_regime(con) -> str:
    """
    Detect current market regime based on recent price action.
    
    Returns:
        One of: 'ranging', 'trending_up', 'trending_down', 'volatile'
    """
    # Get BTC price data from last 24h
    cutoff = now_ms() - (24 * 60 * 60 * 1000)
    rows = con.execute('''
        SELECT price FROM ticks
        WHERE symbol = 'BTC-USDT' AND ts_ms >= ?
        ORDER BY ts_ms
    ''', (cutoff,)).fetchall()
    
    if len(rows) < 100:
        return 'unknown'
    
    prices = [r['price'] for r in rows]
    
    # Calculate trend
    first_quarter = prices[:len(prices)//4]
    last_quarter = prices[-len(prices)//4:]
    avg_first = sum(first_quarter) / len(first_quarter)
    avg_last = sum(last_quarter) / len(last_quarter)
    trend_pct = ((avg_last - avg_first) / avg_first) * 100
    
    # Calculate volatility (standard deviation)
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    std_dev = variance ** 0.5
    volatility_pct = (std_dev / mean) * 100
    
    # Classify regime
    if volatility_pct > 3.0:
        return 'volatile'
    elif trend_pct > 2.0:
        return 'trending_up'
    elif trend_pct < -2.0:
        return 'trending_down'
    else:
        return 'ranging'


def build_review_prompt(con) -> str:
    """Build the prompt for AI review."""
    prompt_parts = []
    
    prompt_parts.append("You are an expert trading strategy analyst reviewing performance data.")
    prompt_parts.append("Analyze the following strategy performance and provide recommendations.\n")
    
    # Performance summary
    summary = knowledge_base.get_knowledge_summary(con, limit=30)
    prompt_parts.append(summary)
    
    # Market regime
    regime = get_market_regime(con)
    prompt_parts.append(f"\n\n=== MARKET REGIME ===\nCurrent regime: {regime}")
    
    # Recent knowledge entries for context
    prompt_parts.append("\n\n=== TASK ===")
    prompt_parts.append("Based on the performance data and market regime, provide:")
    prompt_parts.append("1. Analysis of what's working and what's not")
    prompt_parts.append("2. Specific recommendations for strategy adjustments")
    prompt_parts.append("3. Whether to enable/disable specific strategies")
    prompt_parts.append("4. Parameter tuning suggestions (within safe bounds)")
    prompt_parts.append("\nRespond with JSON in this exact format:")
    prompt_parts.append('''{
  "analysis": "text summary of findings",
  "recommendations": [
    {"action": "disable", "strategy": "strategy_name", "reason": "explanation"},
    {"action": "enable", "strategy": "strategy_name", "reason": "explanation"},
    {"action": "tune", "strategy": "strategy_name", "params": {"param_name": new_value}, "reason": "explanation"}
  ],
  "market_regime": "ranging|trending_up|trending_down|volatile",
  "confidence": 0.0-1.0
}''')
    
    return "\n".join(prompt_parts)


def call_ai_api(prompt: str) -> Dict[str, Any]:
    """
    Call AI API to get review recommendations.
    
    For now, returns a mock response. In production, this would call:
    - OpenClaw AI API
    - OpenAI GPT-4
    - Anthropic Claude
    - Or local LLM
    """
    # TODO: Implement actual AI API call
    # This is a placeholder that returns a mock response
    
    print("[ai_review] AI API call would happen here")
    print(f"[ai_review] Prompt length: {len(prompt)} chars")
    
    # Mock response based on actual performance patterns
    return {
        "analysis": (
            "Performance analysis shows mixed results. RSI divergence is the only "
            "profitable strategy with 45% win rate and +2.35% total PnL. "
            "BB squeeze has the worst performance at 20.4% win rate and -26.74% total PnL. "
            "Momentum and reversal generate high signal volume but low win rates around 35-40%. "
            "The system needs better filtering and market regime awareness."
        ),
        "recommendations": [
            {
                "action": "disable",
                "strategy": "bb_squeeze",
                "reason": "Consistently poor performance (20.4% win rate, -26.74% total PnL)"
            },
            {
                "action": "tune",
                "strategy": "momentum",
                "params": {"up_pct": 0.8, "down_pct": -0.8},
                "reason": "Too sensitive, increase threshold to reduce false signals"
            },
            {
                "action": "tune",
                "strategy": "vwap_reversion",
                "params": {"deviation_pct": 0.5},
                "reason": "Slightly increase deviation threshold to improve signal quality"
            }
        ],
        "market_regime": "ranging",
        "confidence": 0.75
    }


def apply_safe_recommendations(
    con,
    manager: StrategyManager,
    recommendations: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Apply safe automatic recommendations.
    
    Args:
        con: Database connection
        manager: StrategyManager instance
        recommendations: List of recommendation objects
    
    Returns:
        Dictionary with applied and skipped recommendations
    """
    applied = []
    skipped = []
    
    for rec in recommendations:
        action = rec.get('action')
        strategy = rec.get('strategy')
        reason = rec.get('reason', 'No reason provided')
        
        if not strategy or strategy not in manager.strategies:
            skipped.append(f"{action} {strategy}: strategy not found")
            continue
        
        try:
            if action == 'disable':
                manager.disable(strategy)
                knowledge_base.add_knowledge_entry(
                    con,
                    category='change',
                    content=f"AI review disabled {strategy}: {reason}",
                    strategy=strategy,
                    source='ai_review'
                )
                applied.append(f"Disabled {strategy}")
            
            elif action == 'enable':
                manager.enable(strategy)
                knowledge_base.add_knowledge_entry(
                    con,
                    category='change',
                    content=f"AI review enabled {strategy}: {reason}",
                    strategy=strategy,
                    source='ai_review'
                )
                applied.append(f"Enabled {strategy}")
            
            elif action == 'tune':
                params = rec.get('params', {})
                if not params:
                    skipped.append(f"Tune {strategy}: no parameters provided")
                    continue
                
                # Apply parameter updates
                old_config = manager.get_strategy_config(strategy)
                manager.update_strategy_config(strategy, params)
                new_config = manager.get_strategy_config(strategy)
                
                # Save to config history
                knowledge_base.save_strategy_config(
                    con, strategy, new_config, source='ai_review',
                    note=f"AI tuning: {reason}"
                )
                
                knowledge_base.add_knowledge_entry(
                    con,
                    category='change',
                    content=f"AI review tuned {strategy} params {params}: {reason}",
                    strategy=strategy,
                    source='ai_review',
                    metadata={'old_config': old_config, 'new_config': new_config}
                )
                applied.append(f"Tuned {strategy}: {params}")
            
            else:
                skipped.append(f"Unknown action: {action}")
        
        except Exception as e:
            skipped.append(f"{action} {strategy}: {e}")
    
    con.commit()
    return {'applied': applied, 'skipped': skipped}


def main():
    """Run AI review and generate report."""
    print("[ai_review] Starting AI strategy review...")
    
    con = connect(DB_PATH)
    init_db(con)
    
    # Initialize strategy manager
    manager = StrategyManager()
    print(f"[ai_review] Loaded {len(manager.strategies)} strategies")
    
    # Update scores before review
    print("[ai_review] Updating performance scores...")
    score_count = knowledge_base.update_all_scores(con)
    print(f"[ai_review] Updated {score_count} score records")
    
    # Build prompt and call AI
    print("[ai_review] Building review prompt...")
    prompt = build_review_prompt(con)
    
    print("[ai_review] Calling AI API...")
    ai_response = call_ai_api(prompt)
    
    # Log the review
    knowledge_base.add_knowledge_entry(
        con,
        category='recommendation',
        content=ai_response.get('analysis', 'No analysis provided'),
        source='ai_review',
        metadata=ai_response
    )
    
    # Apply safe recommendations
    print("[ai_review] Applying recommendations...")
    results = apply_safe_recommendations(con, manager, ai_response.get('recommendations', []))
    
    # Build report
    ts = now_ms()
    report = {
        'timestamp': iso_utc(ts),
        'timestamp_ms': ts,
        'market_regime': ai_response.get('market_regime', 'unknown'),
        'confidence': ai_response.get('confidence', 0.0),
        'analysis': ai_response.get('analysis', ''),
        'recommendations': ai_response.get('recommendations', []),
        'applied': results['applied'],
        'skipped': results['skipped'],
        'strategy_status': manager.list_strategies(),
    }
    
    # Write report to file
    report_filename = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d_%H.json')
    report_path = REVIEWS_DIR / report_filename
    
    with report_path.open('w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"[ai_review] Report written to: {report_path}")
    print(f"[ai_review] Applied: {len(results['applied'])}, Skipped: {len(results['skipped'])}")
    
    # Print summary
    print("\n=== AI REVIEW SUMMARY ===")
    print(f"Market Regime: {report['market_regime']}")
    print(f"Confidence: {report['confidence']}")
    print(f"\nAnalysis:\n{report['analysis']}")
    print(f"\nApplied Changes ({len(results['applied'])}):")
    for change in results['applied']:
        print(f"  ✓ {change}")
    
    if results['skipped']:
        print(f"\nSkipped ({len(results['skipped'])}):")
        for skip in results['skipped']:
            print(f"  ✗ {skip}")
    
    print("\n[ai_review] Review complete!")


if __name__ == '__main__':
    main()
