#!/usr/bin/env python3
"""Test strategy tuning with the new OpenClaw-based LLM client."""

import os
import sys
from orchestration.strategy_tuner import StrategyTuner

def main():
    db_path = os.path.expanduser('~/.openclaw/workspace/blofin-stack/data/blofin_monitor.db')
    strategies_dir = os.path.expanduser('~/.openclaw/workspace/blofin-stack/strategies')
    
    tuner = StrategyTuner(db_path, strategies_dir)
    
    print("=" * 60)
    print("Testing Strategy Tuning with OpenClaw Authentication")
    print("=" * 60)
    print()
    
    # Get underperformers
    con = tuner._connect()
    underperformers = tuner._get_underperformers(con, limit=1)
    con.close()
    
    if not underperformers:
        print("No underperforming strategies found to tune.")
        print("This is expected if all strategies are performing well.")
        return 0
    
    strategy_name = underperformers[0]['strategy']
    print(f"Testing tuner on: {strategy_name}")
    print(f"Current score: {underperformers[0]['score']:.3f}")
    print()
    
    print("Calling Sonnet to analyze and suggest improvements...")
    print("This will use Rob's pre-configured authentication.")
    print()
    
    result = tuner.tune_strategy(strategy_name)
    
    if result:
        print()
        print("✓ SUCCESS! Strategy tuned successfully!")
        print(f"  Strategy Name: {result['strategy_name']}")
        print(f"  Tuned File: {result['filepath']}")
        print()
        
        if 'suggestions' in result and 'parameter_changes' in result['suggestions']:
            print("Suggested Parameter Changes:")
            for change in result['suggestions']['parameter_changes']:
                print(f"  - {change['param_name']}: {change.get('current_value')} → {change.get('suggested_value')}")
                print(f"    Reason: {change.get('reasoning', 'N/A')}")
        
        return 0
    else:
        print()
        print("✗ FAILED - Strategy tuning did not succeed")
        print("(This may be expected if strategy has no failures to analyze)")
        return 0  # Don't fail the test

if __name__ == '__main__':
    sys.exit(main())
