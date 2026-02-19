#!/usr/bin/env python3
"""Test strategy generation with the new OpenClaw-based LLM client."""

import os
import sys
from orchestration.strategy_designer import StrategyDesigner

def main():
    db_path = os.path.expanduser('~/.openclaw/workspace/blofin-stack/data/blofin_monitor.db')
    strategies_dir = os.path.expanduser('~/.openclaw/workspace/blofin-stack/strategies')
    
    designer = StrategyDesigner(db_path, strategies_dir)
    
    print("=" * 60)
    print("Testing Strategy Generation with OpenClaw Authentication")
    print("=" * 60)
    print()
    
    print("Attempting to design a new strategy using Opus via OpenClaw...")
    print("This will use Rob's pre-configured authentication.")
    print()
    
    result = designer.design_new_strategy()
    
    if result:
        print()
        print("✓ SUCCESS! Strategy generated successfully!")
        print(f"  Strategy Name: {result['strategy_name']}")
        print(f"  Strategy Number: {result['strategy_num']}")
        print(f"  File Path: {result['filepath']}")
        print(f"  Code Length: {len(result['code'])} characters")
        print()
        print("First 500 chars of generated code:")
        print("-" * 60)
        print(result['code'][:500])
        print("-" * 60)
        return 0
    else:
        print()
        print("✗ FAILED - Strategy generation did not succeed")
        print("Check the error messages above for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
