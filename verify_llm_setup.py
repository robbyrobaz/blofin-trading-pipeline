#!/usr/bin/env python3
"""
Verify that the LLM integration is properly configured for strategy generation.
This script tests the Anthropic API connection and validates model access.
"""

import sys
from pathlib import Path

def check_api_key():
    """Check if API key is configured."""
    print("1. Checking API key configuration...")
    from orchestration.llm_client import _get_api_key
    
    key = _get_api_key()
    if not key:
        print("   ✗ No API key found!")
        print("   → Add ANTHROPIC_API_KEY to .env file")
        print("   → Get key from: https://console.anthropic.com/settings/keys")
        return False
    
    if not key.startswith('sk-ant-api'):
        print(f"   ⚠ API key has unusual format: {key[:15]}...")
        print("   → Expected format: sk-ant-api...")
        print("   → OAuth tokens (sk-ant-oat...) won't work")
        return False
    
    print(f"   ✓ API key found: {key[:20]}...")
    return True

def test_haiku():
    """Test Haiku (cheapest model for validation)."""
    print("\n2. Testing Haiku API access...")
    try:
        from orchestration.llm_client import call_llm
        response = call_llm("Respond with: 'Haiku OK'", model='haiku', max_tokens=20)
        print(f"   ✓ Haiku works: {response.strip()}")
        return True
    except Exception as e:
        print(f"   ✗ Haiku failed: {e}")
        return False

def test_sonnet():
    """Test Sonnet (strategy tuner model)."""
    print("\n3. Testing Sonnet API access...")
    try:
        from orchestration.llm_client import call_llm
        response = call_llm("Respond with: 'Sonnet OK'", model='sonnet', max_tokens=20)
        print(f"   ✓ Sonnet works: {response.strip()}")
        return True
    except Exception as e:
        print(f"   ✗ Sonnet failed: {e}")
        return False

def test_opus():
    """Test Opus (strategy designer model)."""
    print("\n4. Testing Opus API access...")
    try:
        from orchestration.llm_client import call_llm
        response = call_llm("Respond with: 'Opus OK'", model='opus', max_tokens=20)
        print(f"   ✓ Opus works: {response.strip()}")
        return True
    except Exception as e:
        print(f"   ✗ Opus failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Blofin LLM Integration Verification")
    print("=" * 60)
    
    # Check prerequisites
    api_key_ok = check_api_key()
    if not api_key_ok:
        print("\n" + "=" * 60)
        print("SETUP REQUIRED:")
        print("  1. Get API key from https://console.anthropic.com/")
        print("  2. Add to .env file: ANTHROPIC_API_KEY=sk-ant-api-...")
        print("  3. Run this script again")
        print("=" * 60)
        sys.exit(1)
    
    # Test models
    haiku_ok = test_haiku()
    sonnet_ok = test_sonnet()
    opus_ok = test_opus()
    
    print("\n" + "=" * 60)
    if haiku_ok and sonnet_ok and opus_ok:
        print("✓✓✓ ALL TESTS PASSED")
        print("\nStrategy generation is ready:")
        print("  • strategy_designer.py can generate new strategies (Opus)")
        print("  • strategy_tuner.py can optimize strategies (Sonnet)")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗✗✗ SOME TESTS FAILED")
        print("\nCheck:")
        print("  • API key is valid and active")
        print("  • Account has access to Claude 3/4 models")
        print("  • Network connectivity to Anthropic API")
        print("=" * 60)
        sys.exit(1)

if __name__ == '__main__':
    main()
