#!/usr/bin/env python3
"""
Verify that the LLM integration is properly configured for strategy generation.
This script tests the OpenClaw gateway connection and validates model access.
"""

import sys
from pathlib import Path

def check_openclaw_available():
    """Check if openclaw CLI is available."""
    print("1. Checking OpenClaw CLI availability...")
    import subprocess
    try:
        result = subprocess.run(
            ['openclaw', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split()[1] if len(result.stdout.split()) > 1 else 'unknown'
            print(f"   ✓ OpenClaw CLI found: {version}")
            return True
        else:
            print("   ✗ OpenClaw CLI not responding properly")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"   ✗ OpenClaw CLI not found: {e}")
        return False

def check_gateway():
    """Check if OpenClaw gateway is running."""
    print("\n2. Checking OpenClaw gateway status...")
    import subprocess
    try:
        result = subprocess.run(
            ['openclaw', 'health'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("   ✓ Gateway is running and healthy")
            return True
        else:
            print("   ✗ Gateway health check failed")
            print(f"   → Run: openclaw gateway start")
            return False
    except Exception as e:
        print(f"   ✗ Gateway not accessible: {e}")
        print("   → Run: openclaw gateway start")
        return False

def test_llm_call(model: str, test_prompt: str = None) -> bool:
    """Test a specific model via llm_client."""
    if test_prompt is None:
        test_prompt = f"Respond with exactly: '{model.upper()} OK'"
    
    print(f"\n3.{ord(model[0])-ord('a')+1}. Testing {model} model...")
    try:
        from orchestration.llm_client import call_llm
        response = call_llm(test_prompt, model=model, max_tokens=50)
        
        # Check if response is reasonable
        if not response or len(response.strip()) == 0:
            print(f"   ✗ {model} returned empty response")
            return False
        
        print(f"   ✓ {model} works")
        print(f"   Response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"   ✗ {model} failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Blofin LLM Integration Verification")
    print("=" * 60)
    
    # Check prerequisites
    cli_ok = check_openclaw_available()
    if not cli_ok:
        print("\n" + "=" * 60)
        print("SETUP REQUIRED:")
        print("  OpenClaw CLI not found or not working")
        print("  → Install: npm install -g @openclaw/cli")
        print("  → Or check PATH")
        print("=" * 60)
        sys.exit(1)
    
    gateway_ok = check_gateway()
    if not gateway_ok:
        print("\n" + "=" * 60)
        print("GATEWAY NOT RUNNING:")
        print("  → Start gateway: openclaw gateway start")
        print("  → Check status: openclaw health")
        print("=" * 60)
        sys.exit(1)
    
    # Test models (note: actual model used depends on OpenClaw routing config)
    print("\n" + "=" * 60)
    print("Testing Models (via OpenClaw Gateway)")
    print("=" * 60)
    
    haiku_ok = test_llm_call('haiku')
    sonnet_ok = test_llm_call('sonnet')
    opus_ok = test_llm_call('opus')
    
    print("\n" + "=" * 60)
    if haiku_ok and sonnet_ok and opus_ok:
        print("✓✓✓ ALL TESTS PASSED")
        print("\nStrategy generation is ready:")
        print("  • strategy_designer.py can generate new strategies")
        print("  • strategy_tuner.py can optimize strategies")
        print("\nNote: Actual model selection is controlled by OpenClaw's")
        print("      routing configuration (see: openclaw models status)")
        print("=" * 60)
        sys.exit(0)
    else:
        print("⚠ PARTIAL SUCCESS")
        print("\nSome models failed, but basic connectivity works.")
        print("Check:")
        print("  • Model auth: openclaw models status")
        print("  • Model config: openclaw models list")
        print("  • Gateway logs: openclaw logs")
        print("=" * 60)
        sys.exit(1)

if __name__ == '__main__':
    main()
