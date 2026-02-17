#!/usr/bin/env python3
"""Quick test of the llm_client to verify Anthropic API integration."""

import sys
from orchestration.llm_client import call_llm

def test_haiku():
    """Test with Haiku (cheapest, fastest)."""
    print("Testing llm_client with Haiku...")
    prompt = "Respond with exactly: 'API working'"
    try:
        response = call_llm(prompt, model='haiku', max_tokens=50)
        print(f"✓ Response received: {response[:100]}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_sonnet():
    """Test with Sonnet (strategy tuner model)."""
    print("\nTesting llm_client with Sonnet...")
    prompt = "You are a trading strategy optimizer. Respond with 'Sonnet online' if you understand."
    try:
        response = call_llm(prompt, model='sonnet', max_tokens=100)
        print(f"✓ Response received: {response[:100]}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

if __name__ == '__main__':
    haiku_ok = test_haiku()
    sonnet_ok = test_sonnet()
    
    if haiku_ok and sonnet_ok:
        print("\n✓✓✓ All LLM tests passed! Strategy generation should work.")
        sys.exit(0)
    else:
        print("\n✗✗✗ LLM tests failed. Check API key configuration.")
        sys.exit(1)
