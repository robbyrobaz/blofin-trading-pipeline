"""Thin wrapper for Anthropic API calls used by the pipeline."""

import json
import os
from pathlib import Path

def _get_api_key():
    """Get Anthropic API key from multiple sources (env var, .env file, auth profiles)."""
    # 1. Check environment variable
    key = os.environ.get('ANTHROPIC_API_KEY')
    if key:
        return key
    
    # 2. Check .env file in project root
    try:
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('ANTHROPIC_API_KEY='):
                        key = line.split('=', 1)[1].strip().strip('"\'')
                        if key:
                            return key
    except Exception:
        pass
    
    # 3. Check OpenClaw auth profiles
    try:
        profiles_path = os.path.expanduser('~/.openclaw/agents/main/agent/auth-profiles.json')
        with open(profiles_path) as f:
            data = json.load(f)
        token = data['profiles']['anthropic:default']['token']
        # Validate token format (should start with sk-ant-api)
        if token and token.startswith('sk-ant-api'):
            return token
    except Exception:
        pass
    
    return None

MODEL_MAP = {
    'opus': 'claude-opus-4-20250901',
    'sonnet': 'claude-sonnet-4-20250514',
    'haiku': 'claude-haiku-4-20250414',
}

def call_llm(prompt: str, model: str = 'sonnet', max_tokens: int = 4096) -> str:
    """Call Anthropic API and return the text response."""
    import anthropic
    
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("No Anthropic API key found")
    
    client = anthropic.Anthropic(api_key=api_key)
    model_id = MODEL_MAP.get(model, model)
    
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
