"""LLM client using OpenClaw's gateway for authenticated model calls."""

import subprocess
import json
import hashlib
from datetime import datetime

def call_llm(prompt: str, model: str = 'sonnet', max_tokens: int = 4096) -> str:
    """
    Call Claude via OpenClaw gateway (uses existing auth infrastructure).
    
    This uses subprocess to call `openclaw agent`, relying on OpenClaw's
    configured authentication and model routing. The gateway already has
    Anthropic credentials configured.
    
    Args:
        prompt: The prompt to send to the model
        model: Model preference hint (opus/sonnet/haiku) - note: actual model
               selection is controlled by OpenClaw's routing config
        max_tokens: Maximum tokens in response (informational)
        
    Returns:
        The model's text response
        
    Raises:
        RuntimeError: If the call fails
    """
    # Create a unique session ID for this call to avoid state pollution
    # Include model hint in session name for potential routing
    timestamp = datetime.now().isoformat()
    session_hash = hashlib.md5(f"{model}-{timestamp}".encode()).hexdigest()[:8]
    session_id = f"blofin-llm-{model}-{session_hash}"
    
    # Prepare the message with model preference hint
    # OpenClaw may use this for routing, or fall back to configured defaults
    message = f"[Model preference: {model}]\n\n{prompt}"
    
    cmd = [
        'openclaw', 'agent',
        '--message', message,
        '--session-id', session_id,
        '--json',
        '--channel', 'last',  # Non-delivery mode
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for complex generations
            check=False
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"OpenClaw agent call failed (exit {result.returncode}): {error_msg}"
            )
        
        # Parse JSON response from OpenClaw
        try:
            data = json.loads(result.stdout)
            
            # Extract the actual text response
            # OpenClaw agent JSON structure: {status, result: {payloads: [{text}]}}
            if data.get('status') == 'ok' and 'result' in data:
                payloads = data['result'].get('payloads', [])
                if payloads and len(payloads) > 0:
                    first_payload = payloads[0]
                    if isinstance(first_payload, dict) and 'text' in first_payload:
                        text = first_payload['text']
                        if text:
                            return str(text).strip()
            
            # Fallback: check other possible response structures
            if 'response' in data:
                return str(data['response']).strip()
            if 'text' in data:
                return str(data['text']).strip()
            
            # If we got here, structure was unexpected
            raise RuntimeError(
                f"Unexpected response structure from OpenClaw: {list(data.keys())}"
            )
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse OpenClaw JSON response: {e}")
    
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"OpenClaw agent call timed out after 300s")
    except FileNotFoundError:
        raise RuntimeError(
            "openclaw command not found. Ensure OpenClaw CLI is installed and in PATH."
        )
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Failed to call OpenClaw agent: {e}")


def test_connection() -> bool:
    """
    Test that OpenClaw gateway is accessible and working.
    
    Returns:
        True if connection works, False otherwise
    """
    try:
        response = call_llm("Respond with: 'OK'", model='haiku', max_tokens=10)
        return 'ok' in response.lower() or 'ready' in response.lower()
    except Exception:
        return False
