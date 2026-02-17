"""LLM client using OpenClaw's gateway for authenticated model calls.

Implements Claude-aware rate limiting:
- Tracks requests per 1-minute sliding window (Claude's RPM limit: 1000)
- Backs off exponentially when approaching ceiling
- Per-minute reset (not 4-6 hour cycles as previously assumed)
"""

import subprocess
import json
import hashlib
from datetime import datetime, timedelta
import time
from collections import deque
from threading import Lock


class ClaudeRateLimiter:
    """
    Claude-aware rate limiter for 1-minute request windows.
    
    Limits: 1000 requests per minute (RPM) per Claude Max account.
    Strategy: Track timestamps in 60-second window, back off exponentially near ceiling.
    """
    
    # Claude rate limits
    MAX_RPM = 1000
    BACKOFF_THRESHOLD = 950  # Start backing off at 95% of RPM limit
    
    def __init__(self):
        self.request_times = deque()  # Timestamps of requests in current minute
        self.lock = Lock()
        self.backoff_multiplier = 1.0  # Exponential backoff factor
    
    def _cleanup_old_requests(self):
        """Remove requests older than 60 seconds from the window."""
        cutoff_time = datetime.now() - timedelta(seconds=60)
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
    
    def get_wait_time(self) -> float:
        """
        Calculate wait time before next request.
        
        Returns:
            Seconds to wait (0 if can proceed immediately)
        """
        with self.lock:
            self._cleanup_old_requests()
            
            requests_this_minute = len(self.request_times)
            
            # If under 95% of limit, no wait needed
            if requests_this_minute < self.BACKOFF_THRESHOLD:
                self.backoff_multiplier = 1.0
                return 0.0
            
            # Exponential backoff: requests per second → exponential delay
            # At 95% (950 RPM) → 16 requests/sec → wait ~0.5s
            # At 99% (990 RPM) → 16.5 requests/sec → wait ~1.5s
            deficit = requests_this_minute - self.BACKOFF_THRESHOLD
            backoff_seconds = (deficit / 50.0) ** 1.5  # Aggressive exponential
            
            self.backoff_multiplier = backoff_seconds
            return backoff_seconds
    
    def wait_if_needed(self) -> float:
        """
        Block until safe to make next request. Returns actual wait time.
        """
        wait_time = self.get_wait_time()
        if wait_time > 0:
            print(f"[RATE LIMIT] Backing off {wait_time:.2f}s ({len(self.request_times)} requests this minute)")
            time.sleep(wait_time)
        return wait_time
    
    def record_request(self):
        """Mark that a request was just made."""
        with self.lock:
            self.request_times.append(datetime.now())


# Global rate limiter instance (shared across all call_llm calls)
_rate_limiter = ClaudeRateLimiter()


def call_llm(prompt: str, model: str = 'sonnet', max_tokens: int = 4096) -> str:
    """
    Call Claude via OpenClaw gateway (uses existing auth infrastructure).
    
    This uses subprocess to call `openclaw agent`, relying on OpenClaw's
    configured authentication and model routing. The gateway already has
    Anthropic credentials configured.
    
    **CLAUDE RATE LIMITING:** Automatically waits to respect 1000 RPM limit.
    When approaching ceiling (950+ requests/min), backs off exponentially.
    
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
    # **RATE LIMITING:** Wait if needed before making request
    wait_time = _rate_limiter.wait_if_needed()
    
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
                            _rate_limiter.record_request()
                            return str(text).strip()
            
            # Fallback: check other possible response structures
            if 'response' in data:
                _rate_limiter.record_request()
                return str(data['response']).strip()
            if 'text' in data:
                _rate_limiter.record_request()
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
