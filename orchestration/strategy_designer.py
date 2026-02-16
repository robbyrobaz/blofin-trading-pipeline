#!/usr/bin/env python3
"""
Strategy designer using Claude Opus for creative strategy generation.
Analyzes current portfolio and market conditions to propose new strategies.
"""
import sqlite3
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import subprocess


class StrategyDesigner:
    def __init__(self, db_path: str, strategies_dir: str):
        self.db_path = db_path
        self.strategies_dir = Path(strategies_dir)
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
    
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con
    
    def _get_top_performers(self, con: sqlite3.Connection, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing strategies to understand what works."""
        cursor = con.execute('''
            SELECT 
                strategy, score, win_rate, sharpe_ratio, total_pnl_pct, trades, window
            FROM strategy_scores
            WHERE enabled = 1 AND score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def _get_bottom_performers(self, con: sqlite3.Connection, limit: int = 5) -> List[Dict[str, Any]]:
        """Get worst performing strategies to understand what to avoid."""
        cursor = con.execute('''
            SELECT 
                strategy, score, win_rate, sharpe_ratio, total_pnl_pct, trades, window
            FROM strategy_scores
            WHERE enabled = 1 AND score IS NOT NULL
            ORDER BY score ASC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def _analyze_market_regime(self, con: sqlite3.Connection) -> Dict[str, Any]:
        """Determine current market regime from recent data."""
        # Get recent price data
        cursor = con.execute('''
            SELECT price, ts_ms 
            FROM ticks 
            WHERE symbol = 'BTC-USDT'
            ORDER BY ts_ms DESC 
            LIMIT 1000
        ''')
        prices = [row['price'] for row in cursor.fetchall()]
        
        if len(prices) < 100:
            return {
                'regime': 'unknown',
                'volatility': 'unknown',
                'trend': 'unknown',
                'confidence': 'low'
            }
        
        # Simple regime analysis
        prices = list(reversed(prices))  # Oldest to newest
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        avg_return = sum(returns) / len(returns)
        volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        # Classify
        if abs(avg_return) < 0.0001:
            trend = 'ranging'
        elif avg_return > 0:
            trend = 'trending_up'
        else:
            trend = 'trending_down'
        
        vol_label = 'high' if volatility > 0.01 else 'medium' if volatility > 0.005 else 'low'
        
        return {
            'regime': f"{trend}_{vol_label}_volatility",
            'volatility': vol_label,
            'trend': trend,
            'avg_return': avg_return,
            'volatility_score': volatility,
            'confidence': 'medium'
        }
    
    def _identify_gaps(self, con: sqlite3.Connection) -> List[str]:
        """Identify gaps in current strategy portfolio."""
        # Get all active strategies
        cursor = con.execute('''
            SELECT DISTINCT strategy FROM strategy_scores WHERE enabled = 1
        ''')
        active_strategies = [row['strategy'] for row in cursor.fetchall()]
        
        gaps = []
        
        # Check for missing strategy types
        has_momentum = any('momentum' in s.lower() or 'trend' in s.lower() for s in active_strategies)
        has_mean_reversion = any('reversion' in s.lower() or 'rsi' in s.lower() for s in active_strategies)
        has_volatility = any('volatility' in s.lower() or 'bollinger' in s.lower() for s in active_strategies)
        has_volume = any('volume' in s.lower() for s in active_strategies)
        has_pattern = any('pattern' in s.lower() or 'candlestick' in s.lower() for s in active_strategies)
        
        if not has_momentum:
            gaps.append("No momentum-based strategies")
        if not has_mean_reversion:
            gaps.append("No mean-reversion strategies")
        if not has_volatility:
            gaps.append("No volatility-based strategies")
        if not has_volume:
            gaps.append("No volume-based strategies")
        if not has_pattern:
            gaps.append("No pattern recognition strategies")
        
        # Check for symbol coverage
        cursor = con.execute('''
            SELECT DISTINCT symbol FROM strategy_scores WHERE enabled = 1
        ''')
        covered_symbols = [row['symbol'] for row in cursor.fetchall() if row['symbol']]
        
        if not covered_symbols or len(covered_symbols) < 3:
            gaps.append("Limited symbol coverage")
        
        return gaps
    
    def _build_design_prompt(self, top_performers: List[Dict], bottom_performers: List[Dict],
                            market_regime: Dict, gaps: List[str]) -> str:
        """Build comprehensive prompt for Opus."""
        prompt = f"""You are an expert quantitative trading strategy designer. Design a new crypto trading strategy for the blofin-stack system.

CURRENT PORTFOLIO ANALYSIS:

Top 5 Performing Strategies:
"""
        for i, strat in enumerate(top_performers, 1):
            prompt += f"{i}. {strat['strategy']}: score={strat['score']:.3f}, win_rate={strat['win_rate']:.1f}%, sharpe={strat['sharpe_ratio']:.2f}\n"
        
        prompt += f"\nBottom 5 Performing Strategies (AVOID THESE PATTERNS):\n"
        for i, strat in enumerate(bottom_performers, 1):
            prompt += f"{i}. {strat['strategy']}: score={strat['score']:.3f}, win_rate={strat['win_rate']:.1f}%, sharpe={strat['sharpe_ratio']:.2f}\n"
        
        prompt += f"\nCURRENT MARKET REGIME:\n"
        prompt += f"  - Regime: {market_regime['regime']}\n"
        prompt += f"  - Trend: {market_regime['trend']}\n"
        prompt += f"  - Volatility: {market_regime['volatility']}\n"
        
        prompt += f"\nIDENTIFIED GAPS IN PORTFOLIO:\n"
        for gap in gaps:
            prompt += f"  - {gap}\n"
        
        prompt += f"""
REQUIREMENTS:
1. Design a strategy that fills one or more gaps
2. Learn from top performers (what works)
3. Avoid patterns from bottom performers (what doesn't work)
4. Adapt to current market regime
5. Return ONLY valid Python code implementing the strategy
6. Use this template structure:

```python
#!/usr/bin/env python3
\"\"\"
Strategy Name: <descriptive name>
Type: <momentum/mean-reversion/volatility/hybrid>
Timeframe: <1m/5m/15m/1h>
Description: <brief explanation of logic>
\"\"\"

class Strategy:
    def __init__(self):
        self.name = "<strategy_name>"
        self.params = {{
            # Strategy parameters
        }}
    
    def analyze(self, candles, indicators):
        \"\"\"
        Analyze market data and generate signal.
        
        Args:
            candles: List of recent candles [{{open, high, low, close, volume}}, ...]
            indicators: Dict of pre-calculated indicators {{rsi, macd, bbands, etc.}}
        
        Returns:
            'BUY', 'SELL', or 'HOLD'
        \"\"\"
        # Strategy logic here
        return 'HOLD'
    
    def get_confidence(self, candles, indicators):
        \"\"\"Return confidence score 0-1 for current signal.\"\"\"
        return 0.5
```

Provide ONLY the Python code, no explanation, no markdown, just the code.
"""
        return prompt
    
    def _call_opus(self, prompt: str) -> str:
        """Call Claude Opus via OpenClaw CLI with file-based prompt."""
        import tempfile
        import os
        try:
            # Write prompt to temp file to avoid CLI length limits
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name
            
            result = subprocess.run(
                ['openclaw', 'chat', '--model', 'opus', '--file', prompt_file],
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
                env={**os.environ, 'NO_COLOR': '1'}  # Disable ANSI colors
            )
            
            os.unlink(prompt_file)
            
            # Validate output
            if not result.stdout.strip():
                error_msg = f"Empty Opus output. stderr: {result.stderr}"
                print(f"ERROR: {error_msg}")
                raise Exception(error_msg)
            
            if result.returncode != 0:
                error_msg = f"Opus call failed with code {result.returncode}. stderr: {result.stderr}"
                print(f"ERROR: {error_msg}")
                raise Exception(error_msg)
            
            return result.stdout.strip()
        except Exception as e:
            raise Exception(f"Failed to call Opus: {e}")
    
    def _extract_code(self, opus_output: str) -> str:
        """Extract Python code from Opus output with validation."""
        if not opus_output or not opus_output.strip():
            raise ValueError("Cannot extract code from empty Opus output")
        
        # Try to find code block
        code_match = re.search(r'```python\n(.*?)```', opus_output, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            if code:
                return code
        
        # If no code block, try to find shebang
        if opus_output.strip().startswith('#!/usr/bin/env python'):
            return opus_output.strip()
        
        # Last resort: assume entire output is code (if it looks like Python)
        if 'class Strategy' in opus_output or 'def analyze' in opus_output:
            return opus_output.strip()
        
        # Log failure for debugging
        print(f"WARNING: Could not extract code from Opus output. First 500 chars:")
        print(opus_output[:500])
        raise ValueError("Failed to extract valid Python code from Opus output")
    
    def _get_next_strategy_number(self) -> int:
        """Find next available strategy number."""
        existing = list(self.strategies_dir.glob('strategy_*.py'))
        if not existing:
            return 1
        
        numbers = []
        for path in existing:
            match = re.search(r'strategy_(\d+)\.py', path.name)
            if match:
                numbers.append(int(match.group(1)))
        
        return max(numbers) + 1 if numbers else 1
    
    def _save_strategy(self, code: str, strategy_num: int) -> Path:
        """Save strategy code to file with validation."""
        # VALIDATE: Check if code is sufficient
        if not code or len(code.strip()) < 100:
            raise ValueError(f"Code too short ({len(code)} chars), refusing to save")
        
        # VALIDATE: Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            print(f"ERROR: Generated code has syntax errors: {e}")
            print(f"Code:\n{code}")
            raise ValueError(f"Code has syntax errors: {e}")
        
        filename = f"strategy_{strategy_num:03d}.py"
        filepath = self.strategies_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        # Make executable
        filepath.chmod(0o755)
        
        # VALIDATE: File was written
        if not filepath.exists() or filepath.stat().st_size == 0:
            raise IOError(f"Strategy file was not written or is empty: {filepath}")
        
        print(f"✓ Strategy saved: {filepath} ({filepath.stat().st_size} bytes)")
        
        return filepath
    
    def _register_strategy(self, con: sqlite3.Connection, strategy_name: str, 
                          strategy_num: int, config: Dict[str, Any]):
        """Register new strategy in database."""
        ts_ms = int(datetime.utcnow().timestamp() * 1000)
        ts_iso = datetime.utcnow().isoformat() + 'Z'
        
        con.execute('''
            INSERT INTO strategy_configs (ts_ms, ts_iso, strategy, config_json, source, note)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ts_ms, ts_iso, strategy_name, json.dumps(config), 'opus_designer', 
              f'Auto-generated strategy #{strategy_num}'))
        
        con.commit()
    
    def design_new_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Design a new strategy using Opus.
        
        Returns:
            Dict with strategy info, or None if design failed
        """
        con = self._connect()
        try:
            # Gather intelligence
            top_performers = self._get_top_performers(con)
            bottom_performers = self._get_bottom_performers(con)
            market_regime = self._analyze_market_regime(con)
            gaps = self._identify_gaps(con)
            
            # Build prompt
            prompt = self._build_design_prompt(top_performers, bottom_performers, market_regime, gaps)
            
            print("Calling Opus to design new strategy...")
            opus_output = self._call_opus(prompt)
            
            # VALIDATE: Check if Opus returned anything
            if not opus_output or len(opus_output.strip()) < 100:
                print(f"ERROR: Opus returned insufficient output ({len(opus_output)} chars)")
                print(f"Output preview: {opus_output[:200] if opus_output else 'None'}")
                return None
            
            # Extract code
            try:
                code = self._extract_code(opus_output)
            except ValueError as e:
                print(f"ERROR: Code extraction failed: {e}")
                print(f"Opus output length: {len(opus_output)}")
                return None
            
            # VALIDATE: Check if code extraction succeeded
            if not code or len(code) < 100:
                print(f"ERROR: Code extraction produced insufficient code ({len(code)} chars)")
                print(f"Opus output:\n{opus_output[:500]}")
                return None
            
            # Save strategy with validation
            strategy_num = self._get_next_strategy_number()
            try:
                filepath = self._save_strategy(code, strategy_num)
            except (ValueError, IOError) as e:
                print(f"ERROR: Failed to save strategy: {e}")
                return None
            
            # Extract strategy name from code
            name_match = re.search(r'self\.name\s*=\s*["\']([^"\']+)["\']', code)
            strategy_name = name_match.group(1) if name_match else f'strategy_{strategy_num:03d}'
            
            # Register in database
            config = {
                'file': str(filepath),
                'designed_at': datetime.utcnow().isoformat() + 'Z',
                'market_regime': market_regime,
                'gaps_addressed': gaps
            }
            self._register_strategy(con, strategy_name, strategy_num, config)
            
            print(f"Strategy designed and saved: {filepath}")
            
            return {
                'strategy_name': strategy_name,
                'strategy_num': strategy_num,
                'filepath': str(filepath),
                'code': code,
                'config': config
            }
            
        except Exception as e:
            print(f"Strategy design failed: {e}")
            return None
        finally:
            con.close()


if __name__ == '__main__':
    # Quick test
    import os
    db_path = os.path.expanduser('~/.openclaw/workspace/blofin-stack/data/blofin_monitor.db')
    strategies_dir = os.path.expanduser('~/.openclaw/workspace/blofin-stack/strategies')
    
    designer = StrategyDesigner(db_path, strategies_dir)
    print("Testing strategy designer (dry run - won't call Opus)...")
    
    con = designer._connect()
    top = designer._get_top_performers(con, 5)
    bottom = designer._get_bottom_performers(con, 5)
    regime = designer._analyze_market_regime(con)
    gaps = designer._identify_gaps(con)
    con.close()
    
    print(f"Top performers: {len(top)}")
    print(f"Bottom performers: {len(bottom)}")
    print(f"Market regime: {regime['regime']}")
    print(f"Gaps identified: {gaps}")
    print("\nDesigner test complete!")
