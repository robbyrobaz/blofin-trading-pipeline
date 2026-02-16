#!/usr/bin/env python3
"""
Strategy tuner using Claude Sonnet to analyze and improve underperforming strategies.
Identifies failure patterns and suggests parameter adjustments.
"""
import sqlite3
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import subprocess


class StrategyTuner:
    def __init__(self, db_path: str, strategies_dir: str):
        self.db_path = db_path
        self.strategies_dir = Path(strategies_dir)
    
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con
    
    def _get_underperformers(self, con: sqlite3.Connection, threshold_score: float = 0.5,
                            limit: int = 5) -> List[Dict[str, Any]]:
        """Identify strategies that need tuning."""
        cursor = con.execute('''
            SELECT 
                strategy, score, win_rate, sharpe_ratio, total_pnl_pct, 
                trades, max_drawdown_pct, window
            FROM strategy_scores
            WHERE enabled = 1 AND score < ? AND trades > 10
            ORDER BY score ASC
            LIMIT ?
        ''', (threshold_score, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    def _get_strategy_failures(self, con: sqlite3.Connection, strategy_name: str,
                              limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent losing trades for this strategy."""
        # First get signals from this strategy
        cursor = con.execute('''
            SELECT 
                s.id, s.ts_ms, s.ts_iso, s.symbol, s.signal, s.price, s.confidence,
                pt.pnl_pct, pt.exit_price, pt.status
            FROM signals s
            LEFT JOIN confirmed_signals cs ON cs.signal_id = s.id
            LEFT JOIN paper_trades pt ON pt.confirmed_signal_id = cs.id
            WHERE s.strategy = ? AND pt.pnl_pct < 0
            ORDER BY s.ts_ms DESC
            LIMIT ?
        ''', (strategy_name, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    def _get_strategy_code(self, strategy_name: str) -> Optional[str]:
        """Load strategy source code."""
        # Try to find strategy file
        possible_files = [
            self.strategies_dir / f"{strategy_name}.py",
            self.strategies_dir / f"{strategy_name.lower().replace(' ', '_')}.py"
        ]
        
        # Also check numbered strategies
        for path in self.strategies_dir.glob('strategy_*.py'):
            with open(path, 'r') as f:
                content = f.read()
                if f'self.name = "{strategy_name}"' in content or f"self.name = '{strategy_name}'" in content:
                    return content
        
        # Check direct matches
        for filepath in possible_files:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    return f.read()
        
        return None
    
    def _analyze_failure_patterns(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify common patterns in failures."""
        if not failures:
            return {'patterns': [], 'summary': 'No failures to analyze'}
        
        patterns = []
        
        # Analyze by signal type
        buy_failures = [f for f in failures if f['signal'] == 'BUY']
        sell_failures = [f for f in failures if f['signal'] == 'SELL']
        
        if buy_failures:
            avg_buy_loss = sum(f['pnl_pct'] for f in buy_failures) / len(buy_failures)
            patterns.append(f"BUY signals: {len(buy_failures)} failures, avg loss {avg_buy_loss:.2f}%")
        
        if sell_failures:
            avg_sell_loss = sum(f['pnl_pct'] for f in sell_failures) / len(sell_failures)
            patterns.append(f"SELL signals: {len(sell_failures)} failures, avg loss {avg_sell_loss:.2f}%")
        
        # Analyze by confidence
        low_conf_failures = [f for f in failures if f['confidence'] and f['confidence'] < 0.6]
        if low_conf_failures:
            patterns.append(f"Low confidence (<0.6): {len(low_conf_failures)} failures - may need confidence threshold adjustment")
        
        # Analyze timing
        hourly_distribution = {}
        for f in failures:
            hour = datetime.fromisoformat(f['ts_iso'].replace('Z', '+00:00')).hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        if hourly_distribution:
            worst_hour = max(hourly_distribution.items(), key=lambda x: x[1])
            if worst_hour[1] >= len(failures) * 0.3:  # 30% of failures in one hour
                patterns.append(f"Time clustering: {worst_hour[1]} failures at hour {worst_hour[0]}:00 UTC")
        
        return {
            'patterns': patterns,
            'total_failures': len(failures),
            'avg_loss': sum(f['pnl_pct'] for f in failures) / len(failures),
            'buy_failures': len(buy_failures),
            'sell_failures': len(sell_failures)
        }
    
    def _build_tuning_prompt(self, strategy_name: str, strategy_code: str,
                            performance: Dict[str, Any], failure_analysis: Dict[str, Any],
                            failures: List[Dict[str, Any]]) -> str:
        """Build prompt for Sonnet to suggest improvements."""
        prompt = f"""You are a quantitative trading strategy optimizer. Analyze this underperforming strategy and suggest parameter improvements.

STRATEGY NAME: {strategy_name}

CURRENT PERFORMANCE:
- Score: {performance['score']:.3f} (target: >0.6)
- Win Rate: {performance['win_rate']:.1f}%
- Sharpe Ratio: {performance['sharpe_ratio']:.2f}
- Max Drawdown: {performance.get('max_drawdown_pct', 'N/A')}%
- Total Trades: {performance['trades']}

FAILURE ANALYSIS:
"""
        for pattern in failure_analysis['patterns']:
            prompt += f"- {pattern}\n"
        
        prompt += f"""
STRATEGY CODE:
```python
{strategy_code}
```

TASK:
1. Identify specific parameters that may be causing poor performance
2. Suggest concrete parameter value changes
3. Explain the reasoning for each change
4. Return suggestions in this JSON format:

{{
  "parameter_changes": [
    {{
      "param_name": "example_threshold",
      "current_value": 0.5,
      "suggested_value": 0.65,
      "reasoning": "Increase threshold to filter out weak signals"
    }}
  ],
  "code_modifications": [
    {{
      "description": "Add volume filter",
      "change": "Check if volume > avg_volume * 1.5 before signaling",
      "reasoning": "Low volume periods show higher failure rate"
    }}
  ],
  "expected_improvement": "Brief explanation of expected impact"
}}

CRITICAL: Return ONLY valid JSON. No markdown, no code fences, no explanation text before or after the JSON. Start with {{ and end with }}.
"""
        return prompt
    
    def _call_sonnet(self, prompt: str) -> str:
        """Call Claude Sonnet via OpenClaw CLI with file-based prompt."""
        import tempfile
        import os
        try:
            # Write prompt to temp file to avoid CLI length limits
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name
            
            result = subprocess.run(
                ['openclaw', 'chat', '--model', 'sonnet', '--file', prompt_file],
                capture_output=True,
                text=True,
                timeout=120,  # 2 min timeout
                env={**os.environ, 'NO_COLOR': '1'}  # Disable ANSI colors
            )
            
            os.unlink(prompt_file)
            
            # Validate output
            if not result.stdout.strip():
                error_msg = f"Empty Sonnet output. stderr: {result.stderr}"
                print(f"WARNING: {error_msg}")
                return ""  # Return empty to trigger fallback
            
            if result.returncode != 0:
                error_msg = f"Sonnet call failed with code {result.returncode}. stderr: {result.stderr}"
                print(f"WARNING: {error_msg}")
                return ""
            
            return result.stdout.strip()
        except Exception as e:
            print(f"ERROR: Failed to call Sonnet: {e}")
            return ""
    
    def _parse_tuning_suggestions(self, sonnet_output: str) -> Optional[Dict[str, Any]]:
        """Parse JSON suggestions from Sonnet with robust error handling."""
        if not sonnet_output or not sonnet_output.strip():
            print("ERROR: Empty Sonnet output")
            return None
        
        text = sonnet_output.strip()
        
        # Log first 500 chars for debugging
        print(f"DEBUG: Sonnet output preview ({len(text)} chars): {text[:500]}...")
        
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find outermost { ... } with balanced braces
        try:
            depth = 0
            start_idx = None
            for i, ch in enumerate(text):
                if ch == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        candidate = text[start_idx:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # Try fixing common issues
                            pass
        except Exception:
            pass
        
        # Strategy 3: Fix common JSON issues and retry
        try:
            # Remove control characters except \n \r \t
            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
            # Fix trailing commas before } or ]
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
            # Fix single quotes to double quotes (careful with apostrophes)
            # Only do this if no double quotes exist in the content
            json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Extract key fields manually as fallback
        try:
            result = {"parameter_changes": [], "code_modifications": [], "expected_improvement": ""}
            
            # Try to find parameter_changes array
            pc_match = re.search(r'"parameter_changes"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
            if pc_match:
                # Try parsing just that array
                try:
                    result["parameter_changes"] = json.loads('[' + pc_match.group(1) + ']')
                except Exception:
                    pass
            
            ei_match = re.search(r'"expected_improvement"\s*:\s*"([^"]*)"', text)
            if ei_match:
                result["expected_improvement"] = ei_match.group(1)
            
            if result["parameter_changes"]:
                return result
        except Exception:
            pass
        
        # If all parsing fails, log the full output for debugging
        print(f"ERROR: Failed to parse Sonnet output after all strategies")
        print(f"Output length: {len(sonnet_output)} chars")
        print(f"Full output:\n{sonnet_output}")
        return None
    
    def _apply_parameter_changes(self, code: str, changes: List[Dict[str, Any]]) -> str:
        """Apply parameter changes to strategy code."""
        modified_code = code
        
        for change in changes:
            param_name = change['param_name']
            new_value = change['suggested_value']
            
            # Try to find and replace parameter value
            # Look for patterns like: param_name: value or "param_name": value
            patterns = [
                rf'(["\']?{param_name}["\']?\s*:\s*)([^\s,}}]+)',
                rf'(self\.params\[["\'{param_name}["\']\]\s*=\s*)([^\s,}}]+)',
            ]
            
            replaced = False
            for pattern in patterns:
                if re.search(pattern, modified_code):
                    modified_code = re.sub(pattern, rf'\g<1>{new_value}', modified_code)
                    replaced = True
                    break
            
            if not replaced:
                print(f"Warning: Could not auto-apply change to {param_name}")
        
        return modified_code
    
    def _save_tuned_strategy(self, strategy_name: str, code: str, version: int) -> Path:
        """Save tuned strategy version."""
        # Create filename with version
        base_name = strategy_name.lower().replace(' ', '_')
        filename = f"{base_name}_v{version}.py"
        filepath = self.strategies_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        filepath.chmod(0o755)
        return filepath
    
    def _log_tuning(self, con: sqlite3.Connection, strategy_name: str, 
                   suggestions: Dict[str, Any], filepath: str):
        """Log tuning activity to database."""
        ts_ms = int(datetime.utcnow().timestamp() * 1000)
        ts_iso = datetime.utcnow().isoformat() + 'Z'
        
        config = {
            'tuning_suggestions': suggestions,
            'tuned_file': str(filepath),
            'tuned_at': ts_iso
        }
        
        con.execute('''
            INSERT INTO strategy_configs (ts_ms, ts_iso, strategy, config_json, source, note)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ts_ms, ts_iso, strategy_name, json.dumps(config), 'tuner', 
              'Auto-tuned by Sonnet'))
        
        # Also log to knowledge base
        con.execute('''
            INSERT INTO knowledge_entries (ts_ms, ts_iso, category, strategy, content, source)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ts_ms, ts_iso, 'tuning', strategy_name, json.dumps(suggestions), 'sonnet_tuner'))
        
        con.commit()
    
    def tune_strategy(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Tune a specific underperforming strategy.
        
        Args:
            strategy_name: Name of strategy to tune
        
        Returns:
            Dict with tuning results, or None if tuning failed
        """
        con = self._connect()
        try:
            # Get performance data
            perf = con.execute('''
                SELECT * FROM strategy_scores 
                WHERE strategy = ? 
                ORDER BY ts_ms DESC 
                LIMIT 1
            ''', (strategy_name,)).fetchone()
            
            if not perf:
                print(f"No performance data for {strategy_name}")
                return None
            
            performance = dict(perf)
            
            # Get failures
            failures = self._get_strategy_failures(con, strategy_name)
            failure_analysis = self._analyze_failure_patterns(failures)
            
            # Get code
            code = self._get_strategy_code(strategy_name)
            if not code:
                print(f"Could not find code for {strategy_name}")
                return None
            
            # Build prompt and call Sonnet
            prompt = self._build_tuning_prompt(strategy_name, code, performance, 
                                               failure_analysis, failures)
            
            print(f"Calling Sonnet to tune {strategy_name}...")
            sonnet_output = self._call_sonnet(prompt)
            
            # Parse suggestions
            suggestions = self._parse_tuning_suggestions(sonnet_output)
            if not suggestions:
                print("Failed to parse tuning suggestions")
                return None
            
            # Apply changes
            if 'parameter_changes' in suggestions and suggestions['parameter_changes']:
                tuned_code = self._apply_parameter_changes(code, suggestions['parameter_changes'])
                
                # Save tuned version
                version = 2  # TODO: Increment based on existing versions
                filepath = self._save_tuned_strategy(strategy_name, tuned_code, version)
                
                # Log tuning
                self._log_tuning(con, strategy_name, suggestions, str(filepath))
                
                print(f"Strategy tuned and saved: {filepath}")
                
                return {
                    'strategy_name': strategy_name,
                    'filepath': str(filepath),
                    'suggestions': suggestions,
                    'old_performance': performance,
                    'failure_analysis': failure_analysis
                }
            else:
                print("No parameter changes suggested")
                return None
            
        except Exception as e:
            print(f"Tuning failed for {strategy_name}: {e}")
            return None
        finally:
            con.close()
    
    def tune_underperformers(self, max_strategies: int = 3) -> List[Dict[str, Any]]:
        """
        Identify and tune multiple underperforming strategies.
        
        Args:
            max_strategies: Maximum number of strategies to tune in one run
        
        Returns:
            List of tuning results
        """
        con = self._connect()
        try:
            underperformers = self._get_underperformers(con, limit=max_strategies)
            con.close()
            
            results = []
            for strat in underperformers:
                print(f"\nTuning {strat['strategy']} (score: {strat['score']:.3f})...")
                result = self.tune_strategy(strat['strategy'])
                if result:
                    results.append(result)
            
            return results
            
        finally:
            if con:
                con.close()


if __name__ == '__main__':
    # Quick test
    import os
    db_path = os.path.expanduser('~/.openclaw/workspace/blofin-stack/data/blofin_monitor.db')
    strategies_dir = os.path.expanduser('~/.openclaw/workspace/blofin-stack/strategies')
    
    tuner = StrategyTuner(db_path, strategies_dir)
    print("Testing strategy tuner (dry run)...")
    
    con = tuner._connect()
    underperformers = tuner._get_underperformers(con, limit=3)
    print(f"Found {len(underperformers)} underperformers")
    for u in underperformers:
        print(f"  - {u['strategy']}: score={u['score']:.3f}")
    con.close()
    
    print("\nTuner test complete!")
