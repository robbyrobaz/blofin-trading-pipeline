#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from strategies import BaseStrategy, Signal, get_all_strategies


class StrategyManager:
    """Manages loading, enabling/disabling, and execution of trading strategies."""
    
    def __init__(self, strategies_dir: Optional[str] = None):
        """
        Initialize strategy manager.
        
        Args:
            strategies_dir: Path to strategies directory (unused, for future dynamic loading)
        """
        self.strategies: Dict[str, BaseStrategy] = {}  # name -> strategy instance
        self.enabled: Dict[str, bool] = {}  # name -> enabled status
        self.scores: Dict[str, Dict[str, float]] = {}  # name -> {symbol -> score}
        self.last_signals: Dict[Tuple[str, str, str], int] = {}  # (symbol, signal, strategy) -> ts_ms
        
        # Cooldown between same signal type from same strategy
        self.signal_cooldown_ms = int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "240")) * 1000
        
        self._load_strategies()
    
    def _load_strategies(self) -> None:
        """Load all available strategies."""
        try:
            all_strategies = get_all_strategies()
            
            for strategy in all_strategies:
                self.strategies[strategy.name] = strategy
                self.enabled[strategy.name] = True  # All enabled by default
                self.scores[strategy.name] = {}
                
            print(f"[StrategyManager] Loaded {len(self.strategies)} strategies: {list(self.strategies.keys())}")
        except Exception as e:
            print(f"[StrategyManager] Error loading strategies: {e}")
            raise
    
    def detect_all(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> List[Signal]:
        """
        Run all enabled strategies and return detected signals.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            volume: Current volume
            ts_ms: Current timestamp in milliseconds
            prices: Historical price data [(ts_ms, price), ...]
            volumes: Historical volume data [(ts_ms, volume), ...]
        
        Returns:
            List of Signal objects from all enabled strategies
        """
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            if not self.enabled.get(strategy_name, False):
                continue
            
            try:
                signal = strategy.detect(symbol, price, volume, ts_ms, prices, volumes)
                
                if signal is not None:
                    # Check cooldown
                    key = (symbol, signal.signal, strategy_name)
                    last_ts = self.last_signals.get(key, 0)
                    
                    if ts_ms - last_ts >= self.signal_cooldown_ms:
                        signals.append(signal)
                        self.last_signals[key] = ts_ms
                    
            except Exception as e:
                print(f"[StrategyManager] Error in {strategy_name}: {e}")
                continue
        
        return signals
    
    def enable(self, name: str) -> bool:
        """
        Enable a strategy.
        
        Args:
            name: Strategy name
        
        Returns:
            True if successful, False otherwise
        """
        if name in self.strategies:
            self.enabled[name] = True
            print(f"[StrategyManager] Enabled strategy: {name}")
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """
        Disable a strategy.
        
        Args:
            name: Strategy name
        
        Returns:
            True if successful, False otherwise
        """
        if name in self.strategies:
            self.enabled[name] = False
            print(f"[StrategyManager] Disabled strategy: {name}")
            return True
        return False
    
    def get_performance(self, name: str) -> Dict:
        """
        Get performance metrics for a strategy.
        
        Args:
            name: Strategy name
        
        Returns:
            Dictionary with performance data
        """
        if name not in self.strategies:
            return {}
        
        return {
            "name": name,
            "enabled": self.enabled.get(name, False),
            "scores": self.scores.get(name, {}),
        }
    
    def update_strategy_config(self, name: str, params: Dict) -> bool:
        """
        Update strategy configuration parameters.
        
        Args:
            name: Strategy name
            params: Dictionary of parameter updates
        
        Returns:
            True if successful, False otherwise
        """
        if name not in self.strategies:
            return False
        
        try:
            strategy = self.strategies[name]
            strategy.update_config(params)
            print(f"[StrategyManager] Updated config for {name}: {params}")
            return True
        except Exception as e:
            print(f"[StrategyManager] Error updating {name} config: {e}")
            return False
    
    def get_strategy_config(self, name: str) -> Optional[Dict]:
        """
        Get current configuration for a strategy.
        
        Args:
            name: Strategy name
        
        Returns:
            Dictionary of current config parameters, or None if strategy not found
        """
        if name not in self.strategies:
            return None
        
        return self.strategies[name].get_config()
    
    def list_strategies(self) -> List[Dict]:
        """
        List all strategies with their status.
        
        Returns:
            List of dictionaries with strategy info
        """
        result = []
        for name, strategy in self.strategies.items():
            result.append({
                "name": name,
                "version": strategy.version,
                "description": strategy.description,
                "enabled": self.enabled.get(name, False),
                "config": strategy.get_config(),
            })
        return result
