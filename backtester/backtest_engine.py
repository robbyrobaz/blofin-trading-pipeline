#!/usr/bin/env python3
"""
Backtesting engine for strategies and ML models.
Loads historical data and simulates strategy/model execution.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
import json

from .aggregator import aggregate_ticks_to_1m_ohlcv, aggregate_ohlcv, timeframe_to_minutes
from .metrics import calculate_all_metrics


class BacktestEngine:
    """
    Core backtesting engine.
    
    Usage:
        engine = BacktestEngine(symbol='BTC-USDT', days_back=7)
        results = engine.run_strategy(strategy, timeframe='5m')
        print(results['metrics'])
    """
    
    def __init__(
        self,
        symbol: str,
        days_back: int = 7,
        db_path: str = 'data/blofin_monitor.db',
        initial_capital: float = 10000.0,
        limit_rows: int = None
    ):
        """
        Initialize backtester.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            days_back: Number of days of historical data to load
            db_path: Path to SQLite database
            initial_capital: Starting capital for backtest
        """
        self.symbol = symbol
        self.days_back = days_back
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.limit_rows = limit_rows
        
        # Load historical data
        self.ticks = self._load_ticks()
        self.ohlcv_1m = aggregate_ticks_to_1m_ohlcv(self.ticks)
        
    def _load_ticks(self) -> List[Dict[str, Any]]:
        """Load tick data from database."""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        
        # Calculate timestamp range
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = int((datetime.now() - timedelta(days=self.days_back)).timestamp() * 1000)
        
        # Query ticks
        query = '''
            SELECT ts_ms, price
            FROM ticks
            WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ?
            ORDER BY ts_ms ASC
        '''
        if self.limit_rows:
            query += f' LIMIT {self.limit_rows}'
        
        cur = con.execute(query, (self.symbol, start_ts, end_ts))
        
        ticks = [dict(row) for row in cur.fetchall()]
        con.close()
        
        return ticks
    
    def get_ohlcv(self, timeframe: str = '1m') -> List[Dict[str, Any]]:
        """
        Get OHLCV data for specified timeframe.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '15m', '1h')
        
        Returns:
            List of OHLCV candles
        """
        if timeframe == '1m':
            return self.ohlcv_1m
        
        minutes = timeframe_to_minutes(timeframe)
        return aggregate_ohlcv(self.ohlcv_1m, minutes)
    
    def run_strategy(
        self,
        strategy: Any,
        timeframe: str = '5m',
        position_size_pct: float = 100.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run strategy on historical data.
        
        Args:
            strategy: Strategy object with detect() method
            timeframe: Timeframe to run strategy on
            position_size_pct: Position size as percentage of capital (default 100%)
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)
        
        Returns:
            Dict with trades, equity_curve, and metrics
        """
        # Get candles for timeframe
        candles = self.get_ohlcv(timeframe)
        
        if not candles:
            return self._empty_result("No candles available")
        
        # Initialize backtest state
        trades = []
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        open_position = None
        
        # Run strategy on each candle
        for i, candle in enumerate(candles):
            # Check if we have an open position
            if open_position:
                # Check stop loss
                if stop_loss_pct and candle['low'] <= open_position['stop_loss']:
                    # Stop loss hit
                    exit_price = open_position['stop_loss']
                    pnl_pct = ((exit_price - open_position['entry_price']) / open_position['entry_price']) * 100
                    if open_position['side'] == 'SHORT':
                        pnl_pct = -pnl_pct
                    
                    trades.append({
                        'entry_ts': open_position['entry_ts'],
                        'exit_ts': candle['ts_ms'],
                        'side': open_position['side'],
                        'entry_price': open_position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'reason': 'stop_loss'
                    })
                    
                    current_capital *= (1 + pnl_pct / 100)
                    equity_curve.append(current_capital)
                    open_position = None
                    continue
                
                # Check take profit
                if take_profit_pct and candle['high'] >= open_position['take_profit']:
                    # Take profit hit
                    exit_price = open_position['take_profit']
                    pnl_pct = ((exit_price - open_position['entry_price']) / open_position['entry_price']) * 100
                    if open_position['side'] == 'SHORT':
                        pnl_pct = -pnl_pct
                    
                    trades.append({
                        'entry_ts': open_position['entry_ts'],
                        'exit_ts': candle['ts_ms'],
                        'side': open_position['side'],
                        'entry_price': open_position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'reason': 'take_profit'
                    })
                    
                    current_capital *= (1 + pnl_pct / 100)
                    equity_curve.append(current_capital)
                    open_position = None
                    continue
            
            # Get features/context for strategy (last N candles)
            lookback = min(i + 1, 100)  # Max 100 candles lookback
            context_candles = candles[max(0, i - lookback + 1):i + 1]
            
            # Run strategy detection
            try:
                signal = strategy.detect(context_candles, self.symbol)
            except Exception as e:
                # Strategy error, skip this candle
                continue
            
            # Handle signal
            if signal and not open_position:
                # Open new position
                entry_price = candle['close']
                side = signal.get('signal', 'BUY').upper()
                
                open_position = {
                    'entry_ts': candle['ts_ms'],
                    'entry_price': entry_price,
                    'side': side,
                    'stop_loss': entry_price * (1 - (stop_loss_pct or 5) / 100) if side == 'LONG' else entry_price * (1 + (stop_loss_pct or 5) / 100),
                    'take_profit': entry_price * (1 + (take_profit_pct or 10) / 100) if side == 'LONG' else entry_price * (1 - (take_profit_pct or 10) / 100)
                }
            
            elif signal and open_position:
                # Exit signal received
                signal_type = signal.get('signal', '').upper()
                
                # Check if opposite signal (exit)
                should_exit = False
                if open_position['side'] == 'LONG' and signal_type == 'SELL':
                    should_exit = True
                elif open_position['side'] == 'SHORT' and signal_type == 'BUY':
                    should_exit = True
                
                if should_exit:
                    exit_price = candle['close']
                    pnl_pct = ((exit_price - open_position['entry_price']) / open_position['entry_price']) * 100
                    if open_position['side'] == 'SHORT':
                        pnl_pct = -pnl_pct
                    
                    trades.append({
                        'entry_ts': open_position['entry_ts'],
                        'exit_ts': candle['ts_ms'],
                        'side': open_position['side'],
                        'entry_price': open_position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'reason': 'signal'
                    })
                    
                    current_capital *= (1 + pnl_pct / 100)
                    equity_curve.append(current_capital)
                    open_position = None
        
        # Close any open position at end of backtest
        if open_position:
            exit_price = candles[-1]['close']
            pnl_pct = ((exit_price - open_position['entry_price']) / open_position['entry_price']) * 100
            if open_position['side'] == 'SHORT':
                pnl_pct = -pnl_pct
            
            trades.append({
                'entry_ts': open_position['entry_ts'],
                'exit_ts': candles[-1]['ts_ms'],
                'side': open_position['side'],
                'entry_price': open_position['entry_price'],
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'reason': 'backtest_end'
            })
            
            current_capital *= (1 + pnl_pct / 100)
            equity_curve.append(current_capital)
        
        # Calculate metrics
        metrics = calculate_all_metrics(trades, equity_curve) if trades else {}
        
        return {
            'symbol': self.symbol,
            'timeframe': timeframe,
            'days_back': self.days_back,
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'final_capital': current_capital,
            'num_candles': len(candles)
        }
    
    def run_model(
        self,
        model: Any,
        timeframe: str = '1m',
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run ML model on historical data.
        
        Args:
            model: ML model with predict() method
            timeframe: Timeframe to run model on
            threshold: Prediction threshold for classification
        
        Returns:
            Dict with predictions, accuracy, and metrics
        """
        # Get candles
        candles = self.get_ohlcv(timeframe)
        
        if not candles:
            return self._empty_result("No candles available")
        
        predictions = []
        actuals = []
        
        # Run model on each candle
        for i in range(len(candles) - 1):  # Skip last candle (no actual to compare)
            # Get features/context
            lookback = min(i + 1, 100)
            context_candles = candles[max(0, i - lookback + 1):i + 1]
            
            # Predict
            try:
                pred = model.predict(context_candles, self.symbol)
                pred_class = 1 if pred >= threshold else 0
                predictions.append(pred_class)
                
                # Actual: did price go up in next candle?
                actual = 1 if candles[i + 1]['close'] > candles[i]['close'] else 0
                actuals.append(actual)
            except Exception as e:
                continue
        
        # Calculate accuracy metrics
        if predictions:
            accuracy = sum(p == a for p, a in zip(predictions, actuals)) / len(predictions)
            
            # Calculate precision, recall, F1
            tp = sum(p == 1 and a == 1 for p, a in zip(predictions, actuals))
            fp = sum(p == 1 and a == 0 for p, a in zip(predictions, actuals))
            fn = sum(p == 0 and a == 1 for p, a in zip(predictions, actuals))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            accuracy = precision = recall = f1 = 0
        
        return {
            'symbol': self.symbol,
            'timeframe': timeframe,
            'days_back': self.days_back,
            'num_predictions': len(predictions),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def _empty_result(self, reason: str) -> Dict[str, Any]:
        """Return empty result with error reason."""
        return {
            'symbol': self.symbol,
            'error': reason,
            'trades': [],
            'equity_curve': [self.initial_capital],
            'metrics': {}
        }
