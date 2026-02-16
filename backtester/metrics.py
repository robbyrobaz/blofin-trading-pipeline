#!/usr/bin/env python3
"""
Performance metrics calculation for backtesting.
Calculates win rate, P&L, Sharpe ratio, max drawdown, and composite score.
"""

import numpy as np
from typing import List, Dict, Any


def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate win rate (percentage of winning trades).
    
    Args:
        trades: List of trade dicts with 'pnl_pct' or 'pnl' key
    
    Returns:
        Win rate as float between 0 and 1
    """
    if not trades:
        return 0.0
    
    wins = sum(1 for t in trades if get_pnl(t) > 0)
    return wins / len(trades)


def calculate_avg_pnl_pct(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate average P&L percentage across all trades.
    
    Args:
        trades: List of trade dicts with 'pnl_pct' or 'pnl' key
    
    Returns:
        Average P&L as percentage (e.g., 1.5 for 1.5%)
    """
    if not trades:
        return 0.0
    
    pnls = [get_pnl(t) for t in trades]
    return float(np.mean(pnls))


def calculate_total_pnl_pct(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate total cumulative P&L percentage.
    
    Args:
        trades: List of trade dicts with 'pnl_pct' or 'pnl' key
    
    Returns:
        Total P&L as percentage
    """
    if not trades:
        return 0.0
    
    pnls = [get_pnl(t) for t in trades]
    return float(np.sum(pnls))


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).
    
    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Risk-free rate (default 0)
    
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    
    # Calculate excess returns
    excess_returns = returns_array - risk_free_rate
    
    # Calculate Sharpe ratio
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    # Annualize (assuming daily returns)
    sharpe = (mean_excess / std_excess) * np.sqrt(252)
    
    return float(sharpe)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: List of equity values over time
    
    Returns:
        Max drawdown as positive percentage (e.g., 15.5 for -15.5% drawdown)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    equity_array = np.array(equity_curve)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)
    
    # Calculate drawdown at each point
    drawdown = (running_max - equity_array) / running_max * 100
    
    # Return maximum drawdown
    max_dd = float(np.max(drawdown))
    
    return max_dd


def calculate_score(metrics: Dict[str, float]) -> float:
    """
    Calculate composite score from performance metrics.
    
    Formula: (win_rate * 40) + (avg_pnl_pct * 30) + (sharpe * 20) - (max_drawdown * 10)
    
    Args:
        metrics: Dict with keys: win_rate, avg_pnl_pct, sharpe_ratio, max_drawdown_pct
    
    Returns:
        Composite score between 0 and 100 (can be negative for very poor performance)
    """
    win_rate = metrics.get('win_rate', 0.0)
    avg_pnl_pct = metrics.get('avg_pnl_pct', 0.0)
    sharpe = metrics.get('sharpe_ratio', 0.0)
    max_dd = metrics.get('max_drawdown_pct', 0.0)
    
    # Apply formula
    score = (
        (win_rate * 40) +           # Win rate contribution (0-40 points)
        (avg_pnl_pct * 30) +        # Avg P&L contribution (can be negative)
        (sharpe * 20) -             # Sharpe contribution (can be negative)
        (max_dd * 10)               # Drawdown penalty (always reduces score)
    )
    
    # Clamp to reasonable range
    score = max(0, min(100, score))
    
    return float(score)


def get_pnl(trade: Dict[str, Any]) -> float:
    """
    Extract P&L from trade dict (handles both 'pnl_pct' and 'pnl' keys).
    
    Args:
        trade: Trade dict
    
    Returns:
        P&L as percentage
    """
    if 'pnl_pct' in trade:
        return float(trade['pnl_pct'])
    elif 'pnl' in trade:
        return float(trade['pnl'])
    else:
        return 0.0


def calculate_all_metrics(trades: List[Dict[str, Any]], equity_curve: List[float]) -> Dict[str, float]:
    """
    Calculate all performance metrics at once.
    
    Args:
        trades: List of completed trades
        equity_curve: Equity curve over time
    
    Returns:
        Dict with all metrics
    """
    # Calculate returns from trades
    returns = [get_pnl(t) / 100.0 for t in trades]  # Convert % to decimal
    
    metrics = {
        'win_rate': calculate_win_rate(trades),
        'avg_pnl_pct': calculate_avg_pnl_pct(trades),
        'total_pnl_pct': calculate_total_pnl_pct(trades),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown_pct': calculate_max_drawdown(equity_curve),
        'num_trades': len(trades),
        'num_wins': sum(1 for t in trades if get_pnl(t) > 0),
        'num_losses': sum(1 for t in trades if get_pnl(t) <= 0),
    }
    
    # Calculate composite score
    metrics['score'] = calculate_score(metrics)
    
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics into human-readable string.
    
    Args:
        metrics: Metrics dict from calculate_all_metrics()
    
    Returns:
        Formatted string
    """
    lines = [
        f"Trades: {metrics.get('num_trades', 0)} ({metrics.get('num_wins', 0)}W / {metrics.get('num_losses', 0)}L)",
        f"Win Rate: {metrics.get('win_rate', 0):.2%}",
        f"Avg P&L: {metrics.get('avg_pnl_pct', 0):.2f}%",
        f"Total P&L: {metrics.get('total_pnl_pct', 0):.2f}%",
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
        f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%",
        f"Score: {metrics.get('score', 0):.1f}/100"
    ]
    
    return "\n".join(lines)


def is_profitable(metrics: Dict[str, float], min_win_rate: float = 0.5, min_sharpe: float = 0.5) -> bool:
    """
    Determine if strategy is profitable based on metrics.
    
    Args:
        metrics: Metrics dict
        min_win_rate: Minimum win rate threshold (default 0.5)
        min_sharpe: Minimum Sharpe ratio threshold (default 0.5)
    
    Returns:
        True if profitable, False otherwise
    """
    return (
        metrics.get('win_rate', 0) >= min_win_rate and
        metrics.get('sharpe_ratio', 0) >= min_sharpe and
        metrics.get('total_pnl_pct', 0) > 0
    )
