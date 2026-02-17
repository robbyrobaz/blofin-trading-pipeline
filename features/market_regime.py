"""
Market regime detection and classification.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def detect_trend(df: pd.DataFrame, lookback: int = 50) -> Tuple[bool, bool, float]:
    """
    Detect if market is trending up or down.
    
    Returns:
        (is_trending_up, is_trending_down, trend_strength)
    """
    if len(df) < lookback:
        return False, False, 0.0
    
    # Use linear regression slope to detect trend
    recent = df['close'].tail(lookback)
    
    # Check for NaN or insufficient data
    if recent.isna().all() or len(recent.dropna()) < 2:
        return False, False, 0.0
    
    x = np.arange(len(recent))
    y = recent.values
    
    # Remove NaN values for regression
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 2:
        return False, False, 0.0
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    
    # Normalize slope by price with protection
    mean_price = recent.mean()
    if mean_price == 0 or np.isnan(mean_price):
        return False, False, 0.0
    
    normalized_slope = (slope / mean_price) * 100
    
    # Calculate R-squared to measure trend strength
    y_pred = slope * x_valid + intercept
    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
    
    # Protect against division by zero
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    r_squared = max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]
    
    # Trend detected if slope is significant and R-squared is high
    is_trending_up = normalized_slope > 0.1 and r_squared > 0.5
    is_trending_down = normalized_slope < -0.1 and r_squared > 0.5
    
    return is_trending_up, is_trending_down, r_squared


def detect_ranging(df: pd.DataFrame, lookback: int = 50, threshold_pct: float = 2.0) -> bool:
    """
    Detect if market is ranging (sideways movement).
    
    Returns:
        True if market is ranging
    """
    if len(df) < lookback:
        return False
    
    recent = df['close'].tail(lookback)
    
    # Check for NaN or insufficient data
    if recent.isna().all() or len(recent.dropna()) < 2:
        return False
    
    # Calculate range as percentage of mean price with protection
    mean_price = recent.mean()
    if mean_price == 0 or np.isnan(mean_price):
        return False
    
    price_range = (recent.max() - recent.min()) / mean_price * 100
    
    # Check if price is oscillating within a tight range
    is_ranging = price_range < threshold_pct
    
    return is_ranging


def detect_volatile(df: pd.DataFrame, lookback: int = 20, threshold_pct: float = 2.5) -> bool:
    """
    Detect if market is volatile.
    
    Returns:
        True if market is volatile
    """
    if len(df) < lookback:
        return False
    
    recent = df['close'].tail(lookback)
    
    # Check for NaN or insufficient data
    if recent.isna().all() or len(recent.dropna()) < 2:
        return False
    
    # Calculate volatility as standard deviation percentage with protection
    mean_price = recent.mean()
    std_price = recent.std()
    
    if mean_price == 0 or np.isnan(mean_price) or np.isnan(std_price):
        return False
    
    volatility = (std_price / mean_price) * 100
    
    is_volatile = volatility > threshold_pct
    
    return is_volatile


def compute_regime_features(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Compute market regime features from OHLCV dataframe.
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        params: Optional parameters dict
        
    Returns:
        DataFrame with computed features
    """
    params = params or {}
    result = df.copy()
    
    trend_lookback = params.get('trend_lookback', 50)
    ranging_lookback = params.get('ranging_lookback', 50)
    volatile_lookback = params.get('volatile_lookback', 20)
    ranging_threshold = params.get('ranging_threshold_pct', 2.0)
    volatile_threshold = params.get('volatile_threshold_pct', 2.5)
    
    # Initialize regime arrays
    is_trending_up_arr = []
    is_trending_down_arr = []
    is_ranging_arr = []
    is_volatile_arr = []
    trend_strength_arr = []
    regime_type_arr = []
    regime_strength_arr = []
    
    # Calculate regime for each row (using data up to that point)
    for i in range(len(df)):
        if i < max(trend_lookback, ranging_lookback, volatile_lookback):
            # Not enough data yet
            is_trending_up_arr.append(False)
            is_trending_down_arr.append(False)
            is_ranging_arr.append(False)
            is_volatile_arr.append(False)
            trend_strength_arr.append(0.0)
            regime_type_arr.append('unknown')
            regime_strength_arr.append(0.0)
        else:
            # Get data up to current point
            df_slice = df.iloc[:i+1]
            
            # Detect regimes
            trending_up, trending_down, trend_str = detect_trend(df_slice, trend_lookback)
            ranging = detect_ranging(df_slice, ranging_lookback, ranging_threshold)
            volatile = detect_volatile(df_slice, volatile_lookback, volatile_threshold)
            
            is_trending_up_arr.append(trending_up)
            is_trending_down_arr.append(trending_down)
            is_ranging_arr.append(ranging)
            is_volatile_arr.append(volatile)
            trend_strength_arr.append(trend_str)
            
            # Determine primary regime type (priority: volatile > trending > ranging)
            if volatile:
                regime = 'volatile'
                # Strength calculation with protection
                recent_slice = df_slice['close'].tail(volatile_lookback)
                mean_price = recent_slice.mean()
                std_price = recent_slice.std()
                
                if mean_price > 0 and not np.isnan(mean_price) and not np.isnan(std_price):
                    strength = min(1.0, (std_price / mean_price) / volatile_threshold)
                else:
                    strength = 0.0
                    
            elif trending_up:
                regime = 'trending_up'
                strength = trend_str
            elif trending_down:
                regime = 'trending_down'
                strength = trend_str
            elif ranging:
                regime = 'ranging'
                # Strength is inverse of range (tighter range = stronger ranging regime)
                recent_slice = df_slice['close'].tail(ranging_lookback)
                mean_price = recent_slice.mean()
                
                if mean_price > 0 and not np.isnan(mean_price):
                    price_range = (recent_slice.max() - recent_slice.min()) / mean_price * 100
                    strength = max(0.0, 1.0 - (price_range / ranging_threshold))
                else:
                    strength = 0.0
            else:
                regime = 'neutral'
                strength = 0.0
            
            regime_type_arr.append(regime)
            regime_strength_arr.append(strength)
    
    # Add to result
    result['is_trending_up'] = is_trending_up_arr
    result['is_trending_down'] = is_trending_down_arr
    result['is_ranging'] = is_ranging_arr
    result['is_volatile'] = is_volatile_arr
    result['trend_strength'] = trend_strength_arr
    result['regime_type'] = regime_type_arr
    result['regime_strength'] = regime_strength_arr
    
    # Additional features
    # Regime duration (how many bars in current regime)
    regime_duration = [1]
    for i in range(1, len(regime_type_arr)):
        if regime_type_arr[i] == regime_type_arr[i-1]:
            regime_duration.append(regime_duration[-1] + 1)
        else:
            regime_duration.append(1)
    result['regime_duration'] = regime_duration
    
    # Regime transition indicator
    regime_transition = [0]
    for i in range(1, len(regime_type_arr)):
        regime_transition.append(1 if regime_type_arr[i] != regime_type_arr[i-1] else 0)
    result['regime_transition'] = regime_transition
    
    return result


def get_regime_type(df: pd.DataFrame, lookback: int = 50) -> str:
    """
    Get current market regime type.
    
    Returns:
        One of: "trending_up", "trending_down", "ranging", "volatile", "neutral"
    """
    trending_up, trending_down, _ = detect_trend(df, lookback)
    ranging = detect_ranging(df, lookback)
    volatile = detect_volatile(df, lookback)
    
    if volatile:
        return "volatile"
    elif trending_up:
        return "trending_up"
    elif trending_down:
        return "trending_down"
    elif ranging:
        return "ranging"
    else:
        return "neutral"


def get_regime_strength(df: pd.DataFrame, lookback: int = 50) -> float:
    """
    Get strength of current market regime (0.0 - 1.0).
    
    Returns:
        Strength value between 0.0 and 1.0
    """
    regime = get_regime_type(df, lookback)
    
    if regime == "volatile":
        recent = df['close'].tail(lookback)
        mean_price = recent.mean()
        std_price = recent.std()
        
        if mean_price > 0 and not np.isnan(mean_price) and not np.isnan(std_price):
            return min(1.0, (std_price / mean_price) / 0.025)
        else:
            return 0.0
            
    elif regime in ["trending_up", "trending_down"]:
        _, _, strength = detect_trend(df, lookback)
        return strength
        
    elif regime == "ranging":
        recent = df['close'].tail(lookback)
        mean_price = recent.mean()
        
        if mean_price > 0 and not np.isnan(mean_price):
            price_range = (recent.max() - recent.min()) / mean_price * 100
            return max(0.0, 1.0 - (price_range / 2.0))
        else:
            return 0.0
    else:
        return 0.0


def get_available_features() -> list:
    """Return list of all available market regime features."""
    return [
        'is_trending_up', 'is_trending_down', 'is_ranging', 'is_volatile',
        'trend_strength', 'regime_type', 'regime_strength',
        'regime_duration', 'regime_transition'
    ]
