"""
Technical analysis indicators (RSI, MACD, Stochastic, etc.)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    }


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return {
        'stoch_k': stoch_k,
        'stoch_d': stoch_d
    }


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate CCI (Commodity Channel Index)."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
    return williams_r


def compute_adx(df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
    """Calculate ADX (Average Directional Index)."""
    # Calculate +DM and -DM
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth the values
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    
    return {
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di
    }


def compute_adl(df: pd.DataFrame) -> pd.Series:
    """Calculate ADL (Accumulation/Distribution Line)."""
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    clv = clv.fillna(0)  # Handle division by zero when high == low
    adl = (clv * df['volume']).cumsum()
    return adl


def compute_technical_indicators(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Compute technical indicators from OHLCV dataframe.
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        params: Optional parameters dict
        
    Returns:
        DataFrame with computed features
    """
    params = params or {}
    result = df.copy()
    
    # RSI
    rsi_periods = params.get('rsi_periods', [14])
    for period in rsi_periods:
        result[f'rsi_{period}'] = compute_rsi(df['close'], period)
    
    # MACD
    macd_config = params.get('macd_config', [(12, 26, 9)])
    for fast, slow, signal in macd_config:
        macd_result = compute_macd(df['close'], fast, slow, signal)
        result[f'macd_{fast}_{slow}'] = macd_result['macd']
        result[f'macd_signal_{signal}'] = macd_result['macd_signal']
        result[f'macd_histogram'] = macd_result['macd_histogram']
    
    # Stochastic
    stoch_config = params.get('stoch_config', [(14, 3)])
    for k_period, d_period in stoch_config:
        stoch_result = compute_stochastic(df, k_period, d_period)
        result['stoch_k'] = stoch_result['stoch_k']
        result['stoch_d'] = stoch_result['stoch_d']
    
    # CCI
    cci_periods = params.get('cci_periods', [20])
    for period in cci_periods:
        result[f'cci_{period}'] = compute_cci(df, period)
    
    # Williams %R
    williams_periods = params.get('williams_periods', [14])
    for period in williams_periods:
        result[f'williams_r_{period}'] = compute_williams_r(df, period)
    
    # ADX
    adx_periods = params.get('adx_periods', [14])
    for period in adx_periods:
        adx_result = compute_adx(df, period)
        result[f'adx_{period}'] = adx_result['adx']
        result[f'plus_di_{period}'] = adx_result['plus_di']
        result[f'minus_di_{period}'] = adx_result['minus_di']
    
    # Moving Averages
    sma_periods = params.get('sma_periods', [50, 200])
    for period in sma_periods:
        result[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    
    ema_periods = params.get('ema_periods', [9, 21, 50, 200])
    for period in ema_periods:
        result[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # EMA crossovers
    if 9 in ema_periods and 21 in ema_periods:
        result['ema_9_21_cross'] = (result['ema_9'] > result['ema_21']).astype(int)
        result['ema_9_21_crossover'] = result['ema_9_21_cross'].diff()
    
    if 50 in ema_periods and 200 in ema_periods:
        result['ema_50_200_cross'] = (result['ema_50'] > result['ema_200']).astype(int)
        result['ema_50_200_crossover'] = result['ema_50_200_cross'].diff()
    
    # ADL
    result['adl'] = compute_adl(df)
    result['adl_ema_20'] = result['adl'].ewm(span=20, adjust=False).mean()
    
    return result


def get_available_features() -> list:
    """Return list of all available technical indicator features."""
    return [
        'rsi_14',
        'macd_12_26', 'macd_signal_9', 'macd_histogram',
        'stoch_k', 'stoch_d',
        'cci_20',
        'williams_r_14',
        'adx_14', 'plus_di_14', 'minus_di_14',
        'sma_50', 'sma_200',
        'ema_9', 'ema_21', 'ema_50', 'ema_200',
        'ema_9_21_cross', 'ema_9_21_crossover',
        'ema_50_200_cross', 'ema_50_200_crossover',
        'adl', 'adl_ema_20'
    ]
