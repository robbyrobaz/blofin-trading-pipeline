# Feature Library Guide

## Overview

The feature library is the **single source of truth for all features** used by strategies and ML models.

Instead of each strategy/model reimplementing indicators, they request features from `feature_manager.py`. This ensures:

- **Consistency** — Same RSI across all strategies
- **Reusability** — ML can tap any technical indicator
- **Extensibility** — Add a feature once, use everywhere
- **Performance** — Cache expensive calculations

---

## Directory Structure

```
features/
├── __init__.py
├── feature_manager.py       # Central API
├── price_features.py        # OHLC, returns, momentum
├── volume_features.py       # Volume-based
├── technical_indicators.py  # RSI, MACD, Bollinger, ATR, etc.
├── volatility_features.py   # Volatility measures
├── market_microstructure.py # Spreads, imbalance, etc.
├── market_regime.py         # Trending, ranging, volatile
├── custom_features.py       # User-defined combinations
└── tests/
    └── test_features.py     # Unit tests for all features
```

---

## Core API: FeatureManager

```python
from features.feature_manager import FeatureManager

fm = FeatureManager()

# Get specific features
df = fm.get_features(
    symbol='BTC-USDT',
    timeframe='5m',
    feature_list=['close', 'rsi_14', 'volume_sma_20'],
    lookback_bars=100
)
# Returns: pandas DataFrame with 100 rows, 3 columns

# Get feature groups
df = fm.get_features(
    symbol='BTC-USDT',
    timeframe='1m',
    feature_group='momentum',  # All momentum features
    lookback_bars=50
)

# List available features
features = fm.list_available_features()
# Returns: dict with all feature names + descriptions

# Get recommended features for a use case
features = fm.get_features_for_use_case('mean_reversion')
# Returns: ['close', 'vwap', 'volatility', 'volume_sma_20']
```

---

## Available Features by Category

### Price Features (`price_features.py`)

| Feature | Description | Params | Example |
|---------|-------------|--------|---------|
| `close` | Closing price | - | 45230.50 |
| `open` | Opening price | - | 45100.00 |
| `high` | Highest price | - | 45500.00 |
| `low` | Lowest price | - | 45000.00 |
| `hl2` | (High + Low) / 2 | - | 45250.00 |
| `hlc3` | (High + Low + Close) / 3 | - | 45260.17 |
| `returns` | % change from previous close | - | 0.0031 (0.31%) |
| `log_returns` | log(close / prev_close) | - | 0.00309 |
| `momentum_n` | Close - Close[n bars ago] | n=10 | 234.50 |
| `roc_n` | (Close - Close[n bars ago]) / Close[n bars ago] | n=12 | 0.0052 |

### Volume Features (`volume_features.py`)

| Feature | Description | Params | Example |
|---------|-------------|--------|---------|
| `volume` | Raw volume | - | 1234567 |
| `volume_sma_n` | Volume moving average | n=20 | 890000 |
| `volume_ema_n` | Volume exponential MA | n=9 | 920000 |
| `volume_ratio` | Current vol / avg vol | - | 1.45 |
| `volume_surge` | Is volume > 2x average? | threshold=2.0 | True/False |
| `vwap` | Volume-weighted avg price | - | 45220.30 |
| `vwap_deviation` | (Close - VWAP) / VWAP | - | 0.0025 |
| `on_balance_volume` | Cumulative volume indicator | - | 2345678 |
| `money_flow` | Volume * Price | - | 55890123450 |

### Momentum Indicators (`technical_indicators.py`)

| Feature | Description | Params | Example |
|---------|-------------|--------|---------|
| `rsi_n` | Relative Strength Index | n=14 | 65.3 (0-100) |
| `macd_n1_n2` | MACD line | n1=12, n2=26 | 234.50 |
| `macd_signal_n` | MACD signal line | n=9 | 220.00 |
| `macd_histogram` | MACD - Signal | - | 14.50 |
| `stoch_k_n` | Stochastic K line | n=14 | 75.2 (0-100) |
| `stoch_d_n` | Stochastic D line | n=3 | 72.1 (0-100) |
| `cci_n` | Commodity Channel Index | n=20 | 142.5 |
| `williams_r_n` | Williams %R | n=14 | -15.3 (-100 to 0) |
| `adx_n` | Average Directional Index | n=14 | 42.1 (0-100) |

### Volatility Measures (`volatility_features.py`)

| Feature | Description | Params | Example |
|---------|-------------|--------|---------|
| `atr_n` | Average True Range | n=14 | 450.25 |
| `atr_pct_n` | ATR as % of price | n=14 | 0.0099 (0.99%) |
| `std_dev_n` | Standard deviation | n=20 | 234.50 |
| `std_dev_pct_n` | Std dev as % of price | n=20 | 0.0051 (0.51%) |
| `bbands_upper_n` | Bollinger Band upper | n=20 | 46100.00 |
| `bbands_lower_n` | Bollinger Band lower | n=20 | 44300.00 |
| `bbands_width` | Upper - Lower | - | 1800.00 |
| `bbands_width_pct` | (Upper - Lower) / Mid | - | 0.0196 (1.96%) |
| `keltner_upper_n` | Keltner upper | n=20 | 46050.00 |
| `keltner_lower_n` | Keltner lower | n=20 | 44350.00 |
| `historical_volatility_n` | Rolling std dev of returns | n=20 | 0.0145 (1.45% annualized) |

### Trend Indicators (`technical_indicators.py`)

| Feature | Description | Params | Example |
|---------|-------------|--------|---------|
| `ema_n` | Exponential Moving Average | n=9, 21, 50, 200 | 45180.00 |
| `sma_n` | Simple Moving Average | n=50, 200 | 45220.00 |
| `ema_fast_ema_slow` | EMA crossover (1=above, 0=below) | fast=9, slow=21 | 1 |
| `slope_ema_n` | EMA slope (trend strength) | n=20 | 0.0034 |
| `trend_direction_n` | Is uptrend or downtrend? | n=20 | "up" / "down" |
| `higher_high_n` | Is price making higher highs? | n=20 | True/False |
| `lower_low_n` | Is price making lower lows? | n=20 | True/False |

### Market Regime (`market_regime.py`)

| Feature | Description | Params | Example |
|---------|-------------|--------|---------|
| `is_trending_up` | Strong uptrend detected | lookback=50 | True/False |
| `is_trending_down` | Strong downtrend detected | lookback=50 | True/False |
| `is_ranging` | Range-bound market | lookback=50 | True/False |
| `is_volatile` | High volatility | threshold_pct=2.5 | True/False |
| `regime_type` | Market classification | - | "trending_up" / "ranging" / "volatile" |
| `regime_strength` | How strong is the regime? | - | 0.0-1.0 |
| `volatility_regime` | Calm, normal, or elevated? | - | "normal" |

### Custom/Computed Features (`custom_features.py`)

User-defined combinations:

```python
# Example: Mean reversion indicator
# = distance from VWAP relative to volatility
fm.register_custom_feature(
    name='vwap_zscore',
    formula='(close - vwap) / atr_14',
    dependencies=['close', 'vwap', 'atr_14']
)

# Now use it like any other feature
df = fm.get_features(..., feature_list=['vwap_zscore', ...])
```

---

## Usage Examples

### Strategy Using Feature Manager

```python
from features.feature_manager import FeatureManager

class MeanReversionStrategy:
    def __init__(self):
        self.fm = FeatureManager()
    
    def detect(self, symbol, current_price, ts_ms, ...):
        # Get features needed for this strategy
        df = self.fm.get_features(
            symbol=symbol,
            timeframe='5m',
            feature_list=['close', 'vwap', 'atr_14', 'is_ranging'],
            lookback_bars=50
        )
        
        latest = df.iloc[-1]
        
        # Entry logic
        if latest['is_ranging'] and latest['close'] < latest['vwap'] - latest['atr_14']:
            return Signal(signal='BUY', confidence=0.75)
        
        return None
```

### ML Model Using Feature Manager

```python
import pandas as pd
from features.feature_manager import FeatureManager
from sklearn.ensemble import RandomForestClassifier

class DirectionPredictor:
    def __init__(self):
        self.fm = FeatureManager()
        self.model = None
    
    def train(self, symbols, date_range):
        features = self.fm.get_features_for_use_case('direction_prediction')
        
        # Collect training data
        X = []
        y = []
        for symbol in symbols:
            df = self.fm.get_features(
                symbol=symbol,
                timeframe='1m',
                feature_list=features,
                lookback_bars=10000
            )
            # df has all features; compute labels (up/down)
            X.append(df)
            y.append((df['close'].shift(-1) > df['close']).astype(int))
        
        X = pd.concat(X)
        y = pd.concat(y)
        
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X, y)
    
    def predict(self, symbol, timeframe='1m'):
        df = self.fm.get_features(
            symbol=symbol,
            timeframe=timeframe,
            feature_list=self.features,
            lookback_bars=100
        )
        return self.model.predict(df.iloc[-1:])
```

---

## Adding a New Feature

### Step 1: Implement in appropriate module

```python
# features/custom_indicators.py

def calculate_my_indicator(df, param1, param2):
    """
    Calculate my custom indicator.
    
    Args:
        df: DataFrame with OHLCV columns
        param1: first parameter
        param2: second parameter
    
    Returns:
        Series with indicator values
    """
    result = (df['close'] * param1) / df['volume'].rolling(param2).mean()
    return result
```

### Step 2: Register in feature_manager

```python
# features/feature_manager.py

class FeatureManager:
    def __init__(self):
        self.features = {
            # ... existing ...
            'my_indicator_p1_p2': {
                'func': custom_indicators.calculate_my_indicator,
                'params': {'param1': 1.5, 'param2': 20},
                'category': 'custom',
                'description': 'What it does'
            }
        }
```

### Step 3: Test

```python
# features/tests/test_features.py

def test_my_indicator():
    fm = FeatureManager()
    df = fm.get_features('BTC-USDT', '5m', ['my_indicator_p1_p2'], 50)
    assert df is not None
    assert 'my_indicator_p1_p2' in df.columns
    assert not df['my_indicator_p1_p2'].isna().all()
```

### Step 4: Document

Add to FEATURE_LIBRARY.md and list_available_features() output.

---

## Performance Tips

1. **Request only needed features** — Don't ask for all momentum features if you only need RSI
2. **Use feature groups** — `fm.get_features(..., feature_group='volatility')` instead of listing each
3. **Lookback optimization** — Use minimum lookback_bars needed (50 usually sufficient)
4. **Caching** — Feature manager caches features internally; reuse `fm` across calls
5. **Batch requests** — Requesting 10 features once is faster than 10 separate calls

---

## Design Philosophy

- **Separation of concerns** — Feature calculation ≠ strategy/model logic
- **DRY principle** — Calculate each feature once, use everywhere
- **Extensible** — Easy to add features without modifying existing code
- **Testable** — Each feature can be tested independently
- **Composable** — Complex indicators built from simple building blocks

The feature library is the foundation of strategy + ML pipeline. Keep it clean, well-tested, and documented.
