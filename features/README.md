# Blofin Stack Feature Library

A comprehensive feature engineering library for crypto trading analysis.

## Features

The library provides 80+ technical features organized into 5 modules:

### 1. Price Features (`price_features.py`)
- Basic price levels: close, open, high, low, hl2, hlc3, ohlc4
- Returns: simple returns, log returns
- Momentum: momentum_n (n=1,5,10,20,50)
- Rate of change: roc_n 
- Price ranges and gaps

### 2. Volume Features (`volume_features.py`)
- Volume moving averages (SMA, EMA)
- Volume ratios and surge detection
- VWAP (Volume Weighted Average Price) and deviations
- On-Balance Volume (OBV)
- Volume price trend

### 3. Technical Indicators (`technical_indicators.py`)
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Stochastic Oscillator**
- **CCI** (Commodity Channel Index)
- **Williams %R**
- **ADX** (Average Directional Index)
- **Moving Averages** (SMA, EMA with multiple periods)
- **ADL** (Accumulation/Distribution Line)
- EMA crossovers (golden/death cross detection)

### 4. Volatility Features (`volatility_features.py`)
- **ATR** (Average True Range)
- **Standard Deviation**
- **Bollinger Bands** (upper, lower, width, %B)
- **Keltner Channels**
- **Historical Volatility** (annualized)
- Volatility ratios
- Parkinson's volatility (high-low range based)
- Bollinger Band squeeze detection

### 5. Market Regime (`market_regime.py`)
- Trend detection (up/down/ranging)
- Volatility regime detection
- Regime classification (trending_up, trending_down, ranging, volatile, neutral)
- Regime strength (0.0-1.0)
- Regime duration and transition detection

## Installation

No separate installation needed. The feature library is integrated with the blofin-stack project.

Required dependencies:
- pandas
- numpy
- sqlite3

## Quick Start

```python
from features import FeatureManager

# Initialize
fm = FeatureManager()

# Get all features for BTC
df = fm.get_features(
    symbol='BTC-USDT',
    timeframe='1m',
    lookback_bars=500
)

print(df.columns)  # Shows all available features
print(df.tail())   # Last 5 rows with all features
```

## Usage Examples

### Get Specific Features

```python
# Only get RSI and MACD
df = fm.get_features(
    symbol='BTC-USDT',
    timeframe='5m',
    feature_list=['rsi_14', 'macd_12_26', 'macd_signal_9', 'close'],
    lookback_bars=200
)
```

### List Available Features

```python
# List all features
all_features = fm.list_available_features()
print(f"Total features: {len(all_features)}")

# List features by group
price_features = fm.list_available_features(group='price')
technical_features = fm.list_available_features(group='technical')

# Get feature groups
groups = fm.get_feature_groups()
for group_name, features in groups.items():
    print(f"{group_name}: {len(features)} features")
```

### Custom Parameters

```python
# Customize indicator parameters
params = {
    'technical': {
        'rsi_periods': [7, 14, 21],  # Multiple RSI periods
        'ema_periods': [5, 10, 20, 50, 100, 200],
    },
    'volatility': {
        'atr_periods': [7, 14, 28],
        'bb_configs': [(10, 1.5), (20, 2.0), (50, 2.5)]  # (period, std_mult)
    }
}

df = fm.get_features(
    symbol='ETH-USDT',
    timeframe='15m',
    params=params,
    lookback_bars=1000
)
```

### Market Regime Detection

```python
from features.market_regime import get_regime_type, get_regime_strength

# Load data
df = fm.get_features(symbol='BTC-USDT', timeframe='1h', lookback_bars=500)

# Get current regime
regime = get_regime_type(df)
strength = get_regime_strength(df)

print(f"Current regime: {regime}")
print(f"Regime strength: {strength:.2f}")

# Access regime features
print(df[['timestamp', 'regime_type', 'regime_strength', 'is_trending_up']].tail())
```

### Performance Optimization

```python
# Enable caching (default)
fm = FeatureManager(cache_enabled=True)

# First call computes features
df1 = fm.get_features('BTC-USDT', '1m', lookback_bars=500)  # ~200ms

# Second call uses cache (within 60s TTL)
df2 = fm.get_features('BTC-USDT', '1m', lookback_bars=500)  # ~5ms

# Clear cache manually if needed
fm.clear_cache()
```

## Feature Groups

| Group | Features | Description |
|-------|----------|-------------|
| **price** | 25+ | Basic price transformations and momentum |
| **volume** | 15+ | Volume analysis and VWAP |
| **technical** | 25+ | Technical indicators (RSI, MACD, etc.) |
| **volatility** | 15+ | Volatility measures and bands |
| **regime** | 9 | Market regime classification |

## Performance

- **Fast execution**: All features for 100 candles compute in <1 second
- **Efficient caching**: Repeated queries are cached with 60s TTL
- **Optimized calculations**: Uses pandas vectorized operations
- **Memory efficient**: Only requested features are computed

## Testing

Run comprehensive test suite:

```bash
cd /home/rob/.openclaw/workspace/blofin-stack
source .venv/bin/activate
python features/tests/test_features.py
```

Test coverage:
- 22 test cases
- Tests for all feature modules
- Edge case handling (missing data, short data, zero volume)
- Integration tests with FeatureManager

## Architecture

```
features/
├── __init__.py              # Package initialization
├── feature_manager.py       # Central API with caching
├── price_features.py        # Price-based features
├── volume_features.py       # Volume-based features  
├── technical_indicators.py  # Technical analysis indicators
├── volatility_features.py   # Volatility measures
├── market_regime.py         # Market regime detection
├── tests/
│   ├── __init__.py
│   └── test_features.py     # Comprehensive test suite
└── README.md               # This file
```

## Integration with Blofin Stack

The feature library integrates seamlessly with the blofin-stack trading system:

1. **Data Source**: Loads tick data from `blofin_monitor.db`
2. **Timeframe Aggregation**: Automatically aggregates ticks into OHLCV candles
3. **Strategy Development**: Features can be used in strategy backtesting
4. **Real-time Updates**: Cache system ensures fresh features for live trading

## API Reference

### FeatureManager

#### `__init__(db_path=None, cache_enabled=True)`
Initialize the feature manager.

#### `get_features(symbol, timeframe='1m', feature_list=None, lookback_bars=500, params=None)`
Compute features for a symbol.

**Returns**: DataFrame with timestamp, OHLCV, and computed features

#### `list_available_features(group=None)`
List all available features or features in a specific group.

**Returns**: List of feature names

#### `get_feature_groups()`
Get all feature groups and their features.

**Returns**: Dict mapping group names to feature lists

#### `clear_cache()`
Clear the feature cache.

## Future Enhancements

Potential additions:
- Order flow features (bid/ask imbalance, etc.)
- Cross-asset features (correlations, spreads)
- Sentiment indicators
- Machine learning feature selection
- Feature importance ranking
- Real-time streaming features

## Contributing

When adding new features:
1. Add calculation function to appropriate module
2. Update `get_available_features()` in that module
3. Add tests in `test_features.py`
4. Document in this README

## License

Part of the blofin-stack project.
