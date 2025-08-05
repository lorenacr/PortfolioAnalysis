# Portfolio Quantitative Analysis Framework

A comprehensive Python framework for analyzing trading portfolios focusing on temporal clustering, market regime consistency, and strategy frequency validation.

## Features

### Core Analysis Modules

1. **Temporal Clustering Analysis**
   - Identifies trading patterns and time-based clustering
   - Calculates clustering scores and peak trading hours
   - Distinguishes between HFT and swing trading strategies

2. **Market Regime Consistency Analysis**
   - Evaluates performance consistency across different market conditions
   - Analyzes yearly, quarterly, and volatility regime performance
   - Provides consistency scores for strategy robustness assessment

3. **Strategy Frequency Validation**
   - Validates that each strategy maintains optimal trading frequency (4–12 trades/month)
   - Calculates compliance rates and identifies outliers
   - Monitors strategy health and performance degradation

4. **Comprehensive Visualization Suite**
   - Automated chart generation for all key metrics
   - Multi-panel dashboard views
   - Export-ready high-resolution plots

## Quick Start
### :one: Put PDF files in a folder
- Choose a folder to put csv files in (ex: './portfolios/')

### :two: Execution:
- Execute .ipynb file

## Key Metrics Explained

### Temporal Clustering Metrics

- **Clustering Score**: Percentage of trades occurring within 1 hour of each other
  - `< 0.1`: Swing trading (medium-term strategies)
  - `> 0.3`: High-frequency trading patterns

- **Hourly Concentration**: Peak hour trading intensity
  - Higher values indicate strategic timing (e.g., market open)

### Regime Consistency Metrics

- **Overall Consistency**: Combined consistency score (0-1 scale)
  - `> 0.7`: Highly consistent across regimes
  - `0.4-0.7`: Moderately consistent
  - `< 0.4`: Regime-dependent performance

- **Temporal Consistency**: Year-over-year performance stability
- **Seasonal Consistency**: Quarter-over-quarter performance stability
- **Volatility Consistency**: Performance across different volatility regimes

### Strategy Frequency Metrics

- **Compliance Rate**: Percentage of months within target frequency range
  - `> 80%`: Excellent frequency control
  - `60-80%`: Good frequency control
  - `< 60%`: Needs attention

## Output Files

The framework generates several output files:

1. **Analysis Reports**: Comprehensive text-based analysis results
2. **Visualization Charts**: High-resolution PNG files with multi-panel dashboards
3. **Comparison Tables**: CSV exports of portfolio comparison data

## Advanced Configuration

### Custom Volatility Regimes
````python
# Modify volatility regime classification
df['custom_regime'] = pd.cut(
    df['trade_duration_hours'], 
    bins=[0, 4, 12, 48, float('inf')], 
    labels=['Ultra_High', 'High', 'Medium', 'Low']
)
````

### Custom Visualization Themes
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set custom style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
````

### Performance Optimization

For large datasets (>100k trades):
```python
# Use chunked processing
chunk_size = 10000
for chunk in pd.read_csv('large_portfolio.csv', chunksize=chunk_size):
    # Process chunk
    pass
````