# Portfolio Quantitative Analysis Framework

A comprehensive Python framework for analyzing trading portfolios focusing on temporal clustering, market regime consistency, and strategy frequency validation.

---

## :arrow_forward: Quick Start
1. **Put PDF files in a folder**
Choose a folder to put csv files in (ex: './portfolios/')

2. **Adjust configurations**  
Edit `custom_config.py` to point to your data folder, output paths, and tweak `FIGURE_SIZE` if desired.

3. **Run the Notebook**
   ```bash
   jupyter notebook Portfolio_Analysis.ipynb
   ```

4. **Explore results**  
   Outputs (CSV, PNG dashboard, HTML) land in `./results/`. Open, share, and iterate!

---

## :open_file_folder: Modules Overview

- **custom_config.py**  
  Holds global settings and constants: file paths, figure sizes, risk-free rate, benchmark series, etc.

- **data_utils.py**  
  Utilities to load and clean raw trade/market data: reading CSVs, parsing dates, handling missing values, and returning a tidy PnL DataFrame.

- **core_metrics.py**  
  Computes the full suite of performance and risk metrics:
   - **CAGR** • Compound annual growth rate
   - **Volatility** • Annualized std. dev. of returns
   - **Sharpe & Sortino Ratios** • Return per unit of risk
   - **Calmar Ratio & Max Drawdown** • Reward vs. worst drop
   - **VaR / CVaR (95% & 99%)** • Tail loss estimates
   - **Omega Ratio** • Gain/loss distribution ratio
   - **Stability Metrics** (Sharpe Stability, Return Stability) • Steadiness over time
   - **Consistency Metrics** (Recovery Consistency, Regime Consistency) • Reliability of bounce-backs and behavior across regimes
   - **Tail Ratio** • Upside vs. downside extremes
   - **Max Loss Streak** • Longest run of losses

- **portfolio_analyzer.py**  
  Orchestrates full analysis across portfolios: loads data, converts PnL to returns, calculates metrics, and aggregates results.

- **reporting.py**  
  Exports CSV/Excel tables, generates simple HTML/PDF snapshots, and has hooks for emailing or Slack alerts.

- **visualization.py**  
  Creates a six-panel dashboard with console logs at each step:
   1. **Risk–Return Scatter** (Volatility vs. CAGR, colored by Sharpe)
   2. **Drawdown Violin** (Distribution of max drawdowns)
   3. **Sharpe Distribution** (Histogram + KDE)
   4. **Top Performers** (Bar chart of top 10 Final Scores)
   5. **Trading Quality** (Win Rate vs. Profit Factor)
   6. **Score Distribution** (Histogram of Final Scores)

---

## :chart_with_upwards_trend: Metric Deep Dive

Here’s what each metric tells you (with a focus on stability, consistency, and tail ratio):

| Metric                   | What it measures and why it matters                                                                                    |
|--------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Sharpe Stability**     | Std. dev. of rolling-Sharpe (e.g., 90-day). Lower = steadier risk-adjusted performance over time.                      |
| **Return Stability**     | Std. dev. of rolling returns (e.g., monthly). Lower = more uniform month-to-month performance.                         |
| **Recovery Consistency** | % of drawdowns that recover within a target window (e.g., 3 months). Higher = more reliable bounce-backs after losses. |
| **Regime Consistency**   | Frequency of positive returns in different market regimes. High = strategy holds up across bull/bear phases.           |
| **Tail Ratio**           | Ratio of average upside extremes to average downside extremes.                                                         |


Other core metrics (CAGR, Volatility, Sortino, Calmar, Max Drawdown, VaR/CVaR, Omega, Max Loss Streak) follow standard definitions and help you gauge return, risk, and drawdown behavior.

---

## :bar_chart: Dashboard Guide

Each chart in `visualization.py` helps you zero in on a key dimension of performance:

1. **Risk–Return Scatter** 
   - _X-axis_: Annualized volatility (risk)
   - _Y-axis_: CAGR (reward)
   - _Color_: Sharpe Ratio (efficiency)  
     Insight: Seek portfolios in the upper-left (high return, low risk) with warm colors (Sharpe ≥ target).

2. **Drawdown Violin** 
   - Shows the full distribution of max drawdowns.  
     Insight: Narrow, centered violins = consistently controlled drawdowns; wide tails = strategies prone to deep slumps.

3. **Sharpe Distribution** 
   - Histogram + KDE of Sharpe Ratios.  
     Insight: See how many strategies clear your minimum Sharpe threshold (e.g., >1.5).

4. **Top Performers** 
   - Bar chart of the top 10 by composite Final Score.  
     Insight: Quickly identify leaders and how close the competition is.

5. **Trading Quality** 
   - _X-axis_: Win Rate
   - _Y-axis_: Profit Factor  
     Insight: Points in the top-right quadrant combine high hit rates with big win/loss ratios—ideal setups.  
     Threshold lines at 50% win rate and PF=1 guide you.

6. **Score Distribution** 
   - Histogram of Final Scores.  
     Insight: Understand overall scoring dispersion and how many strategies meet your cutoffs.
