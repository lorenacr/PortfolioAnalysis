"""
Configuration and Constants for Portfolio Analysis Framework
==========================================================

Central configuration file with all constants, thresholds, and settings.
"""

import os

# Analysis Parameters
MIN_TRADES_THRESHOLD = 30
CONFIDENCE_LEVEL = 0.95
MONTE_CARLO_SIMULATIONS = 1000
LOOKBACK_PERIOD = 252  # Trading days in a year

# Scoring Weights (Academic/Institutional Standards)
SCORING_WEIGHTS = {
    'sharpe_ratio': 0.25,
    'sortino_ratio': 0.20,
    'max_drawdown': 0.15,
    'var_95': 0.10,
    'cvar_95': 0.10,
    'calmar_ratio': 0.10,
    'omega_ratio': 0.10
}

# Risk Thresholds
RISK_THRESHOLDS = {
    'max_drawdown_limit': 0.20,  # 20% maximum drawdown
    'min_sharpe_ratio': 1.0,     # Minimum Sharpe ratio
    'min_sortino_ratio': 1.2,    # Minimum Sortino ratio
    'max_var_95': 0.05,          # Maximum 5% VaR
    'min_calmar_ratio': 0.5      # Minimum Calmar ratio
}

# Visualization Settings
FIGURE_SIZE = (20, 15)
DPI = 300
COLOR_PALETTE = 'viridis'
STYLE = 'seaborn-v0_8'

# Progress bar colors
LOAD_PORTFOLIO_COLOR = 'green'
ANALYZE_PORTFOLIO_COLOR = 'red'
NORMALIZE_PORTFOLIO_COLOR = 'blue'

# Folder Paths 
INPUT_FOLDER = './portfolios/'   # Input folder where Portfolio CSV files are located
OUTPUT_FOLDER = './results/'     # Results folder where analysis outputs are saved

# File Names (will be saved in results folder)
DASHBOARD_FILENAME = os.path.join(OUTPUT_FOLDER, 'portfolio_dashboard.png')
REPORT_FILENAME = os.path.join(OUTPUT_FOLDER, 'portfolio_analysis_report.txt')
RESULTS_FILENAME = os.path.join(OUTPUT_FOLDER, 'portfolio_analysis_results.csv')
TOP_PERFORMERS_FILENAME = os.path.join(OUTPUT_FOLDER, 'portfolio_top_performers.csv')

# Market Analysis Settings
N_REGIMES = 3
REGIME_LABELS = ['Bear Market', 'Neutral Market', 'Bull Market']
