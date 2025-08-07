"""
Configuration and Constants for Portfolio Analysis Framework
==========================================================

Central configuration file with all constants, thresholds, and settings.
"""

import os

# Analysis Parameters
MIN_TRADES_THRESHOLD = 30
SCORING_METHOD = 'topsis'
INITIAL_CAPITAL = 10000
TRADING_DAYS_PER_YEAR = 252
ANNUAL_RISK_FREE_RATE = 0.05

# Scoring Weights (Academic/Institutional Standards)
SCORING_WEIGHTS = {
    'sharpe_ratio': 0.10, 'sortino_ratio': 0.08, 'calmar_ratio': 0.07,
    'omega_ratio': 0.05, 'cagr': 0.05, 'max_drawdown': 0.08,
    'volatility': 0.05, 'var_95': 0.06, 'cvar_95': 0.06,
    'sharpe_stability': 0.08, 'return_stability': 0.07,
    'recovery_consistency': 0.05, 'regime_consistency': 0.05,
    'tail_ratio': 0.05, 'cvar_99': 0.05, 'max_loss_streak': 0.05
}

# Benefit vs cost criteria
BENEFIT_CRITERIA = [
    'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio',
    'cagr', 'tail_ratio', 'regime_consistency',
    'sharpe_stability', 'return_stability', 'recovery_consistency'
]

COST_CRITERIA = [
    'max_drawdown', 'volatility', 'var_95', 'cvar_95', 'cvar_99',
    'max_loss_streak'
]

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