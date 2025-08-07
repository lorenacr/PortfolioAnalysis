"""
Portfolio Analysis Framework - Main Entry Point
======================================================

Orchestrates complete portfolio analysis using modular components.
Implements institutional-grade quantitative analysis with academic standards.

Usage:
    python main.py
"""

import sys
import warnings

# Ensure current directory is in Python path for imports
if '.' not in sys.path:
    sys.path.insert(0, '.')

# Import our modular components
from portfolio_analyzer import PortfolioAnalyzer
from custom_config import INPUT_FOLDER

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """
    Main execution function - orchestrates complete portfolio analysis workflow.
    """

    # Initialize the analyzer
    analyzer = PortfolioAnalyzer()

    # Run complete analysis workflow
    analyzer.run_complete_analysis(folder_path=INPUT_FOLDER)

    return True

if __name__ == "__main__":
    main()
