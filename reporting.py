"""
Reporting Module for Portfolio Analysis Framework
================================================

Handles report generation, data export, and summary creation
for portfolio analysis results.
"""

from datetime import datetime
from typing import Dict

import pandas as pd

from custom_config import REPORT_FILENAME, RESULTS_FILENAME, TOP_PERFORMERS_FILENAME

def generate_analysis_report(results_df: pd.DataFrame, portfolios_data: Dict, 
                           save_path: str = REPORT_FILENAME) -> str:
    """
    Generate a comprehensive analysis report
    
    Args:
        results_df: DataFrame with analysis results
        portfolios_data: Dictionary with portfolio data
        save_path: Path to save the report
        
    Returns:
        Report content as string
    """
    try:
        if results_df.empty:
            return "No data available for report generation"
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate summary statistics
        total_portfolios = len(results_df)
        avg_composite_score = results_df['Composite_Score'].mean()
        best_portfolio = results_df.loc[results_df['Composite_Score'].idxmax()]
        
        # Create report content
        report_content = f"""
ADVANCED PORTFOLIO QUANTITATIVE ANALYSIS REPORT
==============================================
Generated: {timestamp}
Analysis Framework: Institutional-Grade Quantitative Assessment

EXECUTIVE SUMMARY
================
Total Portfolios Analyzed: {total_portfolios}
Average Composite Score: {avg_composite_score:.3f}
Analysis Period: Complete historical data
Risk Assessment: Comprehensive multi-metric evaluation

TOP PERFORMING PORTFOLIO
========================
Portfolio: {best_portfolio['portfolio']}
Composite Score: {best_portfolio['Composite_Score']:.3f}
Sharpe Ratio: {best_portfolio['Sharpe_Ratio']:.3f}
Sortino Ratio: {best_portfolio['Sortino_Ratio']:.3f}
Maximum Drawdown: {best_portfolio['Max_Drawdown']:.2%}
CAGR: {best_portfolio['cagr']:.2f}%
VaR (95%): {best_portfolio['VaR_95']:.3f}
CVaR (95%): {best_portfolio['CVaR_95']:.3f}

PERFORMANCE DISTRIBUTION ANALYSIS
=================================
Composite Score Statistics:
- Mean: {results_df['Composite_Score'].mean():.3f}
- Median: {results_df['Composite_Score'].median():.3f}
- Standard Deviation: {results_df['Composite_Score'].std():.3f}
- Min: {results_df['Composite_Score'].min():.3f}
- Max: {results_df['Composite_Score'].max():.3f}

Sharpe Ratio Statistics:
- Mean: {results_df['Sharpe_Ratio'].mean():.3f}
- Median: {results_df['Sharpe_Ratio'].median():.3f}
- Portfolios with Sharpe > 1.0: {len(results_df[results_df['Sharpe_Ratio'] > 1.0])} ({len(results_df[results_df['Sharpe_Ratio'] > 1.0])/total_portfolios*100:.1f}%)
- Portfolios with Sharpe > 2.0: {len(results_df[results_df['Sharpe_Ratio'] > 2.0])} ({len(results_df[results_df['Sharpe_Ratio'] > 2.0])/total_portfolios*100:.1f}%)

RISK ANALYSIS
=============
Maximum Drawdown Analysis:
- Average Max Drawdown: {results_df['Max_Drawdown'].mean():.2%}
- Median Max Drawdown: {results_df['Max_Drawdown'].median():.2%}
- Portfolios with DD < 10%: {len(results_df[results_df['Max_Drawdown'] < 0.10])} ({len(results_df[results_df['Max_Drawdown'] < 0.10])/total_portfolios*100:.1f}%)
- Portfolios with DD > 20%: {len(results_df[results_df['Max_Drawdown'] > 0.20])} ({len(results_df[results_df['Max_Drawdown'] > 0.20])/total_portfolios*100:.1f}%)

Value at Risk (VaR 95%) Analysis:
- Average VaR: {results_df['VaR_95'].mean():.3f}
- Median VaR: {results_df['VaR_95'].median():.3f}

TOP 10 PORTFOLIOS BY COMPOSITE SCORE
===================================
"""
        
        # Add top 10 portfolios
        top_10 = results_df.nlargest(10, 'Composite_Score')
        for i, (idx, portfolio) in enumerate(top_10.iterrows(), 1):
            report_content += f"""
{i:2d}. {portfolio['portfolio']}
    Composite Score: {portfolio['Composite_Score']:.3f}
    Sharpe Ratio: {portfolio['Sharpe_Ratio']:.3f}
    Max Drawdown: {portfolio['Max_Drawdown']:.2%}
    CAGR: {portfolio['cagr']:.2f}%
"""
        
        # Add risk warnings
        report_content += f"""

RISK WARNINGS AND RECOMMENDATIONS
=================================
"""
        
        high_risk_portfolios = results_df[results_df['Max_Drawdown'] > 0.20]
        if len(high_risk_portfolios) > 0:
            report_content += f"""
⚠️  HIGH RISK ALERT: {len(high_risk_portfolios)} portfolios show maximum drawdown > 20%
   Consider risk management improvements for these portfolios.

"""
        
        low_sharpe_portfolios = results_df[results_df['Sharpe_Ratio'] < 0.5]
        if len(low_sharpe_portfolios) > 0:
            report_content += f"""
⚠️  LOW PERFORMANCE ALERT: {len(low_sharpe_portfolios)} portfolios show Sharpe ratio < 0.5
   These portfolios may not provide adequate risk-adjusted returns.

"""
        
        # Add methodology section
        report_content += f"""
METHODOLOGY
===========
This analysis employs institutional-grade quantitative methods:

1. Composite Scoring System:
   - Sharpe Ratio (25% weight): Risk-adjusted return measure
   - Sortino Ratio (20% weight): Downside risk-adjusted return
   - Maximum Drawdown (15% weight): Peak-to-trough decline
   - Value at Risk - VaR 95% (10% weight): Potential loss at 95% confidence
   - Conditional VaR - CVaR 95% (10% weight): Expected loss beyond VaR
   - Calmar Ratio (10% weight): Annual return / Maximum drawdown
   - Omega Ratio (10% weight): Probability-weighted ratio of gains vs losses

2. Risk Assessment:
   - Monte Carlo validation with {portfolios_data.get('monte_carlo_sims', 1000)} simulations
   - Market regime detection using Gaussian Mixture Models
   - Statistical significance testing

3. Data Quality Standards:
   - Minimum 30 trades per portfolio for statistical significance
   - Outlier detection and handling
   - Missing data validation

DISCLAIMER
==========
This analysis is for informational purposes only and does not constitute
investment advice. Past performance does not guarantee future results.
All investments carry risk of loss. Please consult with a qualified
financial advisor before making investment decisions.

Report generated by Advanced Portfolio Quantitative Analysis Framework
© 2025 - Institutional Grade Analytics
"""
        
        # Save report to file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 Analysis report saved to {save_path}")
        return report_content
        
    except Exception as e:
        error_msg = f"❌ Report generation failed: {str(e)}"
        print(error_msg)
        return error_msg

def export_results_to_csv(results_df: pd.DataFrame, save_path: str = RESULTS_FILENAME):
    """
    Export analysis results to a CSV file
    
    Args:
        results_df: DataFrame with analysis results
        save_path: Path to save the CSV file
    """
    try:
        if results_df.empty:
            print("⚠️  No results to export")
            return
        
        # Add timestamp column
        results_df['Analysis_Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to CSV
        results_df.to_csv(save_path)
        print(f"💾 Results exported to {save_path}")
        
    except Exception as e:
        print(f"❌ CSV export failed: {str(e)}")

def generate_executive_summary(results_df: pd.DataFrame):
    """
    Create executive summary statistics
    
    Args:
        results_df: DataFrame with analysis results
    """
    try:        
        summary = {
            'Total Portfolios Analyzed': len(results_df),
            'Average Sharpe Ratio': results_df['sharpe_ratio'].mean(),
            'Average Max Drawdown': results_df['max_drawdown'].mean(),
            'Best Performing Portfolio': results_df.iloc[0]['portfolio'],
            'Highest Sharpe Ratio': results_df['sharpe_ratio'].max(),
            'Average Stability Score': results_df.get('sharpe_stability', pd.Series([0])).mean()
        }

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Executive summary creation failed: {str(e)}")

def generate_portfolio_recommendations(results_df: pd.DataFrame, top_n: int = 5):
    """
    Generate investment recommendations based on analysis
    
    Args:
        results_df: DataFrame with analysis results
        top_n: Number of top recommendations to generate
        
    Returns:
        List of recommendation dictionaries
    """
    try:
        print(f"\n🏆 Top {top_n} Performers Analysis:")
        print("-" * 35)
        
        top_performers_df = results_df.head(top_n)

        # Key metrics for top performers
        key_metrics = ['portfolio', 'Final_Score', 'Final_Rank', 'sharpe_ratio',
                       'max_drawdown', 'cagr', 'trade_count']

        # Filter to available metrics
        available_key_metrics = [col for col in key_metrics if col in top_performers_df.columns]

        top_performers_summary = top_performers_df[available_key_metrics]

        print(f"Top {top_n} portfolios by composite score:")
        for idx, row in top_performers_summary.iterrows():
            portfolio_name = row['portfolio']
            score = row.get('Final_Score', 0)
            rank = row.get('Final_Rank', 0)
            print(f"   #{int(rank):2d}: {portfolio_name} (Score: {score:.4f})")

        # Save top performers report
        top_performers_summary.to_csv(TOP_PERFORMERS_FILENAME)
        print(f"\n💾 Top performers report saved to: {TOP_PERFORMERS_FILENAME}")

    except Exception as e:
        print(f"❌ Recommendation generation failed: {str(e)}")

