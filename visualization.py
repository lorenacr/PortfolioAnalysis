"""
Visualization Module for Portfolio Analysis Framework
====================================================

Handles all visualization, dashboard creation, and plotting functionality
for portfolio analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from custom_config import FIGURE_SIZE, DPI, COLOR_PALETTE, DASHBOARD_FILENAME

# Set style
plt.style.use('default')  # Use default instead of seaborn-v0_8-darkgrid
sns.set_palette(COLOR_PALETTE)

def create_performance_dashboard(results_df: pd.DataFrame, save_path: str = DASHBOARD_FILENAME) -> None:
    """
    Create a comprehensive performance dashboard
    
    Args:
        results_df: DataFrame with portfolio analysis results
        save_path: Path to save the dashboard
    """
    try:
        if results_df.empty:
            print("⚠️  No data available for dashboard creation")
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=FIGURE_SIZE)
        fig.suptitle('Portfolio Performance Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Composite Score Distribution
        axes[0, 0].hist(results_df['Composite_Score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Composite Score Distribution')
        axes[0, 0].set_xlabel('Composite Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Sharpe Ratio vs. Max Drawdown
        scatter = axes[0, 1].scatter(results_df['Max_Drawdown'], results_df['Sharpe_Ratio'], 
                                   c=results_df['Composite_Score'], cmap='viridis', alpha=0.7)
        axes[0, 1].set_title('Risk-Return Profile')
        axes[0, 1].set_xlabel('Max Drawdown')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        plt.colorbar(scatter, ax=axes[0, 1], label='Composite Score')
        
        # 3. Return Distribution
        axes[0, 2].hist(results_df['Total_Return'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Total Return Distribution')
        axes[0, 2].set_xlabel('Total Return (%)')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Top 10 Portfolios by Composite Score
        top_10 = results_df.nlargest(10, 'Composite_Score')
        axes[1, 0].barh(range(len(top_10)), top_10['Composite_Score'], color='gold')
        axes[1, 0].set_yticks(range(len(top_10)))
        axes[1, 0].set_yticklabels([p[:15] + '...' if len(p) > 15 else p for p in top_10['Portfolio']])
        axes[1, 0].set_title('Top 10 Portfolios (Composite Score)')
        axes[1, 0].set_xlabel('Composite Score')
        
        # 5. Risk Metrics Comparison
        risk_metrics = ['Max_Drawdown', 'VaR_95', 'CVaR_95']
        risk_data = results_df[risk_metrics].mean()
        axes[1, 1].bar(risk_metrics, risk_data, color=['red', 'orange', 'darkred'])
        axes[1, 1].set_title('Average Risk Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Performance Metrics Correlation
        corr_metrics = ['Sharpe_Ratio', 'Sortino_Ratio', 'Calmar_Ratio', 'Omega_Ratio']
        if all(col in results_df.columns for col in corr_metrics):
            corr_matrix = results_df[corr_metrics].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
            axes[1, 2].set_title('Performance Metrics Correlation')
        
        # 7. Win Rate vs. Profit Factor
        if 'Win_Rate' in results_df.columns and 'Profit_Factor' in results_df.columns:
            axes[2, 0].scatter(results_df['Win_Rate'], results_df['Profit_Factor'], alpha=0.7, color='purple')
            axes[2, 0].set_title('Win Rate vs Profit Factor')
            axes[2, 0].set_xlabel('Win Rate (%)')
            axes[2, 0].set_ylabel('Profit Factor')
        
        # 8. Volatility Distribution
        if 'Volatility' in results_df.columns:
            axes[2, 1].hist(results_df['Volatility'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[2, 1].set_title('Volatility Distribution')
            axes[2, 1].set_xlabel('Volatility')
            axes[2, 1].set_ylabel('Frequency')
        
        # 9. Performance Summary Stats
        axes[2, 2].axis('off')
        summary_text = f"""
        Portfolio Analysis Summary
        ========================
        Total Portfolios: {len(results_df)}
        Avg Composite Score: {results_df['Composite_Score'].mean():.2f}
        Best Sharpe Ratio: {results_df['Sharpe_Ratio'].max():.3f}
        Avg Max Drawdown: {results_df['Max_Drawdown'].mean():.2%}
        Top Portfolio: {results_df.loc[results_df['Composite_Score'].idxmax(), 'Portfolio'][:20]}
        """
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Dashboard saved to {save_path}")
        
    except Exception as e:
        print(f"❌ Dashboard creation failed: {str(e)}")

def plot_portfolio_performance(portfolio_data: pd.DataFrame, portfolio_name: str) -> None:
    """
    Plot individual portfolio performance
    
    Args:
        portfolio_data: DataFrame with portfolio trade data
        portfolio_name: Name of the portfolio
    """
    try:
        if portfolio_data.empty or 'Profit/Loss' not in portfolio_data.columns:
            print(f"⚠️  Cannot plot {portfolio_name}: insufficient data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Portfolio Analysis: {portfolio_name}', fontsize=16, fontweight='bold')
        
        # 1. Cumulative Returns
        cumulative_returns = (1 + portfolio_data['Profit/Loss']).cumprod()
        axes[0, 0].plot(cumulative_returns, linewidth=2, color='blue')
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Return Distribution
        axes[0, 1].hist(portfolio_data['Profit/Loss'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Return Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio (if enough data)
        if len(portfolio_data) > 30:
            rolling_returns = portfolio_data['Profit/Loss'].rolling(window=30)
            rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
            axes[1, 0].plot(rolling_sharpe, linewidth=2, color='red')
            axes[1, 0].set_title('Rolling Sharpe Ratio (30-period)')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Drawdown Chart
        cumulative_returns = (1 + portfolio_data['Profit/Loss']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
        axes[1, 1].set_title('Drawdown')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Portfolio plotting failed for {portfolio_name}: {str(e)}")

def create_comparison_chart(results_df: pd.DataFrame, metric: str, top_n: int = 10) -> None:
    """
    Create comparison chart for specific metric
    
    Args:
        results_df: DataFrame with results
        metric: Metric to compare
        top_n: Number of top portfolios to show
    """
    try:
        if metric not in results_df.columns:
            print(f"⚠️  Metric '{metric}' not found in results")
            return
        
        top_portfolios = results_df.nlargest(top_n, metric)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_portfolios)), top_portfolios[metric])
        plt.yticks(range(len(top_portfolios)), 
                  [p[:20] + '...' if len(p) > 20 else p for p in top_portfolios['Portfolio']])
        plt.xlabel(metric.replace('_', ' ').title())
        plt.title(f'Top {top_n} Portfolios by {metric.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        
        # Color bars based on value
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / len(bars)))
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Comparison chart creation failed: {str(e)}")

def plot_risk_return_scatter(results_df: pd.DataFrame) -> None:
    """
    Create a risk-return scatter plot
    
    Args:
        results_df: DataFrame with portfolio results
    """
    try:
        if results_df.empty:
            return
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(results_df['Max_Drawdown'], results_df['Total_Return'], 
                            c=results_df['Sharpe_Ratio'], cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black')
        
        plt.xlabel('Max Drawdown')
        plt.ylabel('Total Return (%)')
        plt.title('Risk-Return Profile of Portfolios')
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.grid(True, alpha=0.3)
        
        # Add annotations for top performers
        top_5 = results_df.nlargest(5, 'Composite_Score')
        for idx, row in top_5.iterrows():
            plt.annotate(row['Portfolio'][:15], 
                        (row['Max_Drawdown'], row['Total_Return']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Risk-return scatter plot failed: {str(e)}")
