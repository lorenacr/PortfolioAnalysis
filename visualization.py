"""
Portfolio Visualization Module
==================================================

A robust visualization module for portfolio analysis with comprehensive error handling.
"""

import warnings

import matplotlib.pyplot as plt

from custom_config import DASHBOARD_FILENAME, FIGURE_SIZE

warnings.filterwarnings('ignore')

def create_performance_dashboard(results_df, save_path=DASHBOARD_FILENAME):
    """
    Create a comprehensive performance dashboard.
    
    Args:
        results_df: DataFrame with portfolio analysis results
        save_path: Path to save dashboard image
        
    Returns:
        bool: Success status
    """
    if results_df is None or results_df.empty:
        print("⚠️ No data available for dashboard creation")
        return False

    print(f"\n📊 Creating comprehensive dashboard with {len(results_df)} portfolios...")

    try:
        # Create figure with 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
        fig.suptitle('Portfolio Performance Dashboard - Comprehensive Analysis',
                     fontsize=20, y=0.98)

        # 1. Risk-Return Scatter (Top Left)
        print("  📈 Creating Risk-Return scatter plot...")
        if all(col in results_df.columns for col in ['volatility', 'total_return', 'sharpe_ratio']):
            scatter = axes[0, 0].scatter(
                results_df['volatility'],
                results_df['total_return'],
                c=results_df['sharpe_ratio'],
                s=80,
                alpha=0.8,
                cmap='RdYlGn',
                edgecolors='black',
                linewidth=0.5
            )
            axes[0, 0].set_xlabel('Volatility (Risk)', fontsize=12)
            axes[0, 0].set_ylabel('Total Return', fontsize=12)
            axes[0, 0].set_title('Risk-Return Efficiency\n(Color = Sharpe Ratio)', fontsize=14)
            axes[0, 0].grid(True, alpha=0.3)

            cbar = plt.colorbar(scatter, ax=axes[0, 0])
            cbar.set_label('Sharpe Ratio\n(Higher = Better)', fontsize=10)
        else:
            axes[0, 0].text(0.5, 0.5, 'Missing risk-return data', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Risk-Return Efficiency (Missing Data)')

        # 2. Drawdown Analysis (Top Center)
        print("  📉 Creating Drawdown analysis...")
        if all(col in results_df.columns for col in ['max_drawdown', 'total_return']):
            # Simple color coding based on drawdown
            colors = ['green' if dd > -0.1 else 'orange' if dd > -0.2 else 'red'
                      for dd in results_df['max_drawdown']]

            axes[0, 1].scatter(
                results_df['max_drawdown'],
                results_df['total_return'],
                c=colors,
                alpha=0.8,
                s=80,
                edgecolors='black',
                linewidth=0.5
            )
            axes[0, 1].set_xlabel('Max Drawdown', fontsize=12)
            axes[0, 1].set_ylabel('Total Return', fontsize=12)
            axes[0, 1].set_title('Drawdown vs Return Analysis\n(Color = Risk Level)', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Missing drawdown data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Drawdown Analysis (Missing Data)')

        # 3. Sharpe Distribution (Top Right)
        print("  📊 Creating Sharpe Ratio distribution...")
        if 'sharpe_ratio' in results_df.columns:
            clean_sharpe = results_df['sharpe_ratio'].dropna()
            if not clean_sharpe.empty:
                axes[0, 2].hist(clean_sharpe, bins=15, alpha=0.8, color='skyblue', edgecolor='black')
                axes[0, 2].axvline(clean_sharpe.median(), color='red', linestyle='--',
                                   label=f'Median: {clean_sharpe.median():.2f}')
                axes[0, 2].set_xlabel('Sharpe Ratio')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].set_title('Sharpe Ratio Distribution')
                axes[0, 2].grid(True, alpha=0.3)
                axes[0, 2].legend()
            else:
                axes[0, 2].text(0.5, 0.5, 'No valid Sharpe data', ha='center', va='center', transform=axes[0, 2].transAxes)
        else:
            axes[0, 2].text(0.5, 0.5, 'sharpe_ratio column not found', ha='center', va='center', transform=axes[0, 2].transAxes)

        # 4. Top Performers (Bottom Left)
        print("  🏆 Creating top performers ranking...")
        if 'Final_Score' in results_df.columns and 'portfolio' in results_df.columns:
            top_10 = results_df.nlargest(10, 'Final_Score')
            if not top_10.empty:
                y_pos = range(len(top_10))
                bars = axes[1, 0].barh(y_pos, top_10['Final_Score'], color='lightgreen', edgecolor='black')
                axes[1, 0].set_yticks(y_pos)
                clean_names = [name.replace('Portfolio ', 'P') for name in top_10['portfolio']]
                axes[1, 0].set_yticklabels(clean_names, fontsize=8)
                axes[1, 0].set_xlabel('Final Score')
                axes[1, 0].set_title('Top 10 Portfolios')
                axes[1, 0].grid(True, alpha=0.3)

                # Add values on bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    axes[1, 0].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                                    f'{width:.2f}', ha='left', va='center', fontsize=8)
            else:
                axes[1, 0].text(0.5, 0.5, 'No data for top performers', ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'Final_Score or Portfolio columns not found', ha='center', va='center', transform=axes[1, 0].transAxes)

        # 5. Trading Quality (Bottom Center)
        print("  🎯 Creating trading quality analysis...")
        if all(col in results_df.columns for col in ['win_rate', 'avg_win', 'avg_loss']):
            axes[1, 1].scatter(
                results_df['win_rate'],
                results_df['profit_factor'],
                c=results_df['win_rate'],
                s=80,
                alpha=0.8,
                cmap='RdYlGn',
                edgecolors='black',
                linewidth=0.5
            )
            axes[1, 1].set_xlabel('Win Rate')
            axes[1, 1].set_ylabel('Profit Factor')
            axes[1, 1].set_title('Trading Quality\n(Win Rate vs Profit Factor)')
            axes[1, 1].grid(True, alpha=0.3)

            # Add reference lines
            axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even')
            axes[1, 1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='50% Win Rate')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Missing trading data', ha='center', va='center', transform=axes[1, 1].transAxes)

        # 6. Performance Distribution (Bottom Right)
        print("  📈 Creating performance distribution...")
        if 'Final_Score' in results_df.columns:
            clean_scores = results_df['Final_Score'].dropna()
            if not clean_scores.empty:
                axes[1, 2].hist(clean_scores, bins=15, alpha=0.8, color='lightcoral', edgecolor='black')
                axes[1, 2].axvline(clean_scores.median(), color='red', linestyle='--',
                                   label=f'Median: {clean_scores.median():.3f}')
                axes[1, 2].set_xlabel('Final Score')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].set_title('Performance Score\nDistribution')
                axes[1, 2].grid(True, alpha=0.3)
                axes[1, 2].legend()
            else:
                axes[1, 2].text(0.5, 0.5, 'No valid score data', ha='center', va='center', transform=axes[1, 2].transAxes)
        else:
            axes[1, 2].text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=axes[1, 2].transAxes)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save dashboard
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

        print(f"\n✅ Dashboard creation completed!")
        print(f"   📁 Saved to: {save_path}")

        # Show dashboard
        plt.show()

        return True

    except Exception as e:
        print(f"❌ Critical error creating dashboard: {str(e)}")
        return False

    finally:
        plt.close('all')

if __name__ == "__main__":
    print("📊 Portfolio Visualization Module - Simplified Version")
