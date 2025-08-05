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

from custom_config import COLOR_PALETTE, DASHBOARD_FILENAME, FIGURE_SIZE

# Set style
plt.style.use('default')  # Use default instead of seaborn-v0_8-darkgrid
sns.set_palette(COLOR_PALETTE)

def create_performance_dashboard(results_df: pd.DataFrame, save_path: str = DASHBOARD_FILENAME):
    """
    Create a comprehensive performance dashboard
    
    Args:
        results_df: DataFrame with portfolio analysis results
        save_path: Path to save the dashboard
    """
    if results_df.empty:
        print("⚠️  No data available for dashboard creation")
        return
    
    # Create a figure with subplots
    # Create figure with 2x3 layout for the 6 essential charts
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
    fig.suptitle('Portfolio Performance Dashboard', fontsize=20, fontweight='bold')

    # ============================================================================
    # 1. RISK-RETURN EFFICIENCY SCATTER (Top Left)
    # ============================================================================
    ax1 = axes[0, 0]
    scatter = ax1.scatter(results_df['volatility'], results_df['total_return'],
                          c=results_df['sharpe_ratio'], s=80, alpha=0.8, cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Volatility (Risk)')
    ax1.set_ylabel('Total Return')
    ax1.set_title('Risk-Return Efficiency')
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Sharpe Ratio')

    # ============================================================================
    # 2. DRAWDOWN vs RETURN ANALYSIS (Top Center)  
    # ============================================================================
    ax2 = axes[0, 1]
    # Color coding: Green <10%, Orange 10-20%, Red >20%
    colors = ['green' if dd < 0.1 else 'orange' if dd < 0.2 else 'red'
              for dd in results_df['max_drawdown']]

    ax2.scatter(results_df['max_drawdown'], results_df['total_return'],
                c=colors, alpha=0.8, s=80, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Maximum Drawdown')
    ax2.set_ylabel('Total Return')
    ax2.set_title('Drawdown vs Return')
    ax2.grid(True, alpha=0.3)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Safe (<10% DD)'),
                       Patch(facecolor='orange', label='Moderate (10-20% DD)'),
                       Patch(facecolor='red', label='High Risk (>20% DD)')]
    ax2.legend(handles=legend_elements, loc='upper right')

    # ============================================================================
    # 3. TRADING QUALITY ANALYSIS (Top Right)
    # ============================================================================
    ax3 = axes[0, 2]
    profit_factor = results_df['Avg_Win'] / abs(results_df['Avg_Loss'])
    scatter3 = ax3.scatter(results_df['Win_Rate'], profit_factor,
                           c=results_df['Trade_Count'], s=80, alpha=0.8, cmap='viridis',
                           edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Win Rate')
    ax3.set_ylabel('Profit Factor')
    ax3.set_title('Trading Quality')
    ax3.grid(True, alpha=0.3)

    # Add colorbar
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Trade Count')

    # Add reference lines
    ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Min Profit Factor (1.5)')
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Min Win Rate (50%)')

    # ============================================================================
    # 4. SHARPE RATIO DISTRIBUTION (Bottom Left)
    # ============================================================================
    ax4 = axes[1, 0]
    n, bins, patches = ax4.hist(results_df['sharpe_ratio'], bins=30, alpha=0.8,
                                color='skyblue', edgecolor='black', linewidth=1)

    # Color bars based on Sharpe quality
    for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
        if bin_val < 1:
            patch.set_facecolor('red')
        elif bin_val < 2:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('green')

    # Add statistics lines
    median_sharpe = results_df['sharpe_ratio'].median()
    mean_sharpe = results_df['sharpe_ratio'].mean()
    ax4.axvline(median_sharpe, color='blue', linestyle='--', linewidth=2,
                label=f'Median: {median_sharpe:.2f}')
    ax4.axvline(mean_sharpe, color='red', linestyle='-', linewidth=2,
                label=f'Mean: {mean_sharpe:.2f}')

    ax4.set_xlabel('Sharpe Ratio')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Sharpe Ratio Distribution')
    ax4.grid(True, alpha=0.3)

    # ============================================================================
    # 5. CONSISTENCY & STABILITY METRICS (Bottom Center)
    # ============================================================================
    ax5 = axes[1, 1]
    stability_metrics = ['sharpe_stability', 'return_stability', 'recovery_consistency']
    stability_data = [results_df[metric] for metric in stability_metrics]

    # Create boxplot with custom colors
    bp = ax5.boxplot(stability_data, labels=['Sharpe\nStability', 'Return\nStability', 'Recovery\nConsistency'],
                     patch_artist=True, notch=True)

    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax5.set_ylabel('Stability Score (0-1)')
    ax5.set_title('Stability Metrics')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.1)

    # Add reference line for good stability
    ax5.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Good Stability (0.7+)')

    # ============================================================================
    # 6. FINAL RANKING - TOP PERFORMERS (Bottom Right)
    # ============================================================================
    ax6 = axes[1, 2]
    top_10 = results_df.nlargest(10, 'Final_Score')

    # Create horizontal bar chart with gradient colors
    bars = ax6.barh(range(len(top_10)), top_10['Final_Score'],
                    color=plt.cm.RdYlGn(np.linspace(0.4, 1, len(top_10))),
                    edgecolor='black', linewidth=0.5)

    ax6.set_yticks(range(len(top_10)))
    ax6.set_yticklabels([p.replace('Portfolio ', 'P') for p in top_10['Portfolio']])
    ax6.set_xlabel('Final Composite Score')
    ax6.set_title('Top 10 Portfolios')
    ax6.grid(True, alpha=0.3, axis='x')

    # Add score values on bars
    for (bar, score) in enumerate(zip(bars, top_10['Final_Score'])):
        ax6.text(score + 0.01 * score, bar.get_y() + bar.get_height()/2, f'{score:.2f}',
                 va='center', ha='left')

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust layout to make space for the suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Essential portfolio dashboard saved to: {save_path}")

    for i, (bar, score) in enumerate(zip(bars, top_10['Final_Score'])):
        ax6.text(score + 0.001, i, f'{score:.3f}', va='center')

    # ============================================================================
    # FINAL LAYOUT AND SAVE
    # ============================================================================
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for main title
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 ESSENTIAL DASHBOARD saved to: {save_path}")
    plt.show()
