"""
Portfolio Visualization Module
==================================================

A robust visualization module for portfolio analysis with comprehensive error handling.
"""

import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from custom_config import DASHBOARD_FILENAME, FIGURE_SIZE, DPI

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
    print(f"📊 Creating comprehensive dashboard...")

    # 1. Risk-Return scatter
    print("  📈 Creating Risk-Return scatter plot...")
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_SIZE[0] * 1.3, FIGURE_SIZE[1] * 1.3))
    fig.suptitle("Portfolio Performance Dashboard - Comprehensive Analysis", fontsize=20, y=0.98)

    ax = axes[0, 0]
    scatter = ax.scatter(
        results_df["volatility"],
        results_df["cagr"],
        c=results_df["sharpe_ratio"],
        cmap="RdYlGn",
        s=80,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5
    )
    ax.set_title("Risk-Return Efficiency\n(Color = Sharpe Ratio)", fontsize=14)
    ax.set_xlabel("Volatility (Risk)", fontsize=12)
    ax.set_ylabel("CAGR", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax).set_label("Sharpe Ratio\n(Higher = Better)", fontsize=10)

    # 2. Drawdown analysis
    print("  📉 Creating Drawdown analysis...")
    ax = axes[0, 1]
    sns.violinplot(y=results_df["max_drawdown"], ax=ax, color="lightcoral", inner="quartile")
    ax.set_title("Drawdown Distribution", fontsize=14)
    ax.set_ylabel("Max Drawdown", fontsize=12)
    ax.grid(True, alpha=0.3)

    # 3. Sharpe Ratio distribution
    print("  📊 Creating Sharpe Ratio distribution...")
    ax = axes[0, 2]
    sns.histplot(results_df["sharpe_ratio"], bins=15, kde=True, color="teal", ax=ax)
    ax.set_title("Sharpe Ratio Distribution", fontsize=14)
    ax.set_xlabel("Sharpe Ratio", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.3)

    # 4. Top performers ranking
    print("  🏆 Creating top performers ranking...")
    ax = axes[1, 0]
    top10 = results_df.nlargest(10, "Final_Score")
    bars = ax.barh(top10["portfolio"], top10["Final_Score"], color="skyblue", edgecolor="black")
    ax.set_title("Top 10 Portfolios", fontsize=14)
    ax.set_xlabel("Final Score", fontsize=12)
    ax.grid(True, alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}", va="center", fontsize=8)

    # 5. Trading quality analysis
    print("  🎯 Creating trading quality analysis...")
    ax = axes[1, 1]
    scatter_q = ax.scatter(
        results_df["win_rate"],
        results_df["profit_factor"],
        c=results_df["win_rate"],
        cmap="RdYlGn",
        s=80,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5
    )
    ax.set_title("Trading Quality\n(Win Rate vs Profit Factor)", fontsize=14)
    ax.set_xlabel("Win Rate", fontsize=12)
    ax.set_ylabel("Profit Factor", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Break-even")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="50% Win Rate")
    ax.legend()

    # 6. Performance distribution
    print("  📈 Creating performance distribution...")
    ax = axes[1, 2]
    sns.histplot(results_df["Final_Score"], bins=15, color="lightgreen", edgecolor="black", ax=ax)
    median = results_df["Final_Score"].median()
    ax.axvline(median, color="red", linestyle="--", label=f"Median: {median:.2f}")
    ax.set_title("Performance Score Distribution", fontsize=14)
    ax.set_xlabel("Final Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # finalize
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print("✅ Dashboard creation completed!")
    print(f"   📁 Saved to: {save_path}")
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)

