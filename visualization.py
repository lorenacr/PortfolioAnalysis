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

    Saves the figure to disk and opens an interactive window via
    ``matplotlib.pyplot.show()`` (GUI backend required; close the window to unblock).

    Args:
        results_df: DataFrame from the analysis pipeline (must include ``mc_max_dd_p95``).
        save_path: Path to save dashboard image

    Returns:
        bool: Success status
    """

    # Readable fonts when the figure is scaled to fit the display window
    rc_vis = {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }

    with plt.rc_context(rc_vis):
        # 1. Risk-Return scatter
        fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
        fig.suptitle(
            "Portfolio Performance Dashboard — Comprehensive Analysis",
            fontsize=16,
            y=0.995,
            fontweight="bold",
        )

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
        ax.set_title("Risk-Return Efficiency\n(Color = Sharpe Ratio)")
        ax.set_xlabel("Volatility (Risk)")
        ax.set_ylabel("CAGR")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax).set_label("Sharpe Ratio\n(Higher = Better)")

        # 2. Monte Carlo max drawdown at 95% confidence (trade reordering), across portfolios
        ax = axes[0, 1]
        sns.histplot(
            results_df["mc_max_dd_p95"],
            bins=15,
            kde=True,
            color="lightcoral",
            edgecolor="black",
            ax=ax,
        )
        ax.set_title(
            "MC Max Drawdown (95%)\nTrade reordering — across portfolios",
        )
        ax.set_xlabel("95th percentile max DD (Monte Carlo)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # 3. Sharpe Ratio distribution
        ax = axes[0, 2]
        sns.histplot(results_df["sharpe_ratio"], bins=15, kde=True, color="teal", ax=ax)
        ax.set_title("Sharpe Ratio Distribution")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # 4. Top performers ranking
        ax = axes[1, 0]
        top10 = results_df.nlargest(10, "Final_Score")
        bars = ax.barh(top10["portfolio"], top10["Final_Score"], color="skyblue", edgecolor="black")
        ax.set_title("Top 10 Portfolios")
        ax.set_xlabel("Final Score")
        ax.grid(True, alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}", va="center", fontsize=10)

        # 5. Trading quality analysis
        ax = axes[1, 1]
        ax.scatter(
            results_df["win_rate"],
            results_df["profit_factor"],
            c=results_df["win_rate"],
            cmap="RdYlGn",
            s=80,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5
        )
        ax.set_title("Trading Quality\n(Win Rate vs Profit Factor)")
        ax.set_xlabel("Win Rate")
        ax.set_ylabel("Profit Factor")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Break-even")
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="50% Win Rate")
        ax.legend()

        # 6. Performance distribution
        ax = axes[1, 2]
        sns.histplot(results_df["Final_Score"], bins=15, color="lightgreen", edgecolor="black", ax=ax)
        median = results_df["Final_Score"].median()
        ax.axvline(median, color="red", linestyle="--", label=f"Median: {median:.2f}")
        ax.set_title("Performance Score Distribution")
        ax.set_xlabel("Final Score")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        print(f"💾 Dashboard saved to: {save_path}")
        fig.savefig(save_path, dpi=DPI)

    plt.show()
    plt.close(fig)
