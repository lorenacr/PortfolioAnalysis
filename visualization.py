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

    No console spam per subplot; callers (e.g. ``portfolio_analyzer``) own user-facing logs.

    Layout: 2 panels on row 1 (Risk–Return + MC drawdown), 2 on row 2 (Sharpe + Trading
    quality), 1 full-width on row 3 (Final score distribution). This avoids crowding
    the colorbar against adjacent plots.

    Saves the figure to disk and opens an interactive window via
    ``matplotlib.pyplot.show()`` (GUI backend required; close the window to unblock).

    Args:
        results_df: DataFrame from the analysis pipeline (must include ``mc_max_dd_p95``).
        save_path: Path to save dashboard image

    Returns:
        bool: Success status
    """
    rc_vis = {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }

    with plt.rc_context(rc_vis):
        fig = plt.figure(figsize=FIGURE_SIZE)
        # 3 rows × 2 cols: (2+2) + (1 full width)
        gs = fig.add_gridspec(
            3, 2,
            hspace=0.40,
            wspace=0.30,
            height_ratios=[1.0, 1.0, 0.95],
        )
        fig.suptitle(
            "Portfolio Performance Dashboard — Comprehensive Analysis",
            fontsize=16,
            y=0.995,
            fontweight="bold",
        )

        # Row 0, left: Risk-Return + dedicated colorbar column (no neighbor overlap)
        gs_rr = gs[0, 0].subgridspec(1, 2, width_ratios=[1, 0.10], wspace=0.18)
        ax = fig.add_subplot(gs_rr[0, 0])
        cax = fig.add_subplot(gs_rr[0, 1])
        scatter = ax.scatter(
            results_df["volatility"],
            results_df["cagr"],
            c=results_df["sharpe_ratio"],
            cmap="RdYlGn",
            s=80,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.set_title("Risk-Return Efficiency\n(Color = Sharpe Ratio)")
        ax.set_xlabel("Volatility (Risk)")
        ax.set_ylabel("CAGR")
        ax.grid(True, alpha=0.3)
        cb = fig.colorbar(scatter, cax=cax)
        cb.set_label("Sharpe ratio\n(Higher = Better)", fontsize=10)
        cax.tick_params(labelsize=9)

        # Row 0, right: Monte Carlo drawdown (full half-width, away from colorbar)
        ax = fig.add_subplot(gs[0, 1])
        sns.histplot(
            results_df["mc_max_dd_p95"],
            bins=15,
            kde=True,
            color="lightcoral",
            edgecolor="black",
            ax=ax,
        )
        ax.set_title("MC Max Drawdown (95%)\nTrade reordering — across portfolios")
        ax.set_xlabel("95th percentile max DD (Monte Carlo)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Row 1, left: Sharpe distribution
        ax = fig.add_subplot(gs[1, 0])
        sns.histplot(results_df["sharpe_ratio"], bins=15, kde=True, color="teal", ax=ax)
        ax.set_title("Sharpe Ratio Distribution")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Row 1, right: Trading quality
        ax = fig.add_subplot(gs[1, 1])
        ax.scatter(
            results_df["win_rate"],
            results_df["profit_factor"],
            c=results_df["win_rate"],
            cmap="RdYlGn",
            s=80,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.set_title("Trading Quality\n(Win Rate vs Profit Factor)")
        ax.set_xlabel("Win Rate")
        ax.set_ylabel("Profit Factor")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Break-even")
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="50% Win Rate")
        ax.legend(loc="best")

        # Row 2: Performance distribution (spans both columns)
        ax = fig.add_subplot(gs[2, :])
        sns.histplot(results_df["Final_Score"], bins=15, color="lightgreen", edgecolor="black", ax=ax)
        median = results_df["Final_Score"].median()
        ax.axvline(median, color="red", linestyle="--", label=f"Median: {median:.2f}")
        ax.set_title("Performance Score Distribution")
        ax.set_xlabel("Final Score")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(save_path, dpi=DPI)

    plt.show()
    plt.close(fig)
