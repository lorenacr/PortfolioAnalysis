"""
Core Performance Metrics for Portfolio Analysis Framework
========================================================

Contains all performance metric calculations used in portfolio analysis.
Implements institutional-grade financial metrics with academic standards.
Enhanced with advanced stability, tail risk, and regime analysis metrics.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats

from custom_config import CONFIDENCE_LEVEL

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio
    
    Args:
        returns: Series of portfolio returns
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Sharpe ratio
    """
    try:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns / returns.std()) * np.sqrt(252)  # Annualized
    except Exception:
        return 0.0

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino Ratio (focuses on downside deviation)
    
    Args:
        returns: Series of portfolio returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sortino ratio
    """
    try:
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return calculate_sharpe_ratio(returns, risk_free_rate)
        
        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return 0.0
            
        return (excess_returns / downside_deviation) * np.sqrt(252)
    except Exception:
        return 0.0

def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar Ratio (Annual Return / Maximum Drawdown)
    
    Args:
        returns: Series of portfolio returns
        
    Returns:
        Calmar ratio
    """
    try:
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd = calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
    except Exception:
        return 0.0

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate Maximum Drawdown
    
    Args:
        returns: Series of portfolio returns
        
    Returns:
        Maximum drawdown as positive value
    """
    try:
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        return abs(drawdown.min())
    except Exception:
        return 0.0

def calculate_var_cvar(returns: pd.Series, confidence_level: float = None) -> Tuple[float, float]:
    """
    Calculate Value at Risk and Conditional Value at Risk
    
    Args:
        returns: Series of portfolio returns
        confidence_level: Confidence level (default from config)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    try:
        if confidence_level is None:
            confidence_level = CONFIDENCE_LEVEL
        
        if len(returns) == 0:
            return 0.0, 0.0
        
        # Calculate VaR
        var = abs(returns.quantile(confidence_level))
        
        # Calculate CVaR (Expected Shortfall)
        tail_returns = returns[returns <= returns.quantile(confidence_level)]
        cvar = abs(tail_returns.mean()) if len(tail_returns) > 0 else var
        
        return var, cvar
    except Exception:
        return 0.0, 0.0

def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio
    
    Args:
        returns: Series of portfolio returns
        threshold: Return threshold (default 0%)
        
    Returns:
        Omega ratio
    """
    try:
        if len(returns) == 0:
            return 1.0
        
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 1.0
        
        return gains / losses
    except Exception:
        return 1.0

# ============================================================================
# STABILITY METRICS
# ============================================================================

def calculate_sharpe_stability(returns: pd.Series, window: int = 63) -> float:
    """
    Calculate Rolling Sharpe Ratio Stability
    Measures consistency of risk-adjusted returns over time
    
    Args:
        returns: Series of portfolio returns
        window: Rolling window size (default 63 = ~3 months)
        
    Returns:
        Stability score (0-1, higher is better)
    """
    try:
        if len(returns) < window * 2:
            return 0.0
        
        rolling_sharpe = []
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            if window_returns.std() > 0:
                sharpe = (window_returns.mean() / window_returns.std()) * np.sqrt(252)
                rolling_sharpe.append(sharpe)
        
        if len(rolling_sharpe) == 0:
            return 0.0
        
        # Stability is inverse of coefficient of variation
        rolling_sharpe = pd.Series(rolling_sharpe)
        if rolling_sharpe.std() == 0:
            return 1.0
        
        cv = rolling_sharpe.std() / abs(rolling_sharpe.mean()) if rolling_sharpe.mean() != 0 else float('inf')
        return 1.0 / (1.0 + cv)
    except Exception:
        return 0.0

def calculate_return_stability(returns: pd.Series) -> float:
    """
    Calculate Return Stability (inverse of coefficient of variation)
    
    Args:
        returns: Series of portfolio returns
        
    Returns:
        Stability score (0-1, higher is better)
    """
    try:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        cv = returns.std() / abs(returns.mean()) if returns.mean() != 0 else float('inf')
        return 1.0 / (1.0 + cv)
    except Exception:
        return 0.0

def calculate_recovery_consistency(returns: pd.Series, drawdown_threshold: float = 0.01) -> float:
    """
    Calculate Consistency of Drawdown Recovery Patterns
    Measures how consistently the portfolio recovers from drawdowns
    
    Args:
        returns: Series of portfolio returns
        drawdown_threshold: Minimum drawdown to consider (default 1%)
        
    Returns:
        Consistency score (0-1, higher is better)
    """
    try:
        if len(returns) < 50:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        
        # Find drawdown periods
        in_drawdown = drawdown < -drawdown_threshold
        recovery_times = []
        
        i = 0
        while i < len(in_drawdown):
            if in_drawdown.iloc[i]:
                # Start of drawdown
                start = i
                while i < len(in_drawdown) and in_drawdown.iloc[i]:
                    i += 1
                # End of drawdown, measure recovery time
                if i < len(in_drawdown):
                    recovery_times.append(i - start)
            i += 1
        
        if len(recovery_times) == 0:
            return 1.0
        
        # Consistency is inverse of coefficient of variation of recovery times
        recovery_times = pd.Series(recovery_times)
        if recovery_times.std() == 0:
            return 1.0
        
        cv = recovery_times.std() / recovery_times.mean()
        return 1.0 / (1.0 + cv)
    except Exception:
        return 0.0

# ============================================================================
# TAIL RISK METRICS
# ============================================================================

def calculate_tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
    """
    Calculate Tail Ratio (upside tail / downside tail)
    Measures asymmetry in extreme returns
    
    Args:
        returns: Series of portfolio returns
        percentile: Percentile for tail definition (default 5%)
        
    Returns:
        Tail ratio (>1 indicates positive skew)
    """
    try:
        if len(returns) == 0:
            return 1.0
        
        upper_tail = returns.quantile(1 - percentile)
        lower_tail = abs(returns.quantile(percentile))
        
        if lower_tail == 0:
            return 1.0
        
        return upper_tail / lower_tail
    except Exception:
        return 1.0

def calculate_max_loss_streak(returns: pd.Series) -> int:
    """
    Calculate Maximum Consecutive Loss Streak
    
    Args:
        returns: Series of portfolio returns
        
    Returns:
        Maximum number of consecutive losing periods
    """
    try:
        if len(returns) == 0:
            return 0
        
        current_streak = 0
        max_streak = 0
        
        for ret in returns:
            if ret < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    except Exception:
        return 0

def calculate_multi_level_cvar(returns: pd.Series) -> dict:
    """
    Calculate CVaR at multiple confidence levels
    
    Args:
        returns: Series of portfolio returns
        
    Returns:
        Dictionary with CVaR at different confidence levels
    """
    try:
        if len(returns) == 0:
            return {'cvar_90': 0.0, 'cvar_95': 0.0, 'cvar_99': 0.0}
        
        levels = [0.10, 0.05, 0.01]
        result = {}
        
        for level in levels:
            tail_returns = returns[returns <= returns.quantile(level)]
            cvar = abs(tail_returns.mean()) if len(tail_returns) > 0 else 0.0
            result[f'cvar_{int((1-level)*100)}'] = cvar
        
        return result
    except Exception:
        return {'cvar_90': 0.0, 'cvar_95': 0.0, 'cvar_99': 0.0}

# ============================================================================
# REGIME ANALYSIS METRICS
# ============================================================================

def calculate_regime_consistency(returns: pd.Series, vol_threshold: float = 0.02) -> float:
    """
    Calculate Performance Consistency Across Volatility Regimes
    Measures how consistently the strategy performs in different market conditions
    
    Args:
        returns: Series of portfolio returns
        vol_threshold: Volatility threshold for regime definition
        
    Returns:
        Consistency score (0-1, higher is better)
    """
    try:
        if len(returns) < 126:  # Need at least 6 months
            return 0.0
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(21).std() * np.sqrt(252)
        
        # Define regimes based on median volatility
        median_vol = rolling_vol.median()
        high_vol_mask = rolling_vol > median_vol + vol_threshold
        low_vol_mask = rolling_vol < median_vol - vol_threshold
        
        # Calculate Sharpe in each regime
        high_vol_returns = returns[high_vol_mask]
        low_vol_returns = returns[low_vol_mask]
        
        if len(high_vol_returns) < 21 or len(low_vol_returns) < 21:
            return 0.0
        
        high_vol_sharpe = calculate_sharpe_ratio(high_vol_returns)
        low_vol_sharpe = calculate_sharpe_ratio(low_vol_returns)
        
        # Consistency is based on similarity of Sharpe ratios
        if high_vol_sharpe == 0 and low_vol_sharpe == 0:
            return 0.0
        
        avg_sharpe = (high_vol_sharpe + low_vol_sharpe) / 2
        if avg_sharpe == 0:
            return 0.0
        return abs(avg_sharpe)
    except Exception:
        return 0.0
        
        # Consistency is inverse of relative difference
        diff = abs(high_vol_sharpe - low_vol_sharpe) / avg_sharpe
        return 1.0 / (1.0 + diff)
    except Exception:
        return 0.0

# ============================================================================
# COMPREHENSIVE METRICS CALCULATION
# ============================================================================

def calculate_basic_metrics(returns: pd.Series) -> dict:
    """
    Calculate basic portfolio metrics (original function preserved)
    
    Args:
        returns: Series of portfolio returns
        
    Returns:
        Dictionary of basic metrics
    """
    try:
        if len(returns) == 0:
            return {}
        
        var_95, cvar_95 = calculate_var_cvar(returns, 0.05)
        
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'sortino_ratio': calculate_sortino_ratio(returns),
            'calmar_ratio': calculate_calmar_ratio(returns),
            'max_drawdown': calculate_max_drawdown(returns),
            'var_95': var_95,
            'cvar_95': cvar_95,
            'omega_ratio': calculate_omega_ratio(returns)
        }
        
        return metrics
    except Exception as e:
        print(f"⚠️  Basic metrics calculation failed: {str(e)}")
        return {}

def calculate_all_enhanced_metrics(returns: pd.Series) -> dict:
    """
    Calculate comprehensive enhanced metrics for portfolio analysis
    Includes all basic metrics plus advanced stability, tail risk, and regime metrics
    
    Args:
        returns: Series of portfolio returns
        
    Returns:
        Dictionary of all enhanced metrics
    """
    try:
        if len(returns) == 0:
            print("⚠️  Empty returns series provided")
            return {}
        
        # Start with basic metrics
        metrics = calculate_basic_metrics(returns)
        
        if not metrics:
            print("❌ Basic metrics calculation failed")
            return {}
        
        # Add enhanced stability metrics
        metrics.update({
            'sharpe_stability': calculate_sharpe_stability(returns),
            'return_stability': calculate_return_stability(returns),
            'recovery_consistency': calculate_recovery_consistency(returns)
        })
        
        # Add enhanced tail risk metrics
        metrics.update({
            'tail_ratio': calculate_tail_ratio(returns),
            'max_loss_streak': calculate_max_loss_streak(returns)
        })
        
        # Add multi-level CVaR
        cvar_metrics = calculate_multi_level_cvar(returns)
        metrics.update(cvar_metrics)
        
        # Add regime analysis
        metrics['regime_consistency'] = calculate_regime_consistency(returns)
        
        return metrics
        
    except Exception as e:
        print(f"❌ Enhanced metrics calculation failed: {str(e)}")
        return calculate_basic_metrics(returns)  # Fallback to basic metrics

def get_metric_descriptions() -> dict:
    """
    Get descriptions of all available metrics for reporting
    
    Returns:
        Dictionary mapping metric names to descriptions
    """
    return {
        # Basic Performance Metrics
        'total_return': 'Total cumulative return over the period',
        'volatility': 'Annualized volatility (standard deviation)',
        'sharpe_ratio': 'Risk-adjusted return (Sharpe Ratio)',
        'sortino_ratio': 'Downside risk-adjusted return (Sortino Ratio)',
        'calmar_ratio': 'Return to maximum drawdown ratio',
        'max_drawdown': 'Maximum peak-to-trough decline',
        'var_95': 'Value at Risk at 95% confidence level',
        'cvar_95': 'Conditional Value at Risk at 95% confidence level',
        'omega_ratio': 'Probability-weighted ratio of gains to losses',
        
        # Enhanced Stability Metrics
        'sharpe_stability': 'Consistency of risk-adjusted returns over time',
        'return_stability': 'Stability of return patterns (inverse CV)',
        'recovery_consistency': 'Consistency of drawdown recovery patterns',
        
        # Enhanced Tail Risk Metrics
        'tail_ratio': 'Ratio of upside to downside tail returns',
        'max_loss_streak': 'Maximum consecutive losing periods',
        'cvar_90': 'Conditional Value at Risk at 90% confidence level',
        'cvar_99': 'Conditional Value at Risk at 99% confidence level',
        
        # Regime Analysis
        'regime_consistency': 'Performance consistency across volatility regimes'
    }

def calculate_portfolio_metrics(portfolio_data: pd.DataFrame,
                                portfolio_name: str) -> Optional[Dict]:
    """
    Calculate comprehensive metrics for a single portfolio
    
    Args:
        portfolio_data: Portfolio trading data
        portfolio_name: Portfolio identifier
        
    Returns:
        Dictionary of calculated metrics or None if failed
    """
    try:
        # Convert P&L to returns
        pnl_series = pd.to_numeric(portfolio_data['Profit/Loss'], errors='coerce').dropna()

        if len(pnl_series) == 0:
            print(f"   ❌ {portfolio_name}: No valid P&L data")
            return None

        # Calculate returns (assuming P&L is already in return format)
        returns = pnl_series / 100.0 if pnl_series.abs().mean() > 1 else pnl_series

        # Calculate all enhanced metrics
        metrics = calculate_all_enhanced_metrics(returns)

        if not metrics:
            print(f"   ❌ {portfolio_name}: Metrics calculation failed")
            return None

        # Add portfolio metadata
        metrics['Portfolio'] = portfolio_name
        metrics['Trade_Count'] = len(pnl_series)
        metrics['Win_Rate'] = (pnl_series > 0).mean()
        metrics['Avg_Win'] = pnl_series[pnl_series > 0].mean() if (pnl_series > 0).any() else 0
        metrics['Avg_Loss'] = pnl_series[pnl_series < 0].mean() if (pnl_series < 0).any() else 0

        return metrics

    except Exception as e:
        print(f"   ❌ {portfolio_name}: Analysis failed - {str(e)}")
        return None

