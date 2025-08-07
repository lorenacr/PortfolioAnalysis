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

from custom_config import TRADING_DAYS_PER_YEAR, INITIAL_CAPITAL, ANNUAL_RISK_FREE_RATE

SQRT_TRADING_DAYS_PER_YEAR = np.sqrt(252)
DAILY_RISK_FREE_RATE = (1 + ANNUAL_RISK_FREE_RATE)**(1 / TRADING_DAYS_PER_YEAR) - 1

def calculate_sharpe_ratio(returns: pd.Series, volatility: float) -> float:
    """
    Calculate Sharpe Ratio
    
    Args:
        returns: Series of portfolio daily returns
        volatility: Annualized volatility 
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - DAILY_RISK_FREE_RATE
    annualized_mean_return = excess_returns.mean() * TRADING_DAYS_PER_YEAR
    
    return annualized_mean_return / volatility

def calculate_sortino_ratio(returns: pd.Series) -> float:
    """
    Calculate Sortino Ratio (focuses on downside deviation)
    
    Args:
        returns: Series of portfolio daily returns
        
    Returns:
        Sortino ratio
    """
    try:
        excess_returns = returns - DAILY_RISK_FREE_RATE
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        downside_std = downside_returns.std()
        if downside_std == 0 or pd.isna(downside_std):
            return 0.0

        return excess_returns.mean() / downside_std
    
    except Exception:
        return 0.0

def calculate_calmar_ratio(cagr: float, max_drawdown: float, risk_free_rate=ANNUAL_RISK_FREE_RATE) -> float:
    """
    Calculate Calmar Ratio (Annual Return / Maximum Drawdown)
    
    Args:
        cagr: Compound Annual Growth Rate
        max_drawdown: Max drawdown percent
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Calmar ratio
    """
    return (cagr - risk_free_rate) / max_drawdown

def calculate_cagr(pnl_series: pd.Series, total_years_of_trading: float, initial_capital=INITIAL_CAPITAL) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR)
    Link to CAGR formula: 
		http://www.investopedia.com/ask/answers/071014/what-formula-calculating-compound-annual-growth-rate-cagr-excel.asp
    
    Args:
        pnl_series: Series of portfolio PnL
        initial_capital: Base capital 
        total_years_of_trading: Total trading history in years
        
    Returns:
        float: CAGR as decimal (e.g., 0.25 = 25%)
    """
    try:        
        final_capital = initial_capital + pnl_series.sum()
        temp1 = final_capital / initial_capital
        temp2 = 1.0 / total_years_of_trading
        
        return pow(temp1, temp2) - 1.0

    except Exception as e:
        print(f"⚠️ Calculation error: {e}")
        return 0.0

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate Maximum Drawdown
    
    Args:
        returns: Series of portfolio daily returns
        
    Returns:
        Maximum drawdown as a positive value
    """
    try:
        # Calculate cumulative returns (equity curve)
        cumulative_returns = (1 + returns).cumprod()

        # Calculate running maximum (peak)
        peak = cumulative_returns.expanding().max()

        # Calculate drawdown from peak
        drawdown = (cumulative_returns - peak) / peak

        # Return maximum drawdown as positive value
        return abs(drawdown.min())
    
    except Exception:
        return 0.0

def calculate_var_cvar(returns: pd.Series, percentile: float = 0.05) -> Tuple[float, float]:
    """
    Calculate Value at Risk and Conditional Value at Risk
    
    Args:
        returns: Series of portfolio daily returns
        percentile: Confidence level (default from config)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    try:
        # Calculate VaR
        var = abs(returns.quantile(percentile))

        # Calculate CVaR (Expected Shortfall)
        tail_returns = returns[returns <= returns.quantile(percentile)]
        cvar = abs(tail_returns.mean()) if len(tail_returns) > 0 else var

        return var, cvar
    
    except Exception:
        return 0.0, 0.0

def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio
    
    Args:
        returns: Series of portfolio daily returns
        threshold: Return threshold (default 0%)
        
    Returns:
        Omega ratio
    """
    try:
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
        returns: Series of portfolio daily returns
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
                sharpe = window_returns.mean() / window_returns.std()
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
        returns: Series of portfolio daily returns
        
    Returns:
        Stability score (0-1, higher is better)
    """
    try:
        if returns.std() == 0:
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
        returns: Series of portfolio daily returns
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
        returns: Series of portfolio daily returns
        percentile: Percentile for tail definition (default 5%)
        
    Returns:
        Tail ratio (>1 indicates positive skew)
    """
    try:
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
        returns: Series of portfolio daily returns
        
    Returns:
        Maximum number of consecutive losing periods
    """
    try:
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
        returns: Series of portfolio daily returns
        
    Returns:
        Dictionary with CVaR at different confidence levels
    """
    try:
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

def calculate_regime_consistency(returns: pd.Series, volatility: float, vol_threshold=0.00) -> float:
    """
    Calculate Performance Consistency Across Volatility Regimes
    Measures how consistently the strategy performs in different market conditions
    
    Args:
        returns: Series of portfolio daily returns
        volatility: Annualized volatility 
        vol_threshold: Volatility threshold for regime definition
        
    Returns:
        Consistency score (0-1, higher is better)
    """
    try:
        if len(returns) < 126:  # Need at least 6 months
            return 0.0

        # Calculate rolling volatility
        rolling_vol = returns.rolling(21).std()

        # Define regimes based on median volatility
        median_vol = rolling_vol.median()
        high_vol_mask = rolling_vol > median_vol + vol_threshold
        low_vol_mask = rolling_vol < median_vol - vol_threshold

        # Calculate Sharpe in each regime
        high_vol_returns = returns[high_vol_mask]
        low_vol_returns = returns[low_vol_mask]

        if len(high_vol_returns) < 21 or len(low_vol_returns) < 21:
            return 0.0

        high_vol_sharpe = calculate_sharpe_ratio(high_vol_returns, volatility)
        low_vol_sharpe = calculate_sharpe_ratio(low_vol_returns, volatility)

        # Consistency is based on the similarity of Sharpe ratios
        if high_vol_sharpe == 0 and low_vol_sharpe == 0:
            return 0.0

        avg_sharpe = (high_vol_sharpe + low_vol_sharpe) / 2
        if avg_sharpe == 0:
            return 0.0

        # Consistency is inverse of relative difference
        diff = abs(high_vol_sharpe - low_vol_sharpe) / avg_sharpe
        return 1.0 / (1.0 + diff)
    
    except Exception:
        return 0.0

# ============================================================================
# COMPREHENSIVE METRICS CALCULATION
# ============================================================================
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

        # Calculate returns
        daily_pct_returns = convert_pnl_to_returns(portfolio_data)['dailyPctReturns']
        if len(daily_pct_returns) == 0:
            print(f"   ❌ {portfolio_name}: Failed to convert PnL to returns")
            return {}
        
        # Calculate total trading history in years
        start_date = portfolio_data['Open time'].min()
        end_date = portfolio_data['Close time'].max()
        total_years_of_trading = (end_date - start_date).days / 365
        
        # Calculate key metrics 
        cagr = calculate_cagr(pnl_series, total_years_of_trading)
        volatility = (daily_pct_returns - DAILY_RISK_FREE_RATE).std() * SQRT_TRADING_DAYS_PER_YEAR
        max_drawdown = calculate_max_drawdown(daily_pct_returns)
        var_95, cvar_95 = calculate_var_cvar(daily_pct_returns)
        
        metrics = {
            # Portfolio name
            'portfolio': portfolio_name,
            
            # Basic metrics
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': calculate_sharpe_ratio(daily_pct_returns, volatility),
            'sortino_ratio': calculate_sortino_ratio(daily_pct_returns),
            'calmar_ratio': calculate_calmar_ratio(cagr, max_drawdown),
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'omega_ratio': calculate_omega_ratio(daily_pct_returns),

            # Stability metrics
            'sharpe_stability': calculate_sharpe_stability(daily_pct_returns),
            'return_stability': calculate_return_stability(daily_pct_returns),
            'recovery_consistency': calculate_recovery_consistency(daily_pct_returns),
            
            # Tail risk metrics
            'tail_ratio': calculate_tail_ratio(daily_pct_returns),
            'max_loss_streak': calculate_max_loss_streak(daily_pct_returns),
            
            # Regime metrics
            'regime_consistency': calculate_regime_consistency(daily_pct_returns, volatility),
            
            # Portfolio metadata
            'trade_count': len(pnl_series),
            'win_rate': (pnl_series > 0).mean(),
            'profit_factor': pnl_series[pnl_series > 0].sum() / abs(pnl_series[pnl_series < 0].sum()),
            'avg_win': pnl_series[pnl_series > 0].mean(),
            'avg_loss': pnl_series[pnl_series < 0].mean(),
        }

        # Add multi-level CVaR
        cvar_metrics = calculate_multi_level_cvar(daily_pct_returns)
        metrics.update(cvar_metrics)

        return metrics

    except Exception as e:
        print(f"   ❌ {portfolio_name}: Analysis failed - {str(e)}")
        return None

def convert_pnl_to_returns(portfolio_data: pd.DataFrame, initial_capital=INITIAL_CAPITAL) -> pd.DataFrame:
    """
    Convert PnL to returns using balance-based method:
    - Groups PnL by day
    - Calculates cumulative equity from initial capital (initial_capital + PnL)
    - Returns the percentage change on the balance (dailyPctReturns)
    
    Args:
        portfolio_data: Portfolio trading data
        initial_capital: base capital for conversion 
    
    Returns:
        pd.Series: normalized returns
    """
    try:
        # Group PnL by day (sum PnL of all trades closed in the same day)
        daily_percent_returns = (portfolio_data.groupby(portfolio_data['Close time'].dt.date)['Profit/Loss']
                                 .sum()
                                 .reset_index()
                                 .rename(columns={'Close time': 'Date'}))

        # Calculate Equity (initial capital + accumulated PnL)
        daily_percent_returns['Equity'] = initial_capital + daily_percent_returns['Profit/Loss'].cumsum()

        # Calculate daily percentage change (returns)
        daily_percent_returns['dailyPctReturns'] = daily_percent_returns['Equity'].pct_change().dropna()
        
        return daily_percent_returns
    
    except Exception as e:
        print(f"⚠️ Conversion error: {e}")
        return pd.DataFrame()