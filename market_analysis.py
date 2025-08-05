"""
Market Analysis Module for Portfolio Analysis Framework
======================================================

Handles market regime detection using Gaussian Mixture Models and
advanced statistical analysis for portfolio performance evaluation.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

from custom_config import N_REGIMES, MONTE_CARLO_SIMULATIONS

def detect_market_regimes(returns: pd.Series, n_regimes: int = N_REGIMES) -> Tuple[np.ndarray, Dict]:
    """
    Detect market regimes using Gaussian Mixture Models
    
    Args:
        returns: Series of portfolio returns
        n_regimes: Number of regimes to detect
        
    Returns:
        Tuple of (regime_labels, regime_stats)
    """
    try:
        if len(returns) < n_regimes * 10:  # Need sufficient data
            print("⚠️  Insufficient data for regime detection")
            return np.zeros(len(returns)), {}
        
        # Prepare data for GMM
        X = returns.values.reshape(-1, 1)
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regime_labels = gmm.fit_predict(X)
        
        # Calculate regime statistics
        regime_stats = {}
        for i in range(n_regimes):
            regime_returns = returns[regime_labels == i]
            if len(regime_returns) > 0:
                regime_stats[i] = {
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'count': len(regime_returns),
                    'percentage': len(regime_returns) / len(returns) * 100
                }
        
        return regime_labels, regime_stats
    
    except Exception as e:
        print(f"⚠️  Regime detection failed: {str(e)}")
        return np.zeros(len(returns)), {}

def monte_carlo_validation(returns: pd.Series, n_simulations: int = MONTE_CARLO_SIMULATIONS) -> Dict:
    """
    Perform Monte Carlo validation of portfolio performance
    
    Args:
        returns: Series of portfolio returns
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        Dictionary with validation statistics
    """
    try:
        if len(returns) == 0:
            return {}
        
        # Calculate original metrics
        original_mean = returns.mean()
        original_std = returns.std()
        original_sharpe = (original_mean / original_std) * np.sqrt(252) if original_std > 0 else 0
        
        # Monte Carlo simulations
        simulated_sharpes = []
        simulated_returns = []
        
        for _ in range(n_simulations):
            # Generate random returns with the same statistical properties
            sim_returns = np.random.normal(original_mean, original_std, len(returns))
            sim_sharpe = (np.mean(sim_returns) / np.std(sim_returns)) * np.sqrt(252) if np.std(sim_returns) > 0 else 0
            
            simulated_sharpes.append(sim_sharpe)
            simulated_returns.append(np.mean(sim_returns) * 252)  # Annualized
        
        # Calculate percentiles
        sharpe_percentile = stats.percentileofscore(simulated_sharpes, original_sharpe)
        return_percentile = stats.percentileofscore(simulated_returns, original_mean * 252)
        
        return {
            'sharpe_percentile': sharpe_percentile,
            'return_percentile': return_percentile,
            'simulated_sharpe_mean': np.mean(simulated_sharpes),
            'simulated_sharpe_std': np.std(simulated_sharpes),
            'original_sharpe': original_sharpe,
            'confidence_level': 'High' if sharpe_percentile > 75 else 'Medium' if sharpe_percentile > 50 else 'Low'
        }
    
    except Exception as e:
        print(f"⚠️  Monte Carlo validation failed: {str(e)}")
        return {}

def analyze_return_distribution(returns: pd.Series) -> Dict:
    """
    Analyze the distribution characteristics of returns
    
    Args:
        returns: Series of portfolio returns
        
    Returns:
        Dictionary with distribution statistics
    """
    try:
        if len(returns) == 0:
            return {}
        
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Distribution tests
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Normality test
        _, normality_p_value = stats.jarque_bera(returns)
        is_normal = normality_p_value > 0.05
        
        # Tail analysis
        left_tail = np.percentile(returns, 5)
        right_tail = np.percentile(returns, 95)
        
        return {
            'mean': mean_return,
            'std': std_return,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': is_normal,
            'normality_p_value': normality_p_value,
            'left_tail_5pct': left_tail,
            'right_tail_95pct': right_tail,
            'distribution_type': 'Normal' if is_normal else 'Non-Normal'
        }
    
    except Exception as e:
        print(f"⚠️  Distribution analysis failed: {str(e)}")
        return {}

def calculate_rolling_metrics(returns: pd.Series, window: int = 30) -> Dict[str, pd.Series]:
    """
    Calculate rolling performance metrics
    
    Args:
        returns: Series of portfolio returns
        window: Rolling window size
        
    Returns:
        Dictionary of rolling metrics
    """
    try:
        if len(returns) < window:
            return {}
        
        rolling_metrics = {
            'rolling_mean': returns.rolling(window=window).mean(),
            'rolling_std': returns.rolling(window=window).std(),
            'rolling_sharpe': (returns.rolling(window=window).mean() / 
                             returns.rolling(window=window).std()) * np.sqrt(252),
            'rolling_max': returns.rolling(window=window).max(),
            'rolling_min': returns.rolling(window=window).min()
        }
        
        return rolling_metrics
    
    except Exception as e:
        print(f"⚠️  Rolling metrics calculation failed: {str(e)}")
        return {}

def detect_outliers(returns: pd.Series, method: str = 'iqr') -> Tuple[pd.Series, Dict]:
    """
    Detect outliers in return series
    
    Args:
        returns: Series of portfolio returns
        method: Method for outlier detection ('iqr' or 'zscore')
        
    Returns:
        Tuple of (outlier_mask, outlier_stats)
    """
    try:
        if len(returns) == 0:
            return pd.Series(dtype=bool), {}
        
        if method == 'iqr':
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (returns < lower_bound) | (returns > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(returns))
            outlier_mask = z_scores > 3
        
        else:
            outlier_mask = pd.Series([False] * len(returns), index=returns.index)
        
        outlier_stats = {
            'total_outliers': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(returns)) * 100,
            'method_used': method
        }
        
        return outlier_mask, outlier_stats
    
    except Exception as e:
        print(f"⚠️  Outlier detection failed: {str(e)}")
        return pd.Series([False] * len(returns), index=returns.index), {}
