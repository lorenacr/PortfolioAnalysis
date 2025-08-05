"""
Portfolio Analysis Framework - Enhanced Analyzer
===============================================

Enhanced portfolio analyzer with institutional-grade metrics and comprehensive logging.
Preserves all original logging patterns while adding advanced quantitative analysis.
Designed for professional quant trading environments.
"""
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from core_metrics import calculate_portfolio_metrics

# Import enhanced metrics
from custom_config import INPUT_FOLDER, ANALYZE_PORTFOLIO_COLOR, NORMALIZE_PORTFOLIO_COLOR
from data_utils import load_portfolios_from_folder
from reporting import export_results_to_csv, generate_executive_summary, generate_portfolio_recommendations

warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """
    Portfolio Analyzer with Advanced Quantitative Metrics
    
    Implements institutional-grade portfolio analysis with comprehensive logging,
    progress tracking, and advanced risk-adjusted performance metrics.
    """
    
    def __init__(self, normalization_method: str = 'robust_percentile',
                 scoring_method: str = 'topsis', outlier_treatment: str = 'winsorize'):
        """
        Initialize Enhanced Portfolio Analyzer
        
        Args:
            normalization_method: Method for metric normalization
            scoring_method: Scoring methodology (topsis, weighted_sum, etc.)
            outlier_treatment: How to handle outliers (winsorize, cap, remove)
        """
        print("🚀 INITIALIZING ENHANCED PORTFOLIO ANALYSIS ENGINE")
        print("=" * 60)
        
        self.normalization_method = normalization_method
        self.scoring_method = scoring_method
        self.outlier_treatment = outlier_treatment
        self.results_df = pd.DataFrame()
        self.portfolios_data = {}
        
        print(f"   📊 Normalization Method: {normalization_method}")
        print(f"   🎯 Scoring Method: {scoring_method}")
        print(f"   🔧 Outlier Treatment: {outlier_treatment}")
        print("=" * 60)
    
    def load_portfolio_data(self, folder_path: str = INPUT_FOLDER) -> bool:
        """
        Load portfolio data from folder
        
        Args:
            folder_path: Path to folder containing portfolio CSV files
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            self.portfolios_data = load_portfolios_from_folder(folder_path)

            if not self.portfolios_data:
                print("❌ No valid portfolio data found")
                return False

            return True

        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            return False
    
    def normalize_portfolio_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize metrics using robust statistical methods
        
        Args:
            df: DataFrame with calculated metrics
            
        Returns:
            DataFrame with normalized metrics
        """
        try:            
            normalized_df = df.copy()
            
            # Define metrics to normalize (exclude metadata)
            exclude_cols = ['Portfolio', 'Trade_Count', 'Win_Rate', 'Avg_Win', 'Avg_Loss']
            metric_cols = [col for col in df.columns if col not in exclude_cols]
                      
            for col in tqdm(
                    metric_cols,
                    desc="📊 Normalizing",
                    unit="metric",
                    total=len(self.portfolios_data),
                    file=sys.stdout,
                    colour=NORMALIZE_PORTFOLIO_COLOR,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                ):
                try:
                    values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(values) == 0 or values.std() == 0:
                        normalized_df[col] = 0.5  # Neutral score
                        continue
                    
                    if self.normalization_method == 'robust_percentile':
                        # Use percentile-based normalization (more robust to outliers)
                        p25, p75 = values.quantile([0.25, 0.75])
                        iqr = p75 - p25
                        
                        if iqr > 0:
                            normalized_df[col] = (df[col] - values.median()) / iqr
                            # Scale to 0-1 range
                            normalized_df[col] = (normalized_df[col] + 2) / 4
                            normalized_df[col] = normalized_df[col].clip(0, 1)
                        else:
                            normalized_df[col] = 0.5
                    
                    elif self.normalization_method == 'min_max':
                        # Traditional min-max normalization
                        min_val, max_val = values.min(), values.max()
                        if max_val > min_val:
                            normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
                        else:
                            normalized_df[col] = 0.5
                    
                    else:  # z-score normalization
                        normalized_df[col] = (df[col] - values.mean()) / values.std()
                        # Convert to 0-1 scale using sigmoid
                        normalized_df[col] = 1 / (1 + np.exp(-normalized_df[col]))
                
                except Exception as e:
                    print(f"   ⚠️  Normalization failed for {col}: {str(e)}")
                    normalized_df[col] = 0.5
            
            return normalized_df
            
        except Exception as e:
            print(f"   ❌ Normalization failed: {str(e)}")
            return df
    
    def calculate_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite scores using TOPSIS or weighted sum methodology
        
        Args:
            df: DataFrame with normalized metrics
            
        Returns:
            DataFrame with composite scores and rankings
        """
        try:
            # Enhanced weights based on academic literature
            weights = {
                'sharpe_ratio': 0.10, 'sortino_ratio': 0.08, 'calmar_ratio': 0.07,
                'omega_ratio': 0.05, 'total_return': 0.05, 'max_drawdown': 0.08,
                'volatility': 0.05, 'var_95': 0.06, 'cvar_95': 0.06,
                'sharpe_stability': 0.08, 'return_stability': 0.07,
                'recovery_consistency': 0.05, 'regime_consistency': 0.05,
                'tail_ratio': 0.05, 'cvar_99': 0.05, 'max_loss_streak': 0.05
            }
            
            # Benefit vs cost criteria
            benefit_criteria = [
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio',
                'total_return', 'tail_ratio', 'regime_consistency',
                'sharpe_stability', 'return_stability', 'recovery_consistency'
            ]
            
            cost_criteria = [
                'max_drawdown', 'volatility', 'var_95', 'cvar_95', 'cvar_99',
                'max_loss_streak'
            ]
            
            if self.scoring_method == 'topsis':
                scores = self._calculate_topsis_scores(df, weights, benefit_criteria)
            else:
                scores = self._calculate_weighted_scores(df, weights, cost_criteria)
            
            # Add scores and rankings
            df['Final_Score'] = scores
            df['Final_Rank'] = df['Final_Score'].rank(ascending=False, method='min')
            
            # Sort by score
            df = df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"   ❌ Composite score calculation failed: {str(e)}")
            # Fallback to simple Sharpe ranking
            if 'sharpe_ratio' in df.columns:
                df['Final_Score'] = df['sharpe_ratio']
                df['Final_Rank'] = df['sharpe_ratio'].rank(ascending=False, method='min')
            else:
                df['Final_Score'] = 0.5
                df['Final_Rank'] = 1
            return df

    @staticmethod
    def _calculate_topsis_scores(df: pd.DataFrame, weights: Dict[str, float],
                                 benefit_criteria: List[str]) -> pd.Series:
        """
        Calculate TOPSIS scores for portfolio ranking
        
        Args:
            df: Normalized metrics dataframe
            weights: Metric weights dictionary
            benefit_criteria: List of benefit criteria (higher is better)
            
        Returns:
            Series of TOPSIS scores
        """
        try:
            # Get available metrics
            available_metrics = [col for col in weights.keys() if col in df.columns]
            
            if not available_metrics:
                print("   ⚠️  No metrics available for TOPSIS calculation")
                return pd.Series([0.5] * len(df))
            
            # Create decision matrix
            decision_matrix = df[available_metrics].fillna(0.5)
            
            # Apply weights
            weight_vector = np.array([weights[col] for col in available_metrics])
            weighted_matrix = decision_matrix * weight_vector
            
            # Calculate ideal and negative ideal solutions
            ideal_solution = []
            negative_ideal_solution = []
            
            for col in available_metrics:
                if col in benefit_criteria:
                    ideal_solution.append(weighted_matrix[col].max())
                    negative_ideal_solution.append(weighted_matrix[col].min())
                else:  # cost criteria
                    ideal_solution.append(weighted_matrix[col].min())
                    negative_ideal_solution.append(weighted_matrix[col].max())
            
            ideal_solution = np.array(ideal_solution)
            negative_ideal_solution = np.array(negative_ideal_solution)
            
            # Calculate distances to ideal and negative ideal solutions
            distances_to_ideal = []
            distances_to_negative_ideal = []
            
            for idx in range(len(df)):
                alternative = weighted_matrix.iloc[idx].values
                
                # Euclidean distance to ideal solution
                dist_ideal = np.sqrt(np.sum((alternative - ideal_solution) ** 2))
                distances_to_ideal.append(dist_ideal)
                
                # Euclidean distance to negative ideal solution
                dist_negative = np.sqrt(np.sum((alternative - negative_ideal_solution) ** 2))
                distances_to_negative_ideal.append(dist_negative)
            
            # Calculate TOPSIS scores
            distances_to_ideal = np.array(distances_to_ideal)
            distances_to_negative_ideal = np.array(distances_to_negative_ideal)
            
            # Avoid division by zero
            total_distances = distances_to_ideal + distances_to_negative_ideal
            scores = np.where(total_distances > 0, 
                            distances_to_negative_ideal / total_distances, 
                            0.5)
            
            return pd.Series(scores)
            
        except Exception as e:
            print(f"   ❌ TOPSIS calculation failed: {str(e)}")
            return pd.Series([0.5] * len(df))
    
    @staticmethod
    def _calculate_weighted_scores(df: pd.DataFrame, weights: Dict[str, float],
                                   cost_criteria: List[str]) -> pd.Series:
        """
        Calculate weighted sum scores for portfolio ranking
        
        Args:
            df: Normalized metrics dataframe
            weights: Metric weights dictionary
            cost_criteria: List of cost criteria (lower is better)
            
        Returns:
            Series of weighted scores
        """
        try:
            # Get available metrics
            available_metrics = [col for col in weights.keys() if col in df.columns]
            
            if not available_metrics:
                print("   ⚠️  No metrics available for weighted scoring")
                return pd.Series([0.5] * len(df))
            
            scores = pd.Series([0.0] * len(df))
            
            for col in available_metrics:
                weight = weights[col]
                values = df[col].fillna(0.5)
                
                if col in cost_criteria:
                    # For cost criteria, invert the values (1 - normalized_value)
                    values = 1 - values
                
                scores += weight * values
            
            # Normalize scores to 0-1 range
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scores = pd.Series([0.5] * len(df))
            
            return scores
            
        except Exception as e:
            print(f"   ❌ Weighted scoring failed: {str(e)}")
            return pd.Series([0.5] * len(df))
    
    def run_complete_analysis(self, folder_path: str = INPUT_FOLDER) -> bool:
        """
        Run complete portfolio analysis workflow with comprehensive logging
        
        Args:
            folder_path: Path to folder containing portfolio files
            
        Returns:
            Success status
        """
        try:
            print("🚀 STARTING ENHANCED PORTFOLIO ANALYSIS ENGINE")
            print("=" * 60)
            
            # Step 1: Load portfolio data            
            if not self.load_portfolio_data(folder_path):
                print("❌ No portfolios could be analyzed")
                return False
            
            # Step 2: Calculate metrics for all portfolios            
            results = []

            for portfolio_name, portfolio_data in tqdm(
                    self.portfolios_data.items(),
                    desc="📊 Analyzing portfolios",
                    unit="portfolio",
                    total=len(self.portfolios_data),
                    file=sys.stdout,
                    colour=ANALYZE_PORTFOLIO_COLOR,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ):
                metrics = calculate_portfolio_metrics(portfolio_data, portfolio_name)
                if metrics:
                    results.append(metrics)
                        
            if not results:
                print("❌ No portfolios could be analyzed")
                return False
            
            # Create results dataframe
            self.results_df = pd.DataFrame(results)            
            
            # Step 3: Normalize metrics
            normalized_df = self.normalize_portfolio_metrics(self.results_df)
            
            # Step 4: Calculate composite scores
            final_df = self.calculate_composite_scores(normalized_df)
            self.results_df = final_df
            
            print(f"\n✅ Analysis completed for {len(self.results_df)} portfolios")
            
            # Step 5: Generate reports
            self.generate_reports()
            
            return True
            
        except Exception as e:
            print(f"❌ Complete analysis failed: {str(e)}")
            return False
    
    def generate_reports(self) -> None:
        """
        Generate comprehensive analysis reports with preserved logging style
        """
        try:
            if self.results_df.empty:
                print("⚠️  No results available for reporting")
                return
            
            print("📊 Generating comprehensive reports...")
            
            # Generate executive summary
            generate_executive_summary(self.results_df)
            
            # Generate top performers report
            generate_portfolio_recommendations(self.results_df)

            # Export results to CSV
            export_results_to_csv(self.results_df)
            
            print("✅ All reports generated successfully")
            
        except Exception as e:
            print(f"❌ Report generation failed: {str(e)}")

# Enhanced main execution function
def run_enhanced_analysis(folder_path: str = '.') -> bool:
    """
    Main function to run enhanced portfolio analysis
    
    Args:
        folder_path: Path to portfolio data folder
        
    Returns:
        Success status
    """
    try:
        # Initialize enhanced analyzer
        analyzer = PortfolioAnalyzer(
            normalization_method='robust_percentile',
            scoring_method='topsis',
            outlier_treatment='winsorize'
        )
        
        # Run complete analysis
        success = analyzer.run_complete_analysis(folder_path)
        
        if success:
            print("\n🎉 ENHANCED PORTFOLIO ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("📁 Generated Files:")
            print("   • enhanced_portfolio_analysis_results.csv")
            print("   • enhanced_analysis_summary.csv") 
            print("   • enhanced_top_performers.csv")
            print("=" * 60)
        
        return success
        
    except Exception as e:
        print(f"❌ Enhanced analysis execution failed: {str(e)}")
        return False
