"""
Portfolio Analysis Framework
===============================================

Portfolio analyzer with institutional-grade metrics and comprehensive logging.
Designed for professional quant trading environments.
"""

import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from core_metrics import calculate_portfolio_metrics, normalize_portfolio_metrics
from custom_config import INPUT_FOLDER, ANALYZE_PORTFOLIO_COLOR, SCORING_METHOD, COST_CRITERIA, SCORING_WEIGHTS, \
    BENEFIT_CRITERIA
from data_utils import load_portfolios_from_folder
from reporting import export_results_to_csv, generate_executive_summary, generate_portfolio_recommendations
from visualization import create_performance_dashboard

warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """
    Portfolio Analyzer with Advanced Quantitative Metrics
    
    Implements institutional-grade portfolio analysis with comprehensive logging,
    progress tracking, and advanced risk-adjusted performance metrics.
    """

    def __init__(self):
        """ Initialize Enhanced Portfolio Analyzer """

        self.results_df = pd.DataFrame()
        self.portfolios_data = {}

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

    def calculate_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite scores using TOPSIS or weighted sum methodology
        
        Args:
            df: DataFrame with normalized metrics
            
        Returns:
            DataFrame with composite scores and rankings
        """
        try:
            if SCORING_METHOD == 'topsis':
                scores = self._calculate_topsis_scores(df, SCORING_WEIGHTS, BENEFIT_CRITERIA)
            else:
                scores = self._calculate_weighted_scores(df, SCORING_WEIGHTS, COST_CRITERIA)

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
            normalized_df = normalize_portfolio_metrics(self.results_df)

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

    def generate_reports(self):
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

            # Generate dashboard
            create_performance_dashboard(self.results_df)
            print("✅ All reports generated successfully")

        except Exception as e:
            print(f"❌ Report generation failed: {str(e)}")
