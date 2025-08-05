"""
Data Utilities for Portfolio Analysis Framework
==============================================

Handles data loading, cleaning, validation, and preprocessing.
Uses semicolon as the standard CSV delimiter.
"""

import glob
import os
import pandas as pd
import sys 

from tqdm import tqdm
from typing import Dict, Optional

from custom_config import MIN_TRADES_THRESHOLD, LOAD_PORTFOLIO_COLOR

def load_portfolio_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load a single portfolio CSV file using semicolon delimiter
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        df = pd.read_csv(file_path, delimiter=';')
        
        if df.empty:
            print(f"⚠️  Empty file: {file_path}")
            return None
            
        return df
        
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {str(e)}")
        return None

def clean_portfolio_data(df: pd.DataFrame, portfolio_name: str) -> Optional[pd.DataFrame]:
    """
    Clean and validate portfolio data
    
    Args:
        df: Raw portfolio DataFrame
        portfolio_name: Name of the portfolio for logging
        
    Returns:
        Cleaned DataFrame or None if insufficient data
    """
    if df is None or df.empty:
        return None
    
    # Check for required columns
    profit_loss_columns = ['Profit/Loss', 'P&L', 'PnL', 'Profit', 'Loss', 'Return', 'Returns']
    profit_col = None
    
    for col in profit_loss_columns:
        if col in df.columns:
            profit_col = col
            break
    
    if profit_col is None:
        print(f"⚠️  Skipping {portfolio_name}: No 'Profit/Loss' column found")
        print(f"📋 Available columns: {list(df.columns)}")
        return None
    
    # Clean the data
    df_clean = df.copy()
    
    # Convert profit/loss to numeric
    df_clean[profit_col] = pd.to_numeric(df_clean[profit_col], errors='coerce')
    
    # Remove rows with NaN values in the profit / loss column
    df_clean = df_clean.dropna(subset=[profit_col])
    
    # Remove rows with zero values (no meaningful trades)
    df_clean = df_clean[df_clean[profit_col] != 0]
    
    # Check if we have enough data
    if len(df_clean) < MIN_TRADES_THRESHOLD:
        print(f"⚠️  Insufficient data for {portfolio_name}: {len(df_clean)} trades (minimum: {MIN_TRADES_THRESHOLD})")
        return None
    
    # Rename the profit column to the standard name
    df_clean = df_clean.rename(columns={profit_col: 'Profit/Loss'})
    
    return df_clean

def load_portfolios_from_folder(folder_path: str = './') -> Dict[str, pd.DataFrame]:
    """
    Load all portfolio CSV files from a folder
    
    Args:
        folder_path: Path to the folder containing CSV files
        
    Returns:
        Dictionary mapping portfolio names to DataFrames
    """
    print(f"📁 Loading portfolios from: {folder_path}")
    
    # Find all CSV files that look like portfolios
    csv_files = glob.glob(os.path.join(folder_path, "Portfolio*.csv"))
    
    if not csv_files:
        print("⚠️  No Portfolio CSV files found in the specified folder")
        return {}
    
    portfolios = {}
    loaded_count = 0
    
    for file_path in tqdm(
            csv_files, 
            desc="📊 Loading portfolios", 
            unit="files",
            file=sys.stdout,
            colour=LOAD_PORTFOLIO_COLOR,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"):
        portfolio_name = os.path.basename(file_path).replace('.csv', '')
        
        # Load the CSV file
        df = load_portfolio_csv(file_path)
        
        if df is not None:
            # Clean the data
            df_clean = clean_portfolio_data(df, portfolio_name)
            
            if df_clean is not None:
                portfolios[portfolio_name] = df_clean
                loaded_count += 1
    
    return portfolios

def validate_portfolio_data(df: pd.DataFrame, portfolio_name: str) -> bool:
    """
    Validate portfolio data meets minimum requirements
    
    Args:
        df: Portfolio DataFrame
        portfolio_name: Name for logging
        
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        return False
    
    if len(df) < MIN_TRADES_THRESHOLD:
        print(f"⚠️  Insufficient data for {portfolio_name}: {len(df)} trades")
        return False
    
    if 'Profit/Loss' not in df.columns:
        return False
    
    # Check for valid numeric data
    if df['Profit/Loss'].isna().all():
        return False
    
    return True
