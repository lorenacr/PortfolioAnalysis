import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import re
from pathlib import Path

def extract_portfolio_metrics_from_pdf(pdf_path):
    """
    Extracts metrics from PDF 
    Returns a dictionary containing the main metrics
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    # Patterns to extract metrics
    metrics = {}
    
    # CAGR
    cagr_match = re.search(r'CAGR[:\s]*([0-9.]+)%?', text, re.IGNORECASE)
    if cagr_match:
        metrics['CAGR'] = float(cagr_match.group(1))
    
    # Sharpe Ratio
    sharpe_match = re.search(r'Sharpe[:\s]*([0-9.]+)', text, re.IGNORECASE)
    if sharpe_match:
        metrics['Sharpe'] = float(sharpe_match.group(1))
    
    # Win Rate
    win_match = re.search(r'Win[:\s]*Rate[:\s]*([0-9.]+)%?', text, re.IGNORECASE)
    if win_match:
        metrics['Win_Rate'] = float(win_match.group(1))
    
    # Profit Factor
    pf_match = re.search(r'Profit[:\s]*Factor[:\s]*([0-9.]+)', text, re.IGNORECASE)
    if pf_match:
        metrics['Profit_Factor'] = float(pf_match.group(1))
    
    # Max Drawdown (USD)
    dd_match = re.search(r'(?:Max[:\s]*)?Drawdown[:\s]*\$?([0-9.]+)', text, re.IGNORECASE)
    if dd_match:
        metrics['Max_DD_USD'] = float(dd_match.group(1))
    
    return metrics

def analyze_portfolio_quantitatively(portfolio_data, weights=None):
    """
    Portfolio quantitative analysis
    
    Args:
        portfolio_data: Dictionary ou DataFrame with portfolio metrics
        weights: Dictionary with weights for final score (optional)
    
    Returns:
        DataFrame with full analysis and rankings
    """
    
    # Default weights 
    if weights is None:
        weights = {
            'risk_efficiency': 0.25,
            'consistency': 0.20,
            'stability': 0.20,
            'institutional_grade': 0.15,
            'sharpe_ratio': 0.10,
            'max_dd_protection': 0.10
        }
    
    # Convert to DataFrame if necessary
    if isinstance(portfolio_data, dict):
        df = pd.DataFrame(portfolio_data).T
        df.index.name = 'Portfolio'
        df = df.reset_index()
    else:
        df = portfolio_data.copy()
    
    print("📊 Starting quantitative analysis...")
    print(f"   • {len(df)} loaded portfolios")
    print(f"   • {len(df.columns)-1} metrics by portfolio")
    
    # 1. Adjusted risk metrics 
    df['Return_per_Risk'] = df['CAGR'] / (df['Max_DD_USD'] / 100)
    df['Sharpe_per_DD'] = df['Sharpe'] / (df['Max_DD_USD'] / 1000)
    df['Risk_Efficiency'] = (df['CAGR'] * df['Sharpe']) / df['Max_DD_USD']
    
    # 2. Consistency of returns and stability
    df['Consistency_Score'] = df['Win_Rate'] * df['Profit_Factor'] / 100
    df['Stability_Index'] = (df['Win_Rate'] / 100) * df['Sharpe'] * (df['CAGR'] / 10)
    
    # 3. Institutional quality measurement
    df['Institutional_Grade'] = (
        (df['Sharpe'] >= 2.0).astype(int) * 3 +
        (df['Win_Rate'] >= 60.0).astype(int) * 2 +
        (df['Profit_Factor'] >= 1.5).astype(int) * 2 +
        (df['Max_DD_USD'] <= 400).astype(int) * 3
    )
    
    df['Quality_Tier'] = pd.cut(df['Institutional_Grade'], 
                               bins=[0, 3, 6, 8, 10], 
                               labels=['C', 'B', 'A', 'A+'])
    
    # 4. Weighted final score
    # Normalize metrics (0-1)
    df['risk_eff_norm'] = df['Risk_Efficiency'] / df['Risk_Efficiency'].max()
    df['consistency_norm'] = df['Consistency_Score'] / df['Consistency_Score'].max()
    df['stability_norm'] = df['Stability_Index'] / df['Stability_Index'].max()
    df['institutional_norm'] = df['Institutional_Grade'] / df['Institutional_Grade'].max()
    df['sharpe_norm'] = df['Sharpe'] / df['Sharpe'].max()
    df['dd_protection_norm'] = (df['Max_DD_USD'].max() - df['Max_DD_USD']) / (df['Max_DD_USD'].max() - df['Max_DD_USD'].min())
    
    # Final score
    df['Quant_Score'] = (
        df['risk_eff_norm'] * weights['risk_efficiency'] +
        df['consistency_norm'] * weights['consistency'] +
        df['stability_norm'] * weights['stability'] +
        df['institutional_norm'] * weights['institutional_grade'] +
        df['sharpe_norm'] * weights['sharpe_ratio'] +
        df['dd_protection_norm'] * weights['max_dd_protection']
    )
    
    # Final ranking
    df_final = df.sort_values('Quant_Score', ascending=False).reset_index(drop=True)
    df_final['Rank'] = range(1, len(df_final) + 1)
    
    return df_final

def generate_portfolio_report(df_analyzed, top_n=10):
    """
    Generates detailed report for best portfolios
    """
    print("🏆 Quantitative report - top portfolios")
    print("="*60)
    
    top_portfolios = df_analyzed.head(top_n)
    
    print(f"RANKING TOP {top_n}:")
    print("-" * 85)
    print(f"{'Rank':<4} {'Portfolio':<15} {'Score':<7} {'CAGR':<6} {'Sharpe':<7} {'Win%':<6} {'PF':<5} {'DD':<8}")
    print("-" * 85)
    
    for _, row in top_portfolios.iterrows():
        print(f"{row['Rank']:<4} {row['Portfolio']:<15} {row['Quant_Score']:<7.3f} {row['CAGR']:<6.1f} {row['Sharpe']:<7.2f} {row['Win_Rate']:<6.1f} {row['Profit_Factor']:<5.2f} ${row['Max_DD_USD']:<7.0f}")
    
    # Winner analysis 
    winner = top_portfolios.iloc[0]
    print(f"\n🥇 WINNER: {winner['Portfolio']}")
    print(f"   Quantitative Score: {winner['Quant_Score']:.3f}")
    print(f"   CAGR: {winner['CAGR']:.1f}%")
    print(f"   Sharpe ratio: {winner['Sharpe']:.2f}")
    print(f"   Win Rate: {winner['Win_Rate']:.1f}%")
    print(f"   Profit Factor: {winner['Profit_Factor']:.2f}")
    print(f"   Max Drawdown: ${winner['Max_DD_USD']:.0f}")
    print(f"   Institutional quality: {winner['Institutional_Grade']:.0f}/10")
    
    return top_portfolios

def batch_analyze_portfolios(pdf_folder_path, output_file="portfolio_analysis_results.csv"):
    """
    Analyzes multiple portfolios in batch from PDF files
    
    Args:
        pdf_folder_path: Path to PDF folder
        output_file: Output file name
    
    Returns:
        DataFrame containing full analysis
    """
    print(f"🔄 Batch analysis - processing PDFs...")
    
    pdf_files = list(Path(pdf_folder_path).glob("*.pdf"))
    print(f"   • {len(pdf_files)} PDF files found")
    
    if len(pdf_files) == 0:
        print("❌ None PDF found!")
        return None
    
    portfolio_data = {}
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"   Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        
        try:
            metrics = extract_portfolio_metrics_from_pdf(pdf_file)
            if metrics:
                portfolio_name = pdf_file.stem  # Filename without extension
                portfolio_data[portfolio_name] = metrics
            else:
                print(f"   ⚠️ It was not possivle to extract metrics from {pdf_file.name}")
        except Exception as e:
            print(f"   ❌ Error to process {pdf_file.name}: {e}")
    
    if not portfolio_data:
        print("❌ No metric extracted from PDFs!")
        return None
    
    print(f"✅ {len(portfolio_data)} portfolios processed with success")
    
    # Quantitative analysis
    df_analyzed = analyze_portfolio_quantitatively(portfolio_data)
    
    # Save results
    df_analyzed.to_csv(output_file, index=False)
    print(f"💾 Results saved in: {output_file}")
    
    # Generate report
    generate_portfolio_report(df_analyzed)
    
    return df_analyzed

# Use example:
# df_results = batch_analyze_portfolios("./portfolio_pdfs/", "analysis_591_portfolios.csv")
