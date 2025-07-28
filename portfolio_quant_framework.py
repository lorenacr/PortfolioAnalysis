# Progress bar clean version
framework_clean_progress = ''
import pandas as pd
import fitz  # PyMuPDF
import re
from pathlib import Path
from tqdm import tqdm
import sys

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
    sharpe_match = re.search(r'Sharpe[:\s]*Ratio[:\s]*([0-9.]+)', text, re.IGNORECASE)
    if sharpe_match:
        metrics['Sharpe'] = float(sharpe_match.group(1))
    
    # Win Rate
    win_match = re.search(r'Winning[:\s]*Percentage[:\s]*([0-9.]+)%?', text, re.IGNORECASE)
    if win_match:
        metrics['Win_Rate'] = float(win_match.group(1))
    
    # Profit Factor
    pf_match = re.search(r'Profit[:\s]*Factor[:\s]*([0-9.]+)', text, re.IGNORECASE)
    if pf_match:
        metrics['Profit_Factor'] = float(pf_match.group(1))
    
    # Max Drawdown (USD)
    dd_match = re.search(r'(?<!%)\s*DRAWDOWN\s*\n\s*\$\s*([0-9.]+)', text, re.IGNORECASE)
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
    Generates a detailed report for the best portfolios
    """
    print("🏆 Quantitative report - top portfolios")
    print("="*60)
    
    top_portfolios = df_analyzed.head(top_n)

    # Creates a formatted table
    report_table = top_portfolios[['Rank', 'Portfolio', 'Quant_Score', 'CAGR', 'Sharpe',
                                   'Win_Rate', 'Profit_Factor', 'Max_DD_USD', 'Quality_Tier']].copy()

    # Applies formatting
    report_table['CAGR'] = report_table['CAGR'].round(2).astype(str) + '%'
    report_table['Sharpe'] = report_table['Sharpe'].round(2)
    report_table['Win_Rate'] = report_table['Win_Rate'].round(1).astype(str) + '%'
    report_table['Profit_Factor'] = report_table['Profit_Factor'].round(2)
    report_table['Max_DD_USD'] = '$' + report_table['Max_DD_USD'].round(0).astype(int).astype(str)
    report_table['Quant_Score'] = report_table['Quant_Score'].round(3)

    # Column names
    report_table.columns = ['#', 'Portfolio', 'Score', 'CAGR', 'Sharpe', 'Win%', 'PF', 'Max DD', 'Tier']

    print(f"📊 TOP {top_n} PORTFOLIOS:")
    print(report_table.to_string(index=False))
    
    # Winner analysis 
    winner = top_portfolios.iloc[0]
    print(f"\n🥇 WINNER: {winner['Portfolio']}")
    print(f"   Quantitative Score: {winner['Quant_Score']:.3f}")
    print(f"   CAGR: {winner['CAGR']:.1f}%")
    print(f"   Sharpe: {winner['Sharpe']:.2f}")
    print(f"   Win Rate: {winner['Win_Rate']:.1f}%")
    print(f"   Profit Factor: {winner['Profit_Factor']:.2f}")
    print(f"   Max Drawdown: ${winner['Max_DD_USD']:.0f}")
    print(f"   Institutional quality: {winner['Institutional_Grade']:.0f}/10")

    return top_portfolios

def batch_analyze_portfolios(pdf_folder_path, output_file="portfolio_analysis_results.csv"):
    """
    Analyzes multiple portfolios in a batch from PDF files
    
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
        print("❌ No PDF files found!")
        return None
    
    portfolio_data = {}
    successful_extractions = 0

    print("\n📊 Processing portfolios:")

    # Sets the progress bar to avoid creating new lines
    progress_bar = tqdm(pdf_files,
                        desc="📁 Extracting metrics",
                        unit="file",
                        bar_format="{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        file=sys.stdout,
                        leave=True)
    
    for pdf_file in progress_bar:
        try:
            current_file = pdf_file.name[:25] + "..." if len(pdf_file.name) > 25 else pdf_file.name
            progress_bar.set_description(f"📁 Processing: {current_file}")
            
            metrics = extract_portfolio_metrics_from_pdf(pdf_file)
            if metrics and len(metrics) > 0:
                portfolio_name = pdf_file.stem  # Filename without extension
                portfolio_data[portfolio_name] = metrics
                successful_extractions += 1
            else:
                progress_bar.set_description(f"   ⚠️ It was not possible to extract metrics from {pdf_file.name}")
        except Exception as e:
            progress_bar.set_description(f"   ❌ Error to process {pdf_file.name}: {e}")

    # Finalize progress bar
    progress_bar.set_description("📊 Processing complete")
    progress_bar.close()
    
    if not portfolio_data:
        print("❌ No metrics extracted from PDFs!")
        return None

    print(f"\n✅ {successful_extractions}/{len(pdf_files)} portfolios processed successfully")
    print(f"📊 Success rate: {(successful_extractions/len(pdf_files)*100):.1f}%")
    
    # Quantitative analysis
    df_analyzed = analyze_portfolio_quantitatively(portfolio_data)
    
    # Save results
    df_analyzed.to_csv(output_file, index=False)
    print(f"💾 Results saved in: {output_file}")
    
    # Generate report
    generate_portfolio_report(df_analyzed)
    
    return df_analyzed

# Use example:
# df_results = batch_analyze_portfolios("./portfolios/", "portfolio_analysis.csv")
