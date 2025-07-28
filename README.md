# PortfolioAnalysis

This framework is to analyze and rank multiple trading portfolios in a way to help choose the best portfolio.

---

## How to use

#### :one: Put PDF files in a folder
- Choose a folder to put PDF files in (ex: './portfolios/')
- Be sure that PDFs are in the SQX format 


#### :two: Execution:
- Execute .ipynb file

#### :three: Customization (optional):
   ```python
   # Personalized weights
   custom_weights = {
       'risk_efficiency': 0.30,      # More weight in the efficiency
       'consistency': 0.25,          # More weight in the consistency  
       'stability': 0.20,
       'institutional_grade': 0.15,
       'sharpe_ratio': 0.05,
       'max_dd_protection': 0.05
   }
   
   df_custom = analyze_portfolio_quantitatively(portfolio_data, custom_weights)
   ```


#### :four: Results:
- CSV file containing all results
- Automatic ranking
- Top performers detailed report
- Advanced metrics for each portfolio

---

## :zap: Expected performance:

- Processing: 5 to 10 minutes for 600 PDFs
- Quantitative analysis: less than 1 minute
- Report generation: instantly
