import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
from tool_box_functions import*
import yfinance as yf

def minimum_var_allocation(returns, confidence_level=0.95, min_allocation=0.0):
    """
    Perform minimum Value at Risk (VaR) optimization with minimum allocation constraints.

    Parameters:
    - returns (DataFrame): Historical returns of assets (rows: time periods, columns: assets).
    - confidence_level (float): Confidence level for VaR calculation (e.g., 0.95 for 95% VaR).
    - min_allocation (float): Minimum allocation for each asset.

    Returns:
    - dict: Portfolio weights, portfolio VaR.
    """
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    # Define portfolio VaR function
    def portfolio_var(weights):
        portfolio_mean = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Calculate VaR using the normal distribution
        z_score = norm.ppf(1 - confidence_level)
        return -portfolio_mean + z_score * portfolio_vol  # Negative sign since we minimize

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
    bounds = [(min_allocation, 1.0) for _ in range(num_assets)]  # Minimum allocation constraint

    # Initial guess for weights
    initial_weights = np.ones(num_assets) / num_assets

    # Optimization
    result = minimize(portfolio_var, initial_weights, bounds=bounds, constraints=constraints)

    # Extract optimal weights and calculate portfolio VaR
    optimal_weights = result.x
    portfolio_mean = np.dot(optimal_weights, mean_returns)
    portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    z_score = norm.ppf(1 - confidence_level)
    portfolio_var = -portfolio_mean + z_score * portfolio_vol

    return {
        'weights': optimal_weights,
        'portfolio_var': portfolio_var
    }

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data or load your data
    ticker_symbols = ["^GSPC", "^VIX", 'GM', 'IBM', "AAPL", "MSFT", 'AMZN', 'BABA']
    Portfolio_1 = pd.DataFrame(yf.download(ticker_symbols, "2000-01-01").Close).dropna()
    log_ret = get_log_returns(Portfolio_1).dropna()

    # Perform minimum VaR optimization with minimum allocation constraint
    confidence_level = 0.95  # 95% confidence level
    min_allocation = 0.05  # Minimum 5% allocation to each asset

    results = minimum_var_allocation(log_ret, confidence_level, min_allocation)

    # Display results
    print("Optimal Minimum Value at Risk Portfolio Allocation:")
    for asset, weight in zip(log_ret.columns, results['weights']):
        print(f"{asset}: {weight:.2%}")

    print(f"\nPortfolio VaR (95% confidence level): {results['portfolio_var']:.4f}")

    # Plot the optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(log_ret.columns, results['weights'], color='skyblue')
    plt.title('Optimal Minimum Value at Risk Portfolio Weights with Minimum Allocation Constraint')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()
