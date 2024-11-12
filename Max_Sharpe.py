import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf
from tool_box_functions import*

def maximum_sharpe_ratio_allocation(returns, risk_free_rate=0.0, min_allocation=0.0):
    """
    Perform maximum Sharpe ratio optimization with minimum allocation constraints.

    Parameters:
    - returns (DataFrame): Historical returns of assets (rows: time periods, columns: assets).
    - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.
    - min_allocation (float): Minimum allocation for each asset.

    Returns:
    - dict: Portfolio weights, expected return, volatility, and Sharpe ratio.
    """
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    # Define portfolio return and volatility
    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Define the Sharpe ratio (to maximize, so we minimize the negative)
    def negative_sharpe_ratio(weights):
        excess_return = portfolio_return(weights) - risk_free_rate
        vol = portfolio_volatility(weights)
        return -(excess_return / vol)

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
    bounds = [(min_allocation, 1.0) for _ in range(num_assets)]  # Minimum allocation constraint

    # Initial guess for weights
    initial_weights = np.ones(num_assets) / num_assets

    # Optimization
    result = minimize(negative_sharpe_ratio, initial_weights, bounds=bounds, constraints=constraints)

    # Extract optimal weights
    optimal_weights = result.x
    port_return = portfolio_return(optimal_weights)
    port_vol = portfolio_volatility(optimal_weights)
    sharpe_ratio = (port_return - risk_free_rate) / port_vol

    return {
        'weights': optimal_weights,
        'expected_return': port_return,
        'volatility': port_vol,
        'sharpe_ratio': sharpe_ratio
    }

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data or load your data
    ticker_symbols = ["^GSPC", "^VIX", 'GM', 'IBM', "AAPL", "MSFT", 'AMZN', 'BABA']
    Portfolio_1 = pd.DataFrame(yf.download(ticker_symbols, "2000-01-01").Close).dropna()
    log_ret = get_log_returns(Portfolio_1).dropna()

    # Perform maximum Sharpe ratio optimization with minimum allocation constraint
    risk_free_rate = 0.0001  # Risk-free rate
    min_allocation = 0.05  # Minimum 5% allocation to each asset

    results = maximum_sharpe_ratio_allocation(log_ret, risk_free_rate, min_allocation)

    # Display results
    print("Optimal Maximum Sharpe Ratio Portfolio Allocation:")
    for asset, weight in zip(log_ret.columns, results['weights']):
        print(f"{asset}: {weight:.2%}")

    print(f"\nExpected Portfolio Return: {results['expected_return']:.2%}")
    print(f"Portfolio Volatility: {results['volatility']:.2%}")
    print(f"Portfolio Sharpe Ratio: {results['sharpe_ratio']:.2f}")

    # Plot the optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(log_ret.columns, results['weights'], color='skyblue')
    plt.title('Optimal Maximum Sharpe Ratio Portfolio Weights with Minimum Allocation Constraint')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()