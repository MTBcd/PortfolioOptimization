import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tool_box_functions import*

def minimum_variance_allocation(returns, min_allocation=0.0):
    """
    Perform minimum variance optimization with minimum allocation constraints.

    Parameters:
    - returns (DataFrame): Historical returns of assets (rows: time periods, columns: assets).
    - min_allocation (float): Minimum allocation for each asset.

    Returns:
    - dict: Portfolio weights, volatility.
    """
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)

    # Define the objective function (minimize portfolio variance)
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
    bounds = [(min_allocation, 1.0) for _ in range(num_assets)]  # Minimum allocation constraint

    # Initial guess for weights
    initial_weights = np.ones(num_assets) / num_assets

    # Optimization
    result = minimize(portfolio_volatility, initial_weights, bounds=bounds, constraints=constraints)

    # Extract optimal weights
    optimal_weights = result.x
    portfolio_vol = portfolio_volatility(optimal_weights)

    return {
        'weights': optimal_weights,
        'volatility': portfolio_vol
    }

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data or load your data
    ticker_symbols = ["^GSPC", "^VIX", 'GM', 'IBM', "AAPL", "MSFT", 'AMZN', 'BABA']
    Portfolio_1 = pd.DataFrame(yf.download(ticker_symbols, "2000-01-01").Close).dropna()
    log_ret = get_log_returns(Portfolio_1).dropna()

    # Perform minimum variance optimization with minimum allocation constraint
    min_allocation = 0.05  # Minimum 5% allocation to each asset

    results = minimum_variance_allocation(log_ret, min_allocation)

    # Display results
    print("Optimal Minimum Variance Portfolio Allocation:")
    for asset, weight in zip(log_ret.columns, results['weights']):
        print(f"{asset}: {weight:.2%}")

    print(f"\nPortfolio Volatility: {results['volatility']:.2%}")

    # Plot the optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(log_ret.columns, results['weights'], color='skyblue')
    plt.title('Optimal Minimum Variance Portfolio Weights with Minimum Allocation Constraint')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()