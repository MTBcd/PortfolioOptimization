import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tool_box_functions import*

def maximum_diversification_allocation(returns, min_allocation=0.0):
    """
    Perform maximum diversification optimization with minimum allocation constraints.

    Parameters:
    - returns (DataFrame): Historical returns of assets (rows: time periods, columns: assets).
    - min_allocation (float): Minimum allocation for each asset.

    Returns:
    - dict: Portfolio weights, diversification ratio.
    """
    # Calculate asset volatilities and covariance matrix
    volatilities = returns.std()
    cov_matrix = returns.cov()
    num_assets = len(volatilities)

    # Define portfolio volatility
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Define diversification ratio objective (maximize it)
    def diversification_ratio(weights):
        weighted_vols = np.dot(weights, volatilities)
        portfolio_vol = portfolio_volatility(weights)
        return -weighted_vols / portfolio_vol  # Negative for minimization in scipy

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
    bounds = [(min_allocation, 1.0) for _ in range(num_assets)]  # Minimum allocation constraint

    # Initial guess for weights
    initial_weights = np.ones(num_assets) / num_assets

    # Optimization
    result = minimize(diversification_ratio, initial_weights, bounds=bounds, constraints=constraints)

    # Extract optimal weights
    optimal_weights = result.x
    weighted_vols = np.dot(optimal_weights, volatilities)
    portfolio_vol = portfolio_volatility(optimal_weights)
    diversification_ratio_value = weighted_vols / portfolio_vol

    return {
        'weights': optimal_weights,
        'diversification_ratio': diversification_ratio_value
    }

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data or load your data
    ticker_symbols = ["^GSPC", "^VIX", 'GM', 'IBM', "AAPL", "MSFT", 'AMZN', 'BABA']
    Portfolio_1 = pd.DataFrame(yf.download(ticker_symbols, "2000-01-01").Close).dropna()
    log_ret = get_log_returns(Portfolio_1).dropna()

    # Perform maximum diversification optimization with minimum allocation constraint
    min_allocation = 0.05  # Minimum 5% allocation to each asset

    results = maximum_diversification_allocation(log_ret, min_allocation)

    # Display results
    print("Optimal Maximum Diversification Portfolio Allocation:")
    for asset, weight in zip(log_ret.columns, results['weights']):
        print(f"{asset}: {weight:.2%}")

    print(f"\nDiversification Ratio: {results['diversification_ratio']:.4f}")

    # Plot the optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(log_ret.columns, results['weights'], color='skyblue')
    plt.title('Optimal Maximum Diversification Portfolio Weights with Minimum Allocation Constraint')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()
