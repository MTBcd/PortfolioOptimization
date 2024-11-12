import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tool_box_functions import*

def mean_variance_optimization(returns, target_return=None, risk_free_rate=0.0, min_allocation=0.0):
    """
    Perform mean-variance optimization using Markowitz's theory with minimum allocation constraints.

    Parameters:
    - returns (DataFrame): Historical returns of assets (rows: time periods, columns: assets).
    - target_return (float): The desired portfolio return. If None, optimizes for the Sharpe ratio.
    - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.
    - min_allocation (float): Minimum allocation for each asset.

    Returns:
    - dict: Portfolio weights, expected return, volatility, and Sharpe ratio.
    """
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    # Define the objective function (minimize portfolio variance or maximize Sharpe ratio)
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def negative_sharpe_ratio(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = portfolio_volatility(weights)
        return -(portfolio_return - risk_free_rate) / portfolio_vol

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
    bounds = [(min_allocation, 1.0) for _ in range(num_assets)]  # Minimum allocation constraint

    if target_return is not None:
        # Additional constraint for target return
        constraints.append({'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return})
        objective_function = portfolio_volatility
    else:
        objective_function = negative_sharpe_ratio

    # Initial guess for weights
    initial_weights = np.ones(num_assets) / num_assets

    # Optimization
    result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)

    # Extract optimal weights
    optimal_weights = result.x
    portfolio_return = np.dot(optimal_weights, mean_returns)
    portfolio_vol = portfolio_volatility(optimal_weights)
    portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

    return {
        'weights': optimal_weights,
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe_ratio': portfolio_sharpe
    }

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data or load your data
    ticker_symbols = ["^GSPC", "^VIX", 'GM', 'IBM', "AAPL", "MSFT", 'AMZN', 'BABA']
    Portfolio_1 = pd.DataFrame(yf.download(ticker_symbols, "2000-01-01").Close).dropna()
    log_ret = get_log_returns(Portfolio_1).dropna()

    # Perform optimization
    target_return = 0.005  # Target daily return (adjust as needed)
    risk_free_rate = 0.001  # Risk-free rate
    min_alloc = 0.05

    results = mean_variance_optimization(log_ret, target_return, risk_free_rate, min_alloc)

    # Display results
    print("Optimal Portfolio Allocation:")
    for asset, weight in zip(log_ret.columns, results['weights']):
        print(f"{asset}: {weight:.2%}")

    print(f"\nExpected Portfolio Return: {results['expected_return']:.2%}")
    print(f"Portfolio Volatility: {results['volatility']:.2%}")
    print(f"Portfolio Sharpe Ratio: {results['sharpe_ratio']:.2f}")

    # Plot the optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(log_ret.columns, results['weights'], color='skyblue')
    plt.title('Optimal Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()