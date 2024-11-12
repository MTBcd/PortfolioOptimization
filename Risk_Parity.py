import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tool_box_functions import*
import yfinance as yf

def risk_parity_allocation(returns, min_allocation=0.0):
    """
    Perform classic risk parity allocation with minimum allocation constraints.

    Parameters:
    - returns (DataFrame): Historical returns of assets (rows: time periods, columns: assets).
    - min_allocation (float): Minimum allocation for each asset.

    Returns:
    - dict: Portfolio weights, risk contributions, and total portfolio volatility.
    """
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)

    # Define the portfolio volatility
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Calculate marginal risk contributions
    def marginal_risk_contributions(weights):
        portfolio_vol = portfolio_volatility(weights)
        mrc = np.dot(cov_matrix, weights) / portfolio_vol
        return mrc

    # Calculate total risk contributions
    def total_risk_contributions(weights):
        portfolio_vol = portfolio_volatility(weights)
        mrc = marginal_risk_contributions(weights)
        return weights * mrc / portfolio_vol

    # Define the objective function (equalize risk contributions)
    def risk_parity_objective(weights):
        trc = total_risk_contributions(weights)
        avg_risk_contribution = np.mean(trc)
        return np.sum((trc - avg_risk_contribution) ** 2)

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
    bounds = [(min_allocation, 1.0) for _ in range(num_assets)]  # Minimum allocation constraint

    # Initial guess for weights
    initial_weights = np.ones(num_assets) / num_assets

    # Optimization
    result = minimize(risk_parity_objective, initial_weights, bounds=bounds, constraints=constraints)

    # Extract optimal weights and risk contributions
    optimal_weights = result.x
    portfolio_vol = portfolio_volatility(optimal_weights)
    trc = total_risk_contributions(optimal_weights)

    return {
        'weights': optimal_weights,
        'risk_contributions': trc,
        'portfolio_volatility': portfolio_vol
    }

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data or load your data
    ticker_symbols = ["^GSPC", "^VIX", 'GM', 'IBM', "AAPL", "MSFT", 'AMZN', 'BABA']
    Portfolio_1 = pd.DataFrame(yf.download(ticker_symbols, "2000-01-01").Close).dropna()
    log_ret = get_log_returns(Portfolio_1).dropna()

    # Perform risk parity allocation with minimum allocation constraint
    min_allocation = 0.05  # Minimum 5% allocation to each asset

    results = risk_parity_allocation(log_ret, min_allocation)

    # Display results
    print("Optimal Risk Parity Portfolio Allocation:")
    for asset, weight, risk_contribution in zip(log_ret.columns, results['weights'], results['risk_contributions']):
        print(f"{asset}: Weight: {weight:.2%}, Risk Contribution: {risk_contribution:.2%}")

    print(f"\nTotal Portfolio Volatility: {results['portfolio_volatility']:.2%}")

    # Plot the optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(log_ret.columns, results['weights'], color='skyblue')
    plt.title('Optimal Risk Parity Portfolio Weights with Minimum Allocation Constraint')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()
