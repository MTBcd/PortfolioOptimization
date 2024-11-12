import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tool_box_functions import*

def inverse_volatility_allocation(returns, min_allocation=0.0):
    """
    Perform inverse volatility allocation with minimum allocation constraints.

    Parameters:
    - returns (DataFrame): Historical returns of assets (rows: time periods, columns: assets).
    - min_allocation (float): Minimum allocation for each asset.

    Returns:
    - dict: Portfolio weights and asset volatilities.
    """
    # Calculate asset volatilities
    volatilities = returns.std()
    num_assets = len(volatilities)

    # Calculate initial inverse volatility weights
    inverse_vols = 1 / volatilities
    weights = inverse_vols / np.sum(inverse_vols)

    # Adjust for minimum allocation constraints
    weights = np.maximum(weights, min_allocation)
    weights /= np.sum(weights)  # Re-normalize to ensure weights sum to 1

    return {
        'weights': weights,
        'volatilities': volatilities
    }

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data or load your data
    ticker_symbols = ["^GSPC", "^VIX", 'GM', 'IBM', "AAPL", "MSFT", 'AMZN', 'BABA']
    Portfolio_1 = pd.DataFrame(yf.download(ticker_symbols, "2000-01-01").Close).dropna()
    log_ret = get_log_returns(Portfolio_1).dropna()

    # Perform inverse volatility allocation with minimum allocation constraint
    min_allocation = 0.05  # Minimum 5% allocation to each asset

    results = inverse_volatility_allocation(log_ret, min_allocation)

    # Display results
    print("Optimal Inverse Volatility Portfolio Allocation:")
    for asset, weight, volatility in zip(log_ret.columns, results['weights'], results['volatilities']):
        print(f"{asset}: Weight: {weight:.2%}, Volatility: {volatility:.4f}")

    # Plot the optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(log_ret.columns, results['weights'], color='skyblue')
    plt.title('Optimal Inverse Volatility Portfolio Weights with Minimum Allocation Constraint')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()
