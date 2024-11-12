
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import cvxopt
from fredapi import Fred
from scipy.optimize import minimize
from scipy.stats import norm

#['TLT', 'IEF', 'LQD', 'HYG', 'MUB', 'BNDX', 'AGG', 'BND', 'TIP', 'VCIT', 'BSV']
ticker_symbols = ['TLT', 'IEF', 'LQD', 'HYG', 'AGG', 'TIP']
maturity = {'TLT': 20, 'IEF': 8.5, 'LQD': 13.14, 'HYG': 4.44, 'AGG': 8.59, 'TIP': 17.5}

Portfolio_1 = pd.DataFrame(yf.download(ticker_symbols, "2013-01-01").Close)
Benchmark_1 = pd.DataFrame(yf.download('FBND', "2013-01-01").Close)

port = Portfolio_1.dropna()
bench = Benchmark_1.dropna()

print(port)
dividend = {}
yields = {}
volumes = {}
for ticker_symbol in ticker_symbols:
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    yield_ = info.get('yield', 'N/A')
    dividend_yield = info.get('trailingAnnualDividendYield', 'N/A')
    volume = info.get('volume', 'N/A')
    print(f"{ticker_symbol} Yield: {yield_}")
    print(f"{ticker_symbol} Dividend Yield: {dividend_yield}")
    print(f"{ticker_symbol} Volume: {volume}")
    dividend[ticker_symbol] = dividend_yield
    yields[ticker_symbol] = yield_
    volumes[ticker_symbol] = volume


def portfolio_returns(assets, weights):
    return weights @ assets.mean()

def portfolio_volatility(assets, weights):
    return weights.T @ assets.cov() @ weights

def portfolio_variance(weights, cov):
    return weights.T @ cov @ weights

def asset_duration(dividend, maturity, yield_rate, price):
    """
    Calculate the Macaulay Duration of a bond.
    Parameters:
    - price: Is a proxy of face value of the bond (principal amount).
    - dividend: Is the proxy of annual coupon rate as a decimal (e.g., 0.05 for 5%).
    - maturity: The number of years to maturity.
    - yield_rate: The annual yield to maturity as a decimal.
    Returns:
    - The Macaulay Duration of the bond in years.
    """
    maturity = int(np.round(maturity))
    cash_flow = np.full(maturity, dividend * price)
    cash_flow[-1] += price
    discounted_cf = [(cf * t) / ((1 + yield_rate) ** t) for t, cf in enumerate(cash_flow, start=1)]
    return np.sum(discounted_cf) / price

def asset_modified_diuration(dividend, maturity, yield_rate, price, componding=1):
    """
    Calculate the Modify Duration of a bond.
    Parameters:
    - price: Is a proxy of face value of the bond (principal amount).
    - dividend: Is the proxy of annual coupon rate as a decimal (e.g., 0.05 for 5%).
    - maturity: The number of years to maturity.
    - yield_rate: The annual yield to maturity as a decimal.
    - componding: componding frequency per year (=1 by default)
    Returns:
    - The Modify Duration of the bond in years.
    """
    maturity = int(np.round(maturity))
    return asset_duration(dividend, maturity, yield_rate, price) / (1 + yield_rate/componding)

def asset_convexity(dividend, maturity, yield_rate, price):
    """
    Calculate the Convexity of a bond.
    Parameters:
    - price: Is a proxy of face value of the bond (principal amount).
    - dividend: Is the proxy of annual coupon rate as a decimal (e.g., 0.05 for 5%).
    - maturity: The number of years to maturity.
    - yield_rate: The annual yield to maturity as a decimal.
    - componding: componding frequency per year (=1 by default)
    Returns:
    - The Convexity of the bond in years.
    """
    maturity = int(np.round(maturity))
    cash_flow = np.full(maturity, dividend * price)
    convexity = np.sum([(t*(t+1)*cf)/((1+yield_rate)**(t+2)) for t, cf in enumerate(cash_flow, start=1)])/price
    return convexity


def portfolio_tracking_error(assets, weights, benchmark, window=252):
    assets_ret = assets.pct_change().dropna()
    bench_ret = benchmark.pct_change().dropna()

    if isinstance(bench_ret, pd.DataFrame):
        bench_ret = bench_ret.squeeze()

    weighted_return = (assets_ret * weights).sum(axis=1)
    ret_aligned = weighted_return.align(bench_ret, join='inner')
    portfolio_returns_aligned, benchmark_returns_aligned = ret_aligned[0], ret_aligned[1]

    return_diff = portfolio_returns_aligned - benchmark_returns_aligned
    tracking_error = np.std(return_diff, ddof=1) * np.sqrt(window)

    return tracking_error


def Parametric_Monte_Carlo_Var(data, weights, window, confidence_level=0.95, simulation: int = 1000):
    data_windowed = data[-window:]

    portfolio_returns = data_windowed.pct_change().dropna()
    weighted_portfolio_returns = (portfolio_returns * weights).sum(axis=1)

    std_ret = np.std(weighted_portfolio_returns, ddof=1)
    mean_ret = np.mean(weighted_portfolio_returns)

    pnl_scenarios = mean_ret + std_ret * np.random.normal(mean_ret, std_ret, simulation)

    window_var = np.quantile(pnl_scenarios, 1 - confidence_level)
    return window_var


print('asset_duration :', asset_duration(dividend['AGG'],maturity['AGG'], yields['AGG'], port['AGG'].iloc[-1]))
print('asset_modified_diuration :', asset_modified_diuration(dividend['AGG'], maturity['AGG'], yields['AGG'], port['AGG'].iloc[-1]))
print('asset_convexity :', asset_convexity(dividend['AGG'], maturity['AGG'], yields['AGG'], port['AGG'].iloc[-1]))

weight = np.ones(len(port.columns)) / len(port.columns)

print('portfolio_tracking_error :', portfolio_tracking_error(port, weight, bench))
print('Parametric_Monte_Carlo_Var :', Parametric_Monte_Carlo_Var(port, weight, window=48))



def Maximum_return(portfolio, dividend, maturity, yields):
    n = len(portfolio.columns)
    init_weights = np.ones(n) / n

    def portfolio_returns(weights):
        return weights @ portfolio.mean()

    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ portfolio.cov() @ weights)

    def asset_duration(dividend, maturity, yield_rate, price):
        maturity = int(np.round(maturity))
        cash_flow = np.full(maturity, dividend * price)
        cash_flow[-1] += price
        discounted_cf = [(cf * t) / ((1 + yield_rate) ** t) for t, cf in enumerate(cash_flow, start=1)]
        return np.sum(discounted_cf) / price

    def portfolio_duration(weights):
        durations = [asset_duration(dividend[ticker],
                                    maturity[ticker],
                                    yields[ticker],
                                    portfolio[ticker].iloc[-1]) for ticker in portfolio.columns]
        return np.dot(weights, durations)

    def portfolio_convexity(weights):
        convexities = [asset_convexity(dividend[ticker], maturity[ticker], yields[ticker], port[ticker].iloc[-1]) for ticker in portfolio.columns]
        return np.dot(weights, convexities)

    def objective(weights):
        return -portfolio_returns(weights)

    constraint = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda weights: 0.20 - portfolio_volatility(weights)},
        {'type': 'ineq', 'fun': lambda weights: 2 - portfolio_duration(weights)},
        {'type': 'ineq', 'fun': lambda weights: portfolio_convexity(weights) - 6},
        {'type': 'ineq', 'fun': lambda weights: weights - 0.1}
    )

    bounds = tuple((0, 1) for _ in range(n))

    optimal_weights = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraint)

    return optimal_weights.x

print(Maximum_return(port.pct_change().dropna(), dividend, maturity, yields))



def optimize_erc(portfolio):
    n = len(portfolio.columns)
    init_weights = np.ones(n) / n
    covariance_matrix = portfolio.cov()

    def portfolio_risk_contribution(weights, covariance_matrix):
        portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
        marginal_risk_contribution = covariance_matrix @ weights
        risk_contribution = np.multiply(weights, marginal_risk_contribution) / portfolio_volatility
        return risk_contribution

    def objective_erc(weights, covariance_matrix):
        risk_contributions = portfolio_risk_contribution(weights, covariance_matrix)
        return np.sum((risk_contributions - np.mean(risk_contributions)) ** 2)

    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)

    optimal_weights = minimize(objective_erc, init_weights, args=(covariance_matrix,), method='SLSQP', bounds=bounds,
                               constraints=constraints)

    return optimal_weights.x


print('ERC optimization :', optimize_erc(port.pct_change().dropna()))



def optimize_te(portfolio, benchmark_returns, target_return):
    n = len(portfolio.columns)
    init_weights = np.ones(n) / n

    def tracking_error(weights, portfolio, benchmark_returns):
        portfolio_returns = (portfolio * weights).sum(axis=1)
        diff_returns = portfolio_returns - benchmark_returns
        return np.std(diff_returns, ddof=1)

    bounds = tuple((0, 1) for _ in range(n))
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda weights: target_return - portfolio_returns(weights)},
    )

    optimal_weights = minimize(tracking_error, init_weights, args=(portfolio, benchmark_returns), method='SLSQP',
                               bounds=bounds, constraints=constraints)

    return optimal_weights.x


print('ERC optimization :', optimize_te(port.pct_change().dropna(), bench.pct_change().dropna(), 0.05))




def optimize_var(portfolio, confidence_level=0.95):
    n = len(portfolio.columns)
    init_weights = np.ones(n) / n
    portfolio_returns = (portfolio * init_weights).sum(axis=1)

    def portfolio_var(weights, portfolio_returns, confidence_level=0.95):
        portfolio_mean = np.mean(portfolio_returns)
        portfolio_std = np.std(portfolio_returns)
        var = norm.ppf(1 - confidence_level, portfolio_mean, portfolio_std)
        return var

    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    optimal_weights = minimize(portfolio_var, init_weights, args=(portfolio_returns, confidence_level), method='SLSQP',
                               bounds=bounds, constraints=constraints)

    return optimal_weights.x




class PortfolioOptimization:
    def __init__(self, price_data):
        self.price_data = price_data


    def get_variance(self):
        return np.var(self.price_data)








