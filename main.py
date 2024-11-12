import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import cvxopt
from fredapi import Fred
warnings.simplefilter("ignore")

fred = Fred(api_key='48947fcc5986451303203ac6738101f9')

#['TLT', 'IEF', 'LQD', 'HYG', 'MUB', 'BNDX', 'AGG', 'BND', 'TIP', 'VCIT', 'BSV']
#['TLT', 'IEF', 'LQD', 'HYG', 'AGG', 'TIP']

symbols = ['DGS10', 'DGS2', 'DGS30', 'MORTGAGE30US', 'MORTGAGE15US', 'AAA', 'BAA', 'DFF', 'DFII5']

data_dict = {}

for symbol in symbols:
    try:
        data = fred.get_series(symbol)
        data_dict[symbol] = data
        print(f"Data for {symbol} fetched successfully.")
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

print(data_dict)

Portfolio_1 = pd.DataFrame(yf.download(['TLT', 'IEF', 'LQD', 'HYG', 'MUB', 'BNDX'], "1990-01-01").Close)
weights = [1/len(Portfolio_1.columns)]*len(Portfolio_1)

print(Portfolio_1)

DF = pd.read_excel(r'C:\Users\LENOVO\Desktop\GITHUB\PortfolioOptimization\fixed_income_portfolio_data.xls')




def portfolio_variance(weights, cov):
    return weights.T @ cov @ weights

print(portfolio_variance(weights, Portfolio_1.cov()))

def objective(weights, mean_ret):
    return weights.T * mean_ret



class PortfolioOptimization:
    def __init__(self, price_data):
        self.price_data = price_data


    def get_variance(self):
        return np.var(self.price_data)
















