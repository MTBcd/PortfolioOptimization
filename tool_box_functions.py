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
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

def get_returns(data):
        returns = data.pct_change().dropna()
        return returns

def get_log_returns(data):
    returns = np.log(data / data.shift(1))
    return returns

def get_variance(data):
        variances = np.var(data)
        return variances

def get_volatility(data):
        volatilities = np.sqrt(get_variance(data))
        return volatilities

def get_correlation_matrix(data):
      covariance = data.cov()
      diag = np.sqrt(np.diag(covariance))
      diag_matrix = np.outer(diag, diag)
      correlation = covariance / diag_matrix
      np.fill_diagonal(correlation.values, 1)
      return correlation

def get_portfolio_return(data, weights):
        port_returns = data @ weights
        return port_returns

def get_portfolio_volatility(data, weights):
        port_volatility = weights.T @ data.cov() @ weights
        return port_volatility

def get_distance(data):
        distance = np.sqrt(1/2 * (1 - get_correlation_matrix(data)))
        return distance

def get_linkage_matrix(data):
        linkage_mtx = linkage(get_distance(data), 'ward')
        return linkage_mtx
