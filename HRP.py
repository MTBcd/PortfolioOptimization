import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import yfinance as yf
from tool_box_functions import*

def hrp_allocation(returns, method='single', min_allocation=0.0):
    """
    Perform Hierarchical Risk Parity (HRP) allocation.

    Parameters:
    - returns (DataFrame): Historical returns of assets (rows: time periods, columns: assets).
    - method (str): Linkage method for hierarchical clustering (e.g., 'single', 'complete', 'average', 'ward').
    - min_allocation (float): Minimum allocation for each asset.

    Returns:
    - dict: Portfolio weights and linkage matrix for hierarchical clustering.
    """
    # Calculate covariance matrix and correlation matrix
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()

    # Perform hierarchical clustering
    distances = pdist(corr_matrix, metric='euclidean')
    linkage_matrix = linkage(distances, method=method)
    
    def recursive_bisection(cov, cluster_indices):
        if len(cluster_indices) == 1:
            return {cluster_indices[0]: 1}

        # Split cluster into two sub-clusters
        sub_clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
        cluster_1 = [cluster_indices[i] for i in range(len(cluster_indices)) if sub_clusters[i] == 1]
        cluster_2 = [cluster_indices[i] for i in range(len(cluster_indices)) if sub_clusters[i] == 2]

        # Debugging print to inspect the clusters
        print(f"Cluster 1: {cluster_1}, Cluster 2: {cluster_2}")

        # Fallback: Handle empty clusters by splitting manually
        if not cluster_1 or not cluster_2:
            print(f"Manual split fallback triggered for cluster: {cluster_indices}")
            mid_point = len(cluster_indices) // 2
            cluster_1 = cluster_indices[:mid_point]
            cluster_2 = cluster_indices[mid_point:]

        # Compute variance for each sub-cluster
        cov_1 = cov.loc[cluster_1, cluster_1]
        cov_2 = cov.loc[cluster_2, cluster_2]
        var_1 = get_cluster_variance(cov_1)
        var_2 = get_cluster_variance(cov_2)

        # Allocate risk proportionally to inverse variance
        alloc_1 = var_2 / (var_1 + var_2)
        alloc_2 = var_1 / (var_1 + var_2)

        print(f"Allocations: Alloc 1={alloc_1:.4f}, Alloc 2={alloc_2:.4f}")

        # Recursively allocate within each sub-cluster
        allocation = {}
        for cluster, alloc in zip([cluster_1, cluster_2], [alloc_1, alloc_2]):
            sub_alloc = recursive_bisection(cov, cluster)
            for key in sub_alloc:
                allocation[key] = sub_alloc[key] * alloc

        return allocation
    

    def get_cluster_variance(cov):
        weights = np.ones(len(cov)) / len(cov)
        return np.dot(weights, np.dot(cov, weights))

    # Convert indices to names
    asset_names = returns.columns.tolist()
    cluster_indices = asset_names  # Use asset labels instead of integer indices
    raw_allocation = recursive_bisection(cov_matrix, cluster_indices)

    # Extract weights and enforce minimum allocation constraints
    weights = pd.Series(raw_allocation).reindex(asset_names).fillna(0)
    weights = np.maximum(weights, min_allocation)
    weights /= weights.sum()

    return {
        'weights': weights,
        'linkage_matrix': linkage_matrix
    }

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data or load your data
    ticker_symbols = ["^GSPC", "^VIX", 'GM', 'IBM', "AAPL", "MSFT", 'AMZN', 'BABA']
    Portfolio_1 = pd.DataFrame(yf.download(ticker_symbols, "2000-01-01").Close).dropna()
    log_ret = get_log_returns(Portfolio_1).dropna()

    # Perform HRP allocation
    method = 'ward'  # Linkage method for clustering
    min_allocation = 0.05  # Minimum 5% allocation to each asset

    results = hrp_allocation(log_ret, method, min_allocation)

    # Display results
    print("Optimal HRP Portfolio Allocation:")
    for asset, weight in zip(log_ret.columns, results['weights']):
        print(f"{asset}: Weight: {weight:.2%}")

    # Plot the dendrogram for hierarchical clustering
    plt.figure(figsize=(10, 6))
    dendrogram(results['linkage_matrix'], labels=log_ret.columns, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Assets')
    plt.ylabel('Distance')
    plt.show()

    # Plot the optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(log_ret.columns, results['weights'], color='skyblue')
    plt.title('Optimal HRP Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()





