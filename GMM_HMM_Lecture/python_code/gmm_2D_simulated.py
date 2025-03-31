"""
Python translation of gmm_2D_simulated.m
Clustering with 2D Gaussian Mixture Models

This script demonstrates:
1. Generation of synthetic 2D data from multiple Gaussian distributions
2. Fitting a Gaussian Mixture Model (GMM) to the data
3. Visualization of the fitted model using contour plots

GMMs are probabilistic models that assume data points are generated from a 
mixture of several Gaussian distributions with unknown parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib import cm

# Generate Data
# Hardcoded to generate 2D variables with 3 distinct clusters

# Create distribution parameters
components = 3  # Number of Gaussian components (clusters)
mu_s = []  # Mean vectors for each component
covs = []  # Covariance matrices for each component

def gen_cov_mat():
    """
    Generate a random covariance matrix that is guaranteed to be 
    positive semi-definite (a requirement for covariance matrices).
    
    Returns:
        cov_mat: A 2x2 random covariance matrix
    """
    cross_cov = np.random.rand()  # Random correlation between dimensions
    temp_cov_mat = np.array([[np.random.rand(), cross_cov], 
                             [cross_cov, np.random.rand()]])
    # Make matrix symmetric and positive semi-definite by multiplying with transpose
    # This is a common technique to ensure valid covariance matrices
    cov_mat = np.dot(temp_cov_mat, temp_cov_mat.T)
    return cov_mat

# Generate random means and covariance matrices for each component
for i in range(components):
    mu_s.append(np.random.rand(2) * 10)  # Scaled to clearly separate clusters
    covs.append(gen_cov_mat())

# Generate data points for each component (cluster)
sample_num = 1000  # Samples per component
samples = np.empty((0, 2))
for i in range(components):
    # Generate multivariate normal samples using the specified mean and covariance
    component_samples = np.random.multivariate_normal(mu_s[i], covs[i], sample_num)
    # Stack the new samples with existing ones
    samples = np.vstack([samples, component_samples])

# Plot the generated samples
plt.figure(figsize=(10, 8))
plt.scatter(samples[:, 0], samples[:, 1])
plt.title('Generated 2D Gaussian Mixture Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# Fitting a Gaussian Mixture Model to the data
# The model will try to identify the original components that generated the data
gm = GaussianMixture(n_components=components)  # Create GMM with same number of components as generated data
gm.fit(samples)  # Estimate model parameters (means, covariances, weights)

# Plot the fitted GMM with probability contours
# First, determine the plot range based on data extents
minx = np.min(samples[:, 0])
maxx = np.max(samples[:, 0])
miny = np.min(samples[:, 1])
maxy = np.max(samples[:, 1])

# Create a grid of points to evaluate the GMM probability density
x1 = np.arange(minx, maxx, 0.1)  # Grid points along x-axis
x2 = np.arange(miny, maxy, 0.1)  # Grid points along y-axis
X1, X2 = np.meshgrid(x1, x2)  # Create 2D grid
X = np.column_stack([X1.ravel(), X2.ravel()])  # Reshape to 2D array of points

# Evaluate the probability density function (PDF) at each grid point
y = np.exp(gm.score_samples(X))  # Convert log probability to actual probability
y = y.reshape(len(x2), len(x1))  # Reshape to match grid dimensions

# Create visualization with data points and probability contours
plt.figure(figsize=(10, 8))
# Plot original data points
plt.scatter(samples[:, 0], samples[:, 1], s=10, marker='.', alpha=0.5, label='Data points')
# Plot probability contours of the fitted GMM
contour = plt.contour(x1, x2, y, 100, cmap=cm.viridis)
plt.title('2D GMM with Probability Density Contours')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(label='Probability Density')
plt.legend()
plt.show()
