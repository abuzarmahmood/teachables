"""
Python translation of gmm_2D_simulated.m
Clustering with 2D Gaussian Mixture Models
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib import cm

# Generate Data
# Hardcoded to generate 2D variables

# Create distribution parameters
components = 3
mu_s = []  # Mean vectors
covs = []  # Covariance matrices

def gen_cov_mat():
    """Generate a random covariance matrix"""
    cross_cov = np.random.rand()
    temp_cov_mat = np.array([[np.random.rand(), cross_cov], 
                             [cross_cov, np.random.rand()]])
    # Make matrix symmetric by multiplying with transpose
    cov_mat = np.dot(temp_cov_mat, temp_cov_mat.T)
    return cov_mat

for i in range(components):
    mu_s.append(np.random.rand(2) * 10)  # Scaled to clearly separate clusters
    covs.append(gen_cov_mat())

# Generate data for each component
sample_num = 1000  # Samples per component
samples = np.empty((0, 2))
for i in range(components):
    samples = np.vstack([samples, np.random.multivariate_normal(mu_s[i], covs[i], sample_num)])

# Plot samples
plt.figure(figsize=(10, 8))
plt.scatter(samples[:, 0], samples[:, 1])
plt.title('Generated 2D Gaussian Mixture Data')
plt.show()

# Fitting GMM
gm = GaussianMixture(n_components=components)
gm.fit(samples)

# Plot output
# Figure out plot range
minx = np.min(samples[:, 0])
maxx = np.max(samples[:, 0])
miny = np.min(samples[:, 1])
maxy = np.max(samples[:, 1])

# Create meshgrid
x1 = np.arange(minx, maxx, 0.1)
x2 = np.arange(miny, maxy, 0.1)
X1, X2 = np.meshgrid(x1, x2)
X = np.column_stack([X1.ravel(), X2.ravel()])

# Evaluate PDF
y = np.exp(gm.score_samples(X))
y = y.reshape(len(x2), len(x1))

# Plot!
plt.figure(figsize=(10, 8))
plt.scatter(samples[:, 0], samples[:, 1], s=10, marker='.')
plt.contour(x1, x2, y, 100, cmap=cm.viridis)
plt.title('2D GMM with Contours')
plt.colorbar()
plt.show()
