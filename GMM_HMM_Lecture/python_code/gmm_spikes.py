"""
Python translation of gmm_spikes.m
Clustering spike waveforms using Gaussian Mixture Models

This script demonstrates:
1. Loading and preprocessing neural spike waveform data
2. Feature extraction using PCA
3. Clustering spike waveforms using Gaussian Mixture Models
4. Visualization and evaluation of clustering results

Spike sorting is a critical step in neural data analysis that groups action potentials
(spikes) based on their shapes to identify individual neurons from extracellular recordings.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.io import loadmat
import random

# If false, shows only the main plot
show_plots = True

# Load spike waveform data from MATLAB file
data = loadmat('spike_waveforms.mat')
waveforms = np.array(data['spike8bit'], dtype=float)  # Standardized waveforms
extrema = np.array(data['extrema'], dtype=float)      # Min/max values for each waveform

# Reconstitute waveforms to their original values
# The waveforms were previously normalized to 8-bit integers (-127 to 127)
# This process reverses that normalization to recover the original amplitudes
full_waveforms = waveforms/127  # Convert to range [-1, 1]
min_mult_waveforms = full_waveforms * extrema[:, 0:1]  # Scale negative values by min amplitude
max_mult_waveforms = full_waveforms * extrema[:, 1:2]  # Scale positive values by max amplitude
# Apply the appropriate scaling based on whether values are positive or negative
full_waveforms[waveforms < 0] = min_mult_waveforms[waveforms < 0]
full_waveforms[waveforms > 0] = max_mult_waveforms[waveforms > 0]

# Set common image display parameters
img_kwargs = dict(interpolation='nearest', aspect='auto')

# Visualize the standardized spike waveforms and their extrema values
if show_plots:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(waveforms, **img_kwargs)
    plt.title('Raw Waveforms')
    plt.xlabel('Time points')
    plt.ylabel('Spike number')
    plt.colorbar(label='Normalized amplitude')
    
    plt.subplot(1, 2, 2)
    plt.imshow(extrema, **img_kwargs)
    plt.title('Waveform Extrema')
    plt.xlabel('Min/Max')
    plt.ylabel('Spike number')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()

# Visualize a random subset of spike waveforms to see their shapes
waveform_count = 50
if show_plots:
    plt.figure(figsize=(10, 6))
    inds = random.sample(range(extrema.shape[0]), waveform_count)
    plt.plot(waveforms[inds, :].T)
    plt.title(f'Standardized waveforms, n = {waveform_count}')
    plt.xlabel('Time points')
    plt.ylabel('Normalized amplitude')
    plt.show()

# Perform Principal Component Analysis (PCA) on spikes
# This reduces dimensionality and extracts the most important features of the waveforms
pca_components = 3  # Number of principal components to keep
pca = PCA(n_components=pca_components)
score = pca.fit_transform(waveforms)  # Transform waveforms to PC space

# Visualize the PCA results
if show_plots:
    plt.figure(figsize=(10, 8))
    # Plot PCA components as a matrix (each row is a spike, each column is a PC)
    ax0 = plt.subplot(2, 1, 1)
    im = ax0.imshow(score, **img_kwargs)
    ax0.set_title('First 3 Principal Components for waveforms')
    ax0.set_xlabel('Principal Component')
    ax0.set_ylabel('Spike number')
    plt.colorbar(im, label='PC value')
    
    # Plot PCA components as a 3D scatterplot to visualize clustering
    ax1 = plt.subplot(2, 1, 2, projection='3d')
    ax1.scatter3D(score[:, 0], score[:, 1], score[:, 2], alpha=0.5)
    ax1.set_title("Scatterplot of first 3 PCs")
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    plt.tight_layout()
    plt.show()

# Feature engineering: combine PCA scores with extrema values
# This creates a richer feature set that captures both waveform shape and amplitude
features = np.hstack((score, extrema))

# Standardize features to have zero mean and unit variance
# This is important for GMM to work properly, as it ensures all features have equal weight
scaler = StandardScaler()
standard_features = scaler.fit_transform(features)

# Visualize the effect of standardization on features
if show_plots:
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(features, **img_kwargs)
    plt.title('Raw Features')
    plt.xlabel('Feature index')
    plt.ylabel('Spike number')
    plt.colorbar(label='Feature value')
    
    plt.subplot(2, 1, 2)
    plt.imshow(standard_features, **img_kwargs)
    plt.title('Standardized Features')
    plt.xlabel('Feature index')
    plt.ylabel('Spike number')
    plt.colorbar(label='Standardized value')
    plt.tight_layout()
    plt.show()

# Perform Gaussian Mixture Model fitting with model selection
# Try different numbers of clusters and select the best model using AIC
# AIC (Akaike Information Criterion) balances model fit against complexity
cluster_range = range(3, 9)  # Try 3 to 8 clusters
AIC = np.zeros(len(cluster_range))  # Store AIC values
GMModels = []  # Store all fitted models

# Fit GMMs with different numbers of clusters
for k, n_clusters in enumerate(cluster_range):
    print(f'==== Cluster {n_clusters} ({k+1}/{len(cluster_range)}) done ====')
    # Create and fit GMM with current number of clusters
    gmm = GaussianMixture(
        n_components=n_clusters,  # Number of clusters
        max_iter=500,             # Maximum EM iterations
        n_init=5                  # Number of initializations to try
    )
    gmm.fit(standard_features)
    
    # Calculate AIC for this model
    AIC[k] = gmm.aic(standard_features)
    GMModels.append(gmm)

# Select the best model (lowest AIC)
numComponents_idx = np.argmin(AIC)
numComponents = cluster_range[numComponents_idx]
print(f'** AIC determined optimal number of clusters = {numComponents} **')

# Use the best model to predict cluster assignments for each spike
cluster_labels = GMModels[numComponents_idx].predict(standard_features)

# Visualize the clustering results by sorting waveforms by cluster
if show_plots:
    plt.figure(figsize=(12, 6))
    # Sort spikes by their cluster labels
    sort_inds = np.argsort(cluster_labels)
    vals = cluster_labels[sort_inds]
    sorted_waveforms = waveforms[sort_inds, :]
    
    # Plot sorted waveforms
    plt.subplot(1, 2, 1)
    plt.imshow(sorted_waveforms, **img_kwargs)
    plt.title('Waveforms Sorted by Cluster')
    plt.xlabel('Time points')
    plt.ylabel('Sorted spike number')
    
    # Plot cluster labels
    plt.subplot(1, 2, 2)
    # Invert values for better visualization (darker = higher cluster number)
    plt.imshow(np.max(vals) - vals.reshape(-1, 1), **img_kwargs)
    plt.title('Cluster Labels')
    plt.xlabel('Column')
    plt.ylabel('Sorted spike number')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.show()

# Analyze the PCA components after sorting by cluster
sorted_score = score[sort_inds, :]

# Identify the two largest clusters which likely correspond to the two neurons
# Count spikes in each cluster
count_per_cluster = np.zeros(numComponents)
for i in range(numComponents):
    count_per_cluster[i] = np.sum(vals == i)

# Find the indices of the two largest clusters
ind_counts = np.argsort(count_per_cluster)
max_clust_vals = ind_counts[-2:]  # Get the two clusters with most spikes

# Create a color array for visualization that highlights the two main neuron clusters
scatter_color = vals.copy()
for i in range(numComponents):
    if i not in max_clust_vals:  # If it's not one of the two largest clusters
        scatter_color[vals == i] = 0  # Set color to 0 (background)
    else:
        # For the two largest clusters, assign special color codes (1 and 2)
        # We use values higher than the max cluster number first to avoid conflicts
        scatter_color[vals == i] = numComponents + np.where(i == max_clust_vals)[0][0] + 1

# Relabel the special color codes to be 1 and 2
scatter_color[scatter_color == (numComponents + 1)] = 1  # First main neuron
scatter_color[scatter_color == (numComponents + 2)] = 2  # Second main neuron

# Define a discrete colormap for visualization
discrete_colormap = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # R, G, B

# Visualize the sorted PCA components and highlight the main neuron clusters
if show_plots:
    plt.figure(figsize=(12, 10))
    
    # Plot sorted PCA components
    plt.subplot(2, 2, 1)
    plt.imshow(sorted_score, **img_kwargs)
    plt.title('First 3 Sorted PCs for waveforms')
    plt.xlabel('Principal Component')
    plt.ylabel('Sorted spike number')
    
    # Plot cluster labels
    plt.subplot(2, 2, 2)
    plt.imshow(np.max(vals) - vals.reshape(-1, 1), **img_kwargs)
    plt.title('Cluster labels')
    plt.xlabel('Column')
    plt.ylabel('Sorted spike number')
    plt.colorbar(label='Cluster')
    
    # 3D scatter plot with main neurons highlighted
    ax = plt.subplot(2, 1, 2, projection='3d')
    scatter = ax.scatter(score[:, 0], score[:, 1], score[:, 2], c=scatter_color)
    plt.colorbar(scatter, label='Cluster')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('Color 1 and 2 are the two largest clusters (likely neurons)')
    plt.tight_layout()
    plt.show()

# Visualize each cluster's waveforms separately
cols = 3  # Number of columns in the subplot grid
rows = int(np.ceil(numComponents / cols))  # Calculate required rows
max_waveforms = 200  # Maximum number of waveforms to plot per cluster

plt.figure(figsize=(12, 8))
for i in range(numComponents):
    plt.subplot(rows, cols, i + 1)
    # Get indices of spikes in this cluster
    these_idx = np.where(cluster_labels == i)[0]
    # Limit the number of waveforms to plot
    max_ind = min(len(these_idx), max_waveforms)
    fin_idx = these_idx[:max_ind]
    # Get the original waveforms for these spikes
    these_waveforms = full_waveforms[fin_idx, :]
    # Plot all waveforms in this cluster
    plt.plot(these_waveforms.T, 'r', alpha=0.2)
    plt.title(f'Cluster {i} (n={len(these_idx)})')
plt.tight_layout()
plt.show()

# Overlay the main neuron clusters to better visualize differences between them
# Note: The GMM outputs cluster labels in a random order, so we need to manually identify
# which clusters correspond to neurons based on inspection of the previous plots
neuron_clusters = [7, 5]  # This should be adjusted based on your specific results
colors = [(1, 0, 0, 0.2), (0, 1, 0, 0.2), (0, 0, 1, 0.2)]  # Red, Green, Blue with transparency
max_waveforms = 200  # Maximum waveforms to plot per neuron

plt.figure(figsize=(10, 6))
for i, cluster in enumerate(neuron_clusters):
    # Get indices of spikes in this neuron cluster
    these_idx = np.where(cluster_labels == cluster)[0]
    # Limit the number of waveforms to plot
    max_ind = min(len(these_idx), max_waveforms)
    fin_idx = these_idx[:max_ind]
    # Get the original waveforms for these spikes
    these_waveforms = full_waveforms[fin_idx, :]
    # Plot all waveforms for this neuron with a distinct color
    plt.plot(these_waveforms.T, color=colors[i], label=f'Neuron {i+1} (Cluster {cluster})')

plt.title('Overlay of Identified Neuron Clusters')
plt.xlabel('Time points')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
