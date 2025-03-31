"""
Python translation of gmm_spikes.m
Clustering spike waveforms using Gaussian Mixture Models
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

# Load data
data = loadmat('spike_waveforms.mat')
waveforms = np.array(data['spike8bit'], dtype=float)
extrema = np.array(data['extrema'], dtype=float)

# Reconstitute waveforms to their original values
# Only useful for comparing final clusters
full_waveforms = waveforms/127
min_mult_waveforms = full_waveforms * extrema[:, 0:1]
max_mult_waveforms = full_waveforms * extrema[:, 1:2]
full_waveforms[waveforms < 0] = min_mult_waveforms[waveforms < 0]
full_waveforms[waveforms > 0] = max_mult_waveforms[waveforms > 0]

img_kwargs = dict(interpolation='nearest', aspect='auto')
# Plot standardized spikes
if show_plots:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(waveforms, **img_kwargs)
    plt.title('Raw Waveforms')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(extrema, **img_kwargs)
    plt.title('Waveform Extrema')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Look at handful of spikes
waveform_count = 50
if show_plots:
    plt.figure()
    inds = random.sample(range(extrema.shape[0]), waveform_count)
    plt.plot(waveforms[inds, :].T)
    plt.title(f'Standardized waveforms, n = {waveform_count}')
    plt.show()

# Perform PCA on spikes to reduce dimensionality and extract "FEATURES"
pca_components = 3
pca = PCA(n_components=pca_components)
score = pca.fit_transform(waveforms)

# Plot PCA components as matrix and as scatterplot
if show_plots:
    plt.figure()
    ax0 = plt.subplot(2, 1, 1)
    im = ax0.imshow(score, **img_kwargs)
    ax0.set_title('First 3 PCs for waveforms')
    plt.colorbar(im)
    ax1 = plt.subplot(2, 1, 2, projection='3d')
    ax1.scatter3D(score[:, 0], score[:, 1], score[:, 2])
    ax1.set_title("Scatterplot of first 3 PCs")
    plt.tight_layout()
    plt.show()

# Combine PCs and Extrema Values as features
features = np.hstack((score, extrema))
# Standardize features
scaler = StandardScaler()
standard_features = scaler.fit_transform(features)

# Plot features vs standard features
if show_plots:
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(features, **img_kwargs)
    plt.title('Raw Features')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(standard_features, **img_kwargs)
    plt.title('Standardized Features')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Perform Gaussian Mixture fits
# Fit a range of cluster numbers and choose best using AIC
cluster_range = range(3, 9)
AIC = np.zeros(len(cluster_range))
GMModels = []

for k, n_clusters in enumerate(cluster_range):
    print(f'==== Cluster {n_clusters} ({k+1}/{len(cluster_range)}) done ====')
    gmm = GaussianMixture(n_components=n_clusters, max_iter=500, n_init=5)
    gmm.fit(standard_features)
    AIC[k] = gmm.aic(standard_features)
    GMModels.append(gmm)

numComponents_idx = np.argmin(AIC)
numComponents = cluster_range[numComponents_idx]
print(f'** AIC determined clusters = {numComponents} **')

# Predict labels as determined by best model
cluster_labels = GMModels[numComponents_idx].predict(standard_features)

# Sort neurons by labels and view
if show_plots:
    plt.figure()
    sort_inds = np.argsort(cluster_labels)
    vals = cluster_labels[sort_inds]
    sorted_waveforms = waveforms[sort_inds, :]
    plt.subplot(1, 2, 1)
    plt.imshow(sorted_waveforms, **img_kwargs)
    plt.title('Sorted waveforms')
    plt.subplot(1, 2, 2)
    plt.imshow(np.max(vals) - vals.reshape(-1, 1), **img_kwargs)  # To show colorbar changing in same direction
    plt.title('Cluster labels')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Plot PCA components again after sorting
sorted_score = score[sort_inds, :]

# Specifically color the 2 largest clusters
# These SHOULD be the 2 neurons
count_per_cluster = np.zeros(numComponents)
for i in range(numComponents):
    count_per_cluster[i] = np.sum(vals == i)

ind_counts = np.argsort(count_per_cluster)
max_clust_vals = ind_counts[-2:]

scatter_color = vals.copy()
for i in range(numComponents):
    if i not in max_clust_vals:  # If it's not in max 2 clusters, set color to 0
        scatter_color[vals == i] = 0
    else:
        # Else, label them 1 and 2
        # Relabelling to higher than the max number of clusters
        # So there is no double counting during clustering
        scatter_color[vals == i] = numComponents + np.where(i == max_clust_vals)[0][0] + 1

# Relabel to be 1 and 2
scatter_color[scatter_color == (numComponents + 1)] = 1
scatter_color[scatter_color == (numComponents + 2)] = 2

discrete_colormap = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

if show_plots:
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(sorted_score, **img_kwargs)
    plt.title('First 3 Sorted PCs for waveforms')
    plt.subplot(2, 2, 2)
    plt.imshow(np.max(vals) - vals.reshape(-1, 1), **img_kwargs)  # To show colorbar changing in same direction
    plt.title('Cluster labels')
    plt.colorbar()
    ax = plt.subplot(2, 1, 2, projection='3d')
    scatter = ax.scatter(score[:, 0], score[:, 1], score[:, 2], c=scatter_color)
    plt.colorbar(scatter)
    plt.title('Color 1 and 2 are largest clusters')
    plt.tight_layout()
    plt.show()

# Plot clusters as determined by AIC
cols = 3
rows = int(np.ceil(numComponents / cols))
max_waveforms = 200

plt.figure(figsize=(12, 8))
for i in range(numComponents):
    plt.subplot(rows, cols, i + 1)
    these_idx = np.where(cluster_labels == i)[0]
    max_ind = min(len(these_idx), max_waveforms)
    fin_idx = these_idx[:max_ind]
    these_waveforms = full_waveforms[fin_idx, :]
    plt.plot(these_waveforms.T, 'r', alpha=0.2)
    plt.title(f'Cluster {i}')
plt.tight_layout()
plt.show()

# Overlay neurons to better visualize difference
# The GMM outputs cluster labels in a random order
# Therefore, the neurons need to be sorted manually
# Enter which clusters had neurons in them
neuron_clusters = [7, 5]  # This should be adjusted based on your results
colors = [(1, 0, 0, 0.2), (0, 1, 0, 0.2), (0, 0, 1, 0.2)]
max_waveforms = 200

plt.figure()
for i, cluster in enumerate(neuron_clusters):
    these_idx = np.where(cluster_labels == cluster)[0]
    max_ind = min(len(these_idx), max_waveforms)
    fin_idx = these_idx[:max_ind]
    these_waveforms = full_waveforms[fin_idx, :]
    plt.plot(these_waveforms.T, color=colors[i])
plt.title('Overlay of neuron clusters')
plt.show()
