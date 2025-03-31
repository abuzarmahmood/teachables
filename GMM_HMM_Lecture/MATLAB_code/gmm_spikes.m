%% Clustering Spike Waveforms using Gaussian Mixture Models
% This script demonstrates:
% 1. Loading and preprocessing neural spike waveform data
% 2. Feature extraction using PCA
% 3. Clustering spike waveforms using Gaussian Mixture Models
% 4. Visualization and evaluation of clustering results
%
% Spike sorting is a critical step in neural data analysis that groups action potentials
% (spikes) based on their shapes to identify individual neurons from extracellular recordings.

clear all
close all
%rng(0)  % Uncomment to set random seed for reproducibility
show_plots = 1;  % Set to 0 to hide intermediate plots

%% Load spike waveform data
%{
Data preprocessing information:
- Spikes have been standardized by dividing values < 0 with the respective
  minimum value of each spike, and dividing values > 0 with the respective
  maximum value of each spike, and converting to '8bit integer' values
- All spikes have been dejittered to have their minima or maxima at the same position
- extrema(:,1) = negative amplitude (minimum value of each spike)
- extrema(:,2) = positive amplitude (maximum value of each spike)

Expected outcomes:
1) Should identify 2 clean neurons 
2) In some cases, one neuron cluster might be split into 2, or all merged into 1 cluster
3) If GMM throws an error, run it again, or reduce the number of clusters
%}

% Load the spike waveform data
load('spike_waveforms.mat');
waveforms = double(spike8bit);  % Convert to double for numerical operations

% Reconstitute waveforms to their original values
% This reverses the normalization to recover the original amplitudes
full_waveforms = waveforms/127;  % Convert to range [-1, 1]
min_mult_waveforms = full_waveforms.*extrema(:,1);  % Scale negative values by min amplitude
max_mult_waveforms = full_waveforms.*extrema(:,2);  % Scale positive values by max amplitude
% Apply the appropriate scaling based on whether values are positive or negative
full_waveforms(find(waveforms < 0)) = min_mult_waveforms(find(waveforms < 0));
full_waveforms(find(waveforms > 0)) = max_mult_waveforms(find(waveforms > 0));

% Visualize the standardized spike waveforms and their extrema values
if show_plots
    figure()
    subplot(1,2,1)
    imagesc(waveforms)
    title('Raw Waveforms')
    xlabel('Time points')
    ylabel('Spike number')
    colorbar()
    subplot(1,2,2)
    imagesc(extrema)
    title('Waveform Extrema')
    xlabel('Min/Max')
    ylabel('Spike number')
    colorbar()
end

%% Visualize a random subset of spike waveforms to see their shapes
waveform_count = 50;
if show_plots
    figure()
    inds = randsample(1:size(extrema,1),waveform_count);
    plot(waveforms(inds,:)')
    title(sprintf('Standardized waveforms, n = %i', waveform_count))
    xlabel('Time points')
    ylabel('Normalized amplitude')
end

%% Perform Principal Component Analysis (PCA) on spikes
% This reduces dimensionality and extracts the most important features of the waveforms
pca_components = 3;  % Number of principal components to keep
[coeff,score,latent] = pca(waveforms);  % Apply PCA to waveforms
score = score(:,1:pca_components);  % Keep only the first 3 principal components

% Visualize the PCA results
if show_plots
    figure()
    % Plot PCA components as a matrix (each row is a spike, each column is a PC)
    subplot(2,1,1)
    imagesc(score)
    title('First 3 Principal Components for waveforms')
    xlabel('Principal Component')
    ylabel('Spike number')
    colorbar()
    
    % Plot PCA components as a 3D scatterplot to visualize clustering
    subplot(2,1,2)
    scatter3(score(:,1),score(:,2),score(:,3))
    title("Scatterplot of first 3 PCs")
    xlabel('PC1')
    ylabel('PC2')
    zlabel('PC3')
end

%% Feature engineering: combine PCA scores with extrema values
% This creates a richer feature set that captures both waveform shape and amplitude
features = [score, extrema];

% Standardize features to have zero mean and unit variance
% This is important for GMM to work properly, as it ensures all features have equal weight
% We do this because there is NO GUARANTEE that the magnitude of the 
% PCs will be comparable to the extrema
standard_features = normalize(features);

% Visualize the effect of standardization on features
if show_plots
    figure
    subplot(2,1,1)
    imagesc(features)
    title('Raw Features')
    xlabel('Feature index')
    ylabel('Spike number')
    colorbar()
    subplot(2,1,2)
    imagesc(standard_features)
    title('Standardized Features')
    xlabel('Feature index')
    ylabel('Spike number')
    colorbar()
end

%% Perform Gaussian Mixture Model fitting with model selection
% Try different numbers of clusters and select the best model using AIC
% AIC (Akaike Information Criterion) balances model fit against complexity
cluster_range = 3:8;  % Try 3 to 8 clusters
AIC = zeros(1,length(cluster_range));  % Store AIC values
GMModels = cell(1,length(cluster_range));  % Store all fitted models
options = statset('MaxIter',500);  % Set maximum iterations to prevent long runs

% Fit GMMs with different numbers of clusters
for k = 1:length(cluster_range)
    fprintf('==== Cluster %i (%i/%i) done ====\n',cluster_range(k),k,length(cluster_range))
    % Create and fit GMM with current number of clusters
    GMModels{k} = fitgmdist(standard_features,cluster_range(k),'Options',options);
    % Calculate AIC for this model
    AIC(k)= GMModels{k}.AIC;
end

% Select the best model (lowest AIC)
[minAIC,numComponents_idx] = min(AIC);
numComponents = cluster_range(numComponents_idx);
fprintf('** AIC determined optimal number of clusters = %i **\n',numComponents)

%% Use the best model to predict cluster assignments for each spike
% Predict assignments of each waveform to particular clusters
cluster_labels = cluster(GMModels{numComponents_idx},standard_features);

% Visualize the clustering results by sorting waveforms by cluster
if show_plots
   figure()
   % Sort spikes by their cluster labels
   [vals, sort_inds] = sort(cluster_labels); 
   sorted_waveforms = waveforms(sort_inds,:);
   
   % Plot sorted waveforms
   subplot(1,2,1)
   imagesc(sorted_waveforms)
   title('Waveforms Sorted by Cluster')
   xlabel('Time points')
   ylabel('Sorted spike number')
   
   % Plot cluster labels
   subplot(1,2,2)
   % Invert values for better visualization (darker = higher cluster number)
   imagesc(max(vals)-vals) % To show colorbar changing in same direction
   title('Cluster Labels')
   xlabel('Column')
   ylabel('Sorted spike number')
   colorbar()
end

%% Analyze the PCA components after sorting by cluster
% Plot PCA components as matrix and as scatterplot
sorted_score = score(sort_inds,:);

% Identify the two largest clusters which likely correspond to the two neurons
% Count spikes in each cluster
count_per_cluster = [];
for i=1:numComponents
   count_per_cluster(i) = sum(vals==i); 
end

% Find the indices of the two largest clusters
[sort_counts, ind_counts] = sort(count_per_cluster);
max_clust_vals = ind_counts(end-1:end);  % Get the two clusters with most spikes

% Create a color array for visualization that highlights the two main neuron clusters
scatter_color = vals;
for i=1:numComponents
    if sum(i==max_clust_vals) < 1 % If it's not one of the two largest clusters
        scatter_color(vals==i) = 0;  % Set color to 0 (background)
    else
        % For the two largest clusters, assign special color codes (1 and 2)
        % We use values higher than the max cluster number first to avoid conflicts
        scatter_color(vals==i) = numComponents+find(i==max_clust_vals);
    end
end

% Relabel the special color codes to be 1 and 2
scatter_color(scatter_color == (numComponents+1)) = 1;  % First main neuron
scatter_color(scatter_color == (numComponents+2)) = 2;  % Second main neuron

% Define a discrete colormap for visualization
discrete_colormap = [1 0 0; 0 1 0; 0 0 1];  % R, G, B

% Visualize the sorted PCA components and highlight the main neuron clusters
if show_plots
    figure()
    % Plot sorted PCA components
    subplot(2,2,1)
    imagesc(sorted_score)
    title('First 3 Sorted PCs for waveforms')
    xlabel('Principal Component')
    ylabel('Sorted spike number')
    
    % Plot cluster labels
    subplot(2,2,2)
    imagesc(max(vals)-vals) % To show colorbar changing in same direction
    title('Cluster labels')
    xlabel('Column')
    ylabel('Sorted spike number')
    colorbar()
    
    % 3D scatter plot with main neurons highlighted
    ax1 = subplot(2,1,2);
    scatter3(score(:,1),score(:,2),score(:,3),[],scatter_color)
    colorbar()
    colormap(ax1,discrete_colormap)
    xlabel('PC1')
    ylabel('PC2')
    zlabel('PC3')
    title('Color 1 and 2 are the two largest clusters (likely neurons)')
end

%% Visualize each cluster's waveforms separately
cols = 3;  % Number of columns in the subplot grid
rows = ceil(numComponents/cols);  % Calculate required rows
max_waveforms = 200;  % Maximum number of waveforms to plot per cluster

figure()
for i = 1:numComponents
    subplot(rows, cols, i);
    % Get indices of spikes in this cluster
    these_idx = find(cluster_labels==i);
    % Limit the number of waveforms to plot
    max_ind = min(length(these_idx),max_waveforms);
    fin_idx = these_idx(1:max_ind);
    % Get the original waveforms for these spikes
    these_waveforms = full_waveforms(fin_idx,:);
    % Plot all waveforms in this cluster
    plot(these_waveforms','Color',[1, 0, 0, 0.2]);
    title(sprintf('Cluster %i (n=%d)',i,length(these_idx)))
end

%% Overlay the main neuron clusters to better visualize differences between them
% Note: The GMM outputs cluster labels in a random order, so we need to manually identify
% which clusters correspond to neurons based on inspection of the previous plots
neuron_clusters = [7,5];  % This should be adjusted based on your specific results
colors = {[1,0,0,0.2],[0,1,0,0.2],[0,0,1,0.2]};  % Red, Green, Blue with transparency
max_waveforms = 200;  % Maximum waveforms to plot per neuron

figure
for i = 1:length(neuron_clusters)
    % Get indices of spikes in this neuron cluster
    these_idx = find(cluster_labels==neuron_clusters(i));
    % Limit the number of waveforms to plot
    max_ind = min(length(these_idx),max_waveforms);
    fin_idx = these_idx(1:max_ind);
    % Get the original waveforms for these spikes
    these_waveforms = full_waveforms(fin_idx,:);
    % Plot all waveforms for this neuron with a distinct color
    plot(these_waveforms','Color',colors{i}, 'DisplayName', sprintf('Neuron %d (Cluster %d)', i, neuron_clusters(i)));
    hold on
end
title('Overlay of Identified Neuron Clusters');
xlabel('Time points');
ylabel('Amplitude');
legend('show');
