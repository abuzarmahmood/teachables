%% Clustering Spikes
clear all
close all
%rng(0)
show_plots = 1;
%% Load data
%{
Spikes have been standardized by dividing values < 0 with the respective
minimum value of each spikes, and dividing valies > 0 with the respective
maximum value of each spike, and converting to '8bit integer' values

All spikes have been dejittered to have their minima or maxima at the same position

extrema(:,1) = negative amplitude
extrema(:,2) = positive amplitude

==== NOTES ====
1) Should get 2 clean neurons 
2) In some cases, one neuron cluster might be split into 2, or all merged
into 1 cluster
3) If GMM throws an error, run it again, or reduce the number of clusters
%}

load('spike_waveforms.mat');
waveforms = double(spike8bit);

% Reconstitute waveforms to their original values
% Only useful for comparing final clusters
full_waveforms = waveforms/127;
min_mult_waveforms = full_waveforms.*extrema(:,1);
max_mult_waveforms = full_waveforms.*extrema(:,2);
full_waveforms(find(waveforms < 0)) =  min_mult_waveforms(find(waveforms < 0));
full_waveforms(find(waveforms > 0)) =  max_mult_waveforms(find(waveforms > 0));

% Plot standardized spikes
if show_plots
    figure()
    subplot(1,2,1)
    imagesc(waveforms)
    title('Raw Waveforms')
    colorbar()
    subplot(1,2,2)
    imagesc(extrema)
    title('Waveform Extrema')
    colorbar()
end

%% Look at handful of spikes
waveform_count = 50;
if show_plots
    figure()
    inds = randsample(1:size(extrema,1),waveform_count);
    plot(waveforms(inds,:)')
    title(sprintf('Standardized waveforms, n = %i', waveform_count))
end
%% Perform PCA on spikes to reduce dimensionality and extract "FEATURES"
pca_components = 3;
[coeff,score,latent] = pca(waveforms);
score = score(:,1:pca_components);

% Plot PCA components as matrix and as scatterplot
% See if we can spot a pattern
if show_plots
    figure()
    subplot(2,1,1)
    imagesc(score)
    title('First 3 PCs for waveforms')
    colorbar()
    subplot(2,1,2)
    scatter3(score(:,1),score(:,2),score(:,3))
    title("Scatterplot of first 3 PCs")
end

%% Combine PCs and Extrema Values as features
features = [score, extrema];
% Standardize features, this use useful for ALMOST ALL Machine Learning
% algorithms
% We do this because there is NO GUARANTEE that that the magnitude of the 
% PCs will be comparable to the extrema
% 
standard_features = normalize(features);

% Plot features vs standard features to get an idea of the process
if show_plots
    figure
    subplot(2,1,1)
    imagesc(features)
    title('Raw Features')
    colorbar()
    subplot(2,1,2)
    imagesc(standard_features)
    title('Standardized Features')
    colorbar()
end

%% Perform Gaussian Mixture fits
% Fit a range of cluster numbers and choose best using AIC
cluster_range = 3:8;
AIC = zeros(1,length(cluster_range));
GMModels = cell(1,length(cluster_range));
options = statset('MaxIter',500); % GMMs fit iteratively, set MaxIter to prevents long runs

for k = 1:length(cluster_range)
    fprintf('==== Cluster %i (%i/%i) done ====\n',cluster_range(k),k,length(cluster_range))
    GMModels{k} = fitgmdist(standard_features,cluster_range(k),'Options',options);
    AIC(k)= GMModels{k}.AIC;
end

[minAIC,numComponents_idx] = min(AIC);
numComponents = cluster_range(numComponents_idx);
fprintf('** AIC determined clusters = %i **\n',numComponents)

%% Predict labels as determined by best model
% Predict assignments of each waveform to particular clusters
cluster_labels = cluster(GMModels{numComponents_idx},standard_features);

% Sort neurons by labels and view
if show_plots
   figure()
   [vals, sort_inds] = sort(cluster_labels); 
   sorted_waveforms = waveforms(sort_inds,:);
   subplot(1,2,1)
   imagesc(sorted_waveforms)
   title('Sorted waveforms')
   subplot(1,2,2)
   imagesc(max(vals)-vals) % To show colorbar changing in same direction
   title('Cluster labels')
   colorbar()
end

%% Plot PCA components again after sorting
% Plot PCA components as matrix and as scatterplot
% See if we can spot a pattern
sorted_score = score(sort_inds,:);

% Specifically color the 2 largest clusters
% These SHOULD be the 2 neurons
count_per_cluster = [];
for i=1:numComponents
   count_per_cluster(i) = sum(vals==i); 
end

[sort_counts, ind_counts] = sort(count_per_cluster);
max_clust_vals = ind_counts(end-1:end);

scatter_color = vals;
for i=1:numComponents
    if sum(i==max_clust_vals) < 1 % If it's not in max 2 clusters, set color to 0
        scatter_color(vals==i) = 0;
    else
        % Else, label them 1 and 2
        % Relabelling to higher than the max number of clusters
        % So there is no double counting during clustering
        scatter_color(vals==i) = numComponents+find(i==max_clust_vals);
    end
end

% Relabel to be 1 and 2
scatter_color(scatter_color == (numComponents+1)) = 1;
scatter_color(scatter_color == (numComponents+2)) = 2;

discrete_colormap = [1 0 0; 0 1 0; 0 0 1];

if show_plots
    figure()
    subplot(2,2,1)
    imagesc(sorted_score)
    title('First 3 Sorted PCs for waveforms')
    subplot(2,2,2)
    imagesc(max(vals)-vals) % To show colorbar changing in same direction
    title('Cluster labels')
    colorbar()
    ax1 = subplot(2,1,2);
    scatter3(score(:,1),score(:,2),score(:,3),[],scatter_color)
    colorbar()
    colormap(ax1,discrete_colormap)
    title('Color 1 and 2 are largest clusters')
end

%% Plot clusters as determined by AIC
cols = 3;
rows = ceil(numComponents/cols);
max_waveforms = 200;

figure()
for i = 1:numComponents
    subplot(rows, cols, i);
    these_idx = find(cluster_labels==i);
    max_ind = min(length(these_idx),max_waveforms);
    fin_idx = these_idx(1:max_ind);
    these_waveforms = full_waveforms(fin_idx,:);
    plot(these_waveforms','Color',[1, 0, 0, 0.2]);
    title(sprintf('Cluster %i',i))
end

%% Overlay neurons to better visualize difference
% The GMM outputs cluster labels in a random order
% Therefore, the neurons need to be sorted manually
% Enter which clusters had neurons in them
neuron_clusters = [7,5];
colors = {[1,0,0,0.2],[0,1,0,0.2],[0,0,1,0.2]};
max_waveforms = 200;
figure
for i = 1:length(neuron_clusters)
    these_idx = find(cluster_labels==neuron_clusters(i));
    max_ind = min(length(these_idx),max_waveforms);
    fin_idx = these_idx(1:max_ind);
    these_waveforms = full_waveforms(fin_idx,:);
    plot(these_waveforms','Color',colors{i});
    hold on
end
