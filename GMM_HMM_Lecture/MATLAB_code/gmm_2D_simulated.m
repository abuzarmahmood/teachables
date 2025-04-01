%% Clustering with 2D Gaussian Mixture Models
% This script demonstrates:
% 1. Generation of synthetic 2D data from multiple Gaussian distributions
% 2. Fitting a Gaussian Mixture Model (GMM) to the data
% 3. Visualization of the fitted model using contour plots
%
% GMMs are probabilistic models that assume data points are generated from a 
% mixture of several Gaussian distributions with unknown parameters.

clear all
close all
%rng(0)  % Uncomment to set random seed for reproducibility

%% Generate Data
% Hardcoded to generate 2D variables with 3 distinct clusters

% Create distribution parameters
components = 3;  % Number of Gaussian components (clusters)
mu_s = {};       % Mean vectors for each component
covs = {};       % Covariance matrices for each component

% Generate random means and covariance matrices for each component
for i=1:components
   mu_s{end+1} = rand(2,1).*10;  % Scaled to clearly separate clusters
   covs{end+1} = gen_cov_mat();  % Generate random covariance matrix
end

% Generate data points for each component (cluster)
sample_num = 1000; % Samples per component
samples = [];
for i=1:components    
    % Generate multivariate normal samples using the specified mean and covariance
    component_samples = mvnrnd(mu_s{i}, covs{i}, sample_num);
    % Stack the new samples with existing ones
    samples = [samples; component_samples];
end

% Plot the generated samples
figure;
scatter(samples(:,1), samples(:,2));
title('Generated 2D Gaussian Mixture Data');
xlabel('Dimension 1');
ylabel('Dimension 2');

%% Fitting a Gaussian Mixture Model to the data
% The model will try to identify the original components that generated the data
gm = fitgmdist(samples, components);  % Create GMM with same number of components as generated data

% Plot the fitted GMM with probability contours
% First, determine the plot range based on data extents
minx = min(samples(:,1));
maxx = max(samples(:,1));
miny = min(samples(:,2));
maxy = max(samples(:,2));

% Create a grid of points to evaluate the GMM probability density
x1 = minx:.1:maxx;  % Grid points along x-axis
x2 = miny:.1:maxy;  % Grid points along y-axis
[X1,X2] = meshgrid(x1,x2);  % Create 2D grid
X = [X1(:) X2(:)];  % Reshape to 2D array of points

% Evaluate the probability density function (PDF) at each grid point
y = pdf(gm,X);  % Calculate probability density
y = reshape(y,length(x2),length(x1));  % Reshape to match grid dimensions

% Create visualization with data points and probability contours
figure;
scatter(samples(:,1),samples(:,2),10,'.'); % Scatter plot with points of size 10
hold on;
contour(x1,x2,y,100);  % Plot probability contours
title('2D GMM with Probability Density Contours');
xlabel('Dimension 1');
ylabel('Dimension 2');
colorbar;  % Add colorbar to show density scale

%% Functions
function cov_mat = gen_cov_mat()
    % Generate a random covariance matrix that is guaranteed to be 
    % positive semi-definite (a requirement for covariance matrices).
    %
    % Returns:
    %   cov_mat: A 2x2 random covariance matrix
    
    cross_cov = rand();  % Random correlation between dimensions
    temp_cov_mat = [rand() cross_cov; cross_cov rand()];
    % Make matrix symmetric and positive semi-definite by multiplying with transpose
    % This is a common technique to ensure valid covariance matrices
    cov_mat = temp_cov_mat*temp_cov_mat'; 
end
