%% Clustering with 2D Gaussian Mixture Models
clear all
close all
%rng(0)
%% Generate Data
% Hardcoded to generate 2D variables

% Create distribution parameters
components = 3;
mu_s = {};  % Mean vectors
covs = {};  % Covariance matrices
for i=1:components
   mu_s{end+1} = rand(2,1).*10;  % Scaled to clearly separate clusters
   covs{end+1} = gen_cov_mat(); 
end

% Generate data for each component
sample_num = 1000; % Samples per component
samples = [];
for i=1:components    
    samples = [samples;mvnrnd(mu_s{i},covs{i},sample_num)];
end

% Plot samples
scatter(samples(:,1),samples(:,2));

%% Fitting GMM
gm = fitgmdist(samples,components);

% Plot output
% FIgure out plot range
minx =  min(samples(:,1));
maxx =  max(samples(:,1));
miny =  min(samples(:,2));
maxy =  max(samples(:,2));
% Create meshgrid
x1 = minx:.1:maxx;
x2 = miny:.1:maxy;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
% Evaluate PDF
y = pdf(gm,X);
y = reshape(y,length(x2),length(x1));

% Plot!
figure
scatter(samples(:,1),samples(:,2),10,'.'); % Scatter plot with points of size 10
hold on
contour(x1,x2,y,100)

%% Functions
function cov_mat = gen_cov_mat()
    cross_cov = rand();
    temp_cov_mat = [rand() cross_cov; cross_cov rand()];
    cov_mat = temp_cov_mat*temp_cov_mat'; % Make matrix symmetric by multiplying with transpose 
end
