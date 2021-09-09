%% Labelling simulated data using HMM
clear all
close all

% If false, shows only the main plot
show_plots = 1;

%% Simulate data using MATLAB function

% Create transitions matrix with strong self-transition probabilities
trans = [0.95,0.05;
         0.05,0.95];
     
% Create very different emissions for easier detection
emis = [3/6 2/6 1/6 1/6 1/6 1/6; % State1 emissions
        2/10 1/10 1/10 3/10 1/10 1/2]; % State2 emissions

% Normalize emission matrix to make sure everything adds up to 1
 emis = emis./sum(emis,2);
 
% Use MATLAB HMM generate function to create timeseries
dat_length = 1000;
[seq,states] = hmmgenerate(dat_length,trans,emis);

% Convert seq to matrix for visualization
seq_mat = zeros(size(emis,2),length(seq));
for i = 1:length(seq)
    seq_mat(seq(i),i) = 1;
end

if show_plots
    figure
    ax1=subplot(3,1,1);
    plot(seq)
    title('Generated sequence (1D)')
    xlabel('Time')
    ylabel('Category')
    ax2=subplot(3,1,2);
    plot(states, 'LineWidth',2)
    title('Sequence of states')
    ax3=subplot(3,1,3);
    imagesc(seq_mat)
    title('Matrix of observations')
    linkaxes([ax1 ax2 ax3],'x')
end
%% Estimate parameters

% First we have to estimate the emission and transition matrices
% MATLAB requires that we give it an initial guess
% We'll assume : no information about the emissions (random values)
% We'll assume : high self-transition probabilities
TRANS_GUESS = eye(size(trans)) + rand(size(trans))*0.05;
EMIS_GUESS = rand(size(emis));
% Make sure numbers add up to 1
EMIS_GUESS = EMIS_GUESS./sum(EMIS_GUESS,2);
TRANS_GUESS = TRANS_GUESS./sum(TRANS_GUESS,2);


% Estimate parameters
[TRANS_EST, EMIS_EST] = hmmtrain(seq, TRANS_GUESS, EMIS_GUESS);

% Orient emission matrix to match original
forward_ = sum(abs(EMIS_EST - emis),'all');
backward_ = sum(abs(EMIS_EST(2:-1:1,:) - emis),'all');
[val, ind] = min([forward_,backward_]);
if ind==1
    plot_emis_est = EMIS_EST;
else
    plot_emis_est = EMIS_EST(2:-1:1,:);
end

% Generate plots to see how we did
if show_plots
   figure
   subplot(2,2,1);
   imagesc(trans,[0,1])
   title("Actual Transition Matrix")
   subplot(2,2,2)
   imagesc(TRANS_EST,[0,1])
   title("Estimated Transition Matrix")
   subplot(2,2,3)
   imagesc(emis,[0,1])
   title("Actual Emission Matrix")
   subplot(2,2,4)
   imagesc(plot_emis_est,[0,1])
   title("Estimated Emission Matrix")
end

%% Estimate State Sequence
% ** NOTE: Inferred state sequence can be "flipped" compared to this actual
% **        sequence. This is becauese the inferred state labels can be
% **        different from ours

% We need to give the decoding algorithm our estimate of the model
% paramters (emission and transition matrices)
% We can check how well it does using the actual parameters vs the
% estimated parameters
pStates_actual = hmmdecode(seq,trans,emis);
pStates_est = hmmdecode(seq,TRANS_EST,EMIS_EST);

% HMM decode returns PROBABILITIES for being in each state at each
% timepoint
% We can just pick the state with the larger probability at each time
state_estim_actual = pStates_actual(1,:) > pStates_actual(2,:);
state_estim_est = pStates_est(1,:) > pStates_est(2,:);

% Concatenate estimates for visualization
all_states = cat(1,states,state_estim_actual, state_estim_est);

figure
ax1=subplot(2,1,1);
imagesc(seq_mat)
title('Observations')
ax2=subplot(2,1,2);
imagesc(all_states)
yticks([1,2,3])
yticklabels({'Actual',sprintf('Using Actual Emissions'),'Using Estimated Emissions'})
title('State Sequences')
linkaxes([ax1 ax2],'x')
