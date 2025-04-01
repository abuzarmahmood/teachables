%% Labelling simulated data using Hidden Markov Models (HMM)
% This script demonstrates:
% 1. Generation of synthetic data from a Hidden Markov Model
% 2. Estimation of HMM parameters from the generated data
% 3. Comparison of estimated parameters with the true parameters
% 4. State sequence decoding using both true and estimated parameters
%
% Hidden Markov Models are probabilistic models where:
% - The system being modeled follows a Markov process with unobservable (hidden) states
% - Each state has a probability distribution over possible output tokens

clear all
close all

% If false, shows only the main plot
show_plots = 1;

%% Simulate data using MATLAB function

% Create transitions matrix with strong self-transition probabilities
% This matrix defines the probability of transitioning from one state to another
% High values on diagonal mean states tend to persist for multiple time steps
trans = [0.95,0.05;   % 95% chance of staying in state 1, 5% chance of transitioning to state 2
         0.05,0.95];  % 95% chance of staying in state 2, 5% chance of transitioning to state 1
     
% Create very different emissions for easier detection
% Each row represents a state, each column represents the probability of emitting a particular observation
emis = [3/6 2/6 1/6 1/6 1/6 1/6; % State1 emissions - more uniform
        2/10 1/10 1/10 3/10 1/10 1/2]; % State2 emissions - more peaked at the last category

% Normalize emission matrix to make sure probabilities sum to 1 for each state
emis = emis./sum(emis,2);
 
% Generate a sequence of observations and corresponding hidden states
dat_length = 1000;  % Length of sequence to generate
[seq,states] = hmmgenerate(dat_length,trans,emis);  % seq contains observations, states contains true hidden states

% Convert sequence to matrix form for better visualization
% Each column represents a time point, each row represents a possible observation
seq_mat = zeros(size(emis,2),length(seq));
for i = 1:length(seq)
    seq_mat(seq(i),i) = 1;  % Mark the observed category at each time point
end

% Visualize the generated data
if show_plots
    figure
    ax1=subplot(3,1,1);
    plot(seq)
    title('Generated sequence (1D)')
    xlabel('Time')
    ylabel('Category')
    
    ax2=subplot(3,1,2);
    plot(states, 'LineWidth',2)
    title('Sequence of states (ground truth)')
    xlabel('Time')
    ylabel('State')
    
    ax3=subplot(3,1,3);
    imagesc(seq_mat)
    title('Matrix of observations')
    xlabel('Time')
    ylabel('Category')
    
    % Link x-axes for easier comparison
    linkaxes([ax1 ax2 ax3],'x')
end
%% Parameter Estimation Section
% In real applications, we don't know the true parameters and need to estimate them from data
% Here we'll pretend we don't know the true parameters and try to recover them

% Create initial guesses for the model parameters
% MATLAB requires that we provide initial guesses for the EM algorithm
% We'll assume:
% 1. No information about the emissions (random values)
% 2. High self-transition probabilities (diagonal-dominant)
TRANS_GUESS = eye(size(trans)) + rand(size(trans))*0.05;  % Identity matrix + small random values
EMIS_GUESS = rand(size(emis));  % Completely random emission probabilities

% Normalize guesses to ensure they are valid probability distributions
EMIS_GUESS = EMIS_GUESS./sum(EMIS_GUESS,2);  % Each row sums to 1
TRANS_GUESS = TRANS_GUESS./sum(TRANS_GUESS,2);  % Each row sums to 1

% Estimate parameters using the Baum-Welch algorithm (implemented in hmmtrain)
% This is an Expectation-Maximization (EM) algorithm for HMMs
[TRANS_EST, EMIS_EST] = hmmtrain(seq, TRANS_GUESS, EMIS_GUESS);

% The HMM has a label switching problem - the state labels might be flipped
% We need to check which orientation of the estimated emission matrix best matches the original
forward_ = sum(abs(EMIS_EST - emis),'all');  % Error if we keep original order
backward_ = sum(abs(EMIS_EST(2:-1:1,:) - emis),'all');  % Error if we flip the states
[val, ind] = min([forward_,backward_]);  % Find which orientation has smaller error

% Choose the orientation that best matches the original
if ind==1
    plot_emis_est = EMIS_EST;  % Keep original order
else
    plot_emis_est = EMIS_EST(2:-1:1,:);  % Flip the states
end

% Visualize the comparison between true and estimated parameters
if show_plots
   figure
   subplot(2,2,1);
   imagesc(trans,[0,1])
   title("Actual Transition Matrix")
   xlabel('To state')
   ylabel('From state')
   colorbar()
   
   subplot(2,2,2)
   imagesc(TRANS_EST,[0,1])
   title("Estimated Transition Matrix")
   xlabel('To state')
   ylabel('From state')
   colorbar()
   
   subplot(2,2,3)
   imagesc(emis,[0,1])
   title("Actual Emission Matrix")
   xlabel('Category')
   ylabel('State')
   colorbar()
   
   subplot(2,2,4)
   imagesc(plot_emis_est,[0,1])
   title("Estimated Emission Matrix")
   xlabel('Category')
   ylabel('State')
   colorbar()
end

%% Estimate State Sequence (Decoding)
% ** NOTE: Inferred state sequence can be "flipped" compared to the actual
% **       sequence. This is because the inferred state labels can be
% **       different from the original labels (label switching problem).

% We need to give the decoding algorithm our estimate of the model
% parameters (emission and transition matrices)
% We can check how well it does using the actual parameters vs the
% estimated parameters

% Get state probabilities using both true and estimated parameters
pStates_actual = hmmdecode(seq,trans,emis);  % Using true parameters
pStates_est = hmmdecode(seq,TRANS_EST,EMIS_EST);  % Using estimated parameters

% hmmdecode returns PROBABILITIES for being in each state at each timepoint
% We can determine the most likely state by picking the one with higher probability
state_estim_actual = pStates_actual(1,:) > pStates_actual(2,:);  # Convert to binary state sequence
state_estim_est = pStates_est(1,:) > pStates_est(2,:);  # Convert to binary state sequence

% Concatenate all state sequences for visualization
all_states = cat(1,states,state_estim_actual, state_estim_est);

% Visualize the observations and different state sequences
figure
ax1=subplot(2,1,1);
imagesc(seq_mat)
title('Observations')
xlabel('Time')
ylabel('Category')

ax2=subplot(2,1,2);
imagesc(all_states)
yticks([1,2,3])
yticklabels({'Actual',sprintf('Using Actual Parameters'),'Using Estimated Parameters'})
title('State Sequences')
xlabel('Time')

% Link x-axes for easier comparison
linkaxes([ax1 ax2],'x')
