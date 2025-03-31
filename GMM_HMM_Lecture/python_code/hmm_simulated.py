"""
Python translation of hmm_simulated.m
Labelling simulated data using Hidden Markov Models (HMM)

This script demonstrates:
1. Generation of synthetic data from a Hidden Markov Model
2. Estimation of HMM parameters from the generated data
3. Comparison of estimated parameters with the true parameters
4. State sequence decoding using both true and estimated parameters

Hidden Markov Models are probabilistic models where:
- The system being modeled follows a Markov process with unobservable (hidden) states
- Each state has a probability distribution over possible output tokens
"""

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import random
from scipy.special import logsumexp

# If false, shows only the main plot
show_plots = True

# Number of repeats for model fitting to find the best model
n_repeats = 10

# Create transitions matrix with strong self-transition probabilities
# This matrix defines the probability of transitioning from one state to another
# High values on diagonal mean states tend to persist for multiple time steps
trans = np.array([[0.975, 0.025],  # 97.5% chance of staying in state 0, 2.5% chance of transitioning to state 1
                  [0.025, 0.975]])  # 97.5% chance of staying in state 1, 2.5% chance of transitioning to state 0

# Create very different emissions for easier detection
# Each row represents a state, each column represents the probability of emitting a particular observation
emis = np.array([[3/6, 2/6, 1/6, 1/6, 1/6, 1/6],  # State 0 emissions - more uniform
                 [2/10, 1/10, 1/10, 3/10, 1/10, 1/2]])  # State 1 emissions - more peaked at the last category

# Normalize emission matrix to make sure probabilities sum to 1 for each state
emis = emis / np.sum(emis, axis=1)[:, np.newaxis]

# Visualize the transition and emission matrices
if show_plots:
    img_kwargs = {'interpolation': 'nearest', 'aspect': 'auto'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot transition matrix
    ax1.imshow(trans, **img_kwargs)
    ax1.set_title('Transition Matrix')
    ax1.set_xlabel('To state')
    ax1.set_ylabel('From state')
    
    # Plot emission matrix (transposed for better visualization)
    im = ax2.imshow(emis.T, **img_kwargs, vmin=0, vmax=1)
    cax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cax, label='Probability')
    ax2.set_title('Emission Matrix')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Observation Category')
    plt.show()

# Initialize HMM model with known parameters for data generation
model = hmm.MultinomialHMM(n_components=2, n_iter=100)
model.startprob_ = np.array([0.5, 0.5])  # Equal probability of starting in either state
model.transmat_ = trans  # Set transition matrix
model.emissionprob_ = emis  # Set emission matrix
# Required parameter for MultinomialHMM
model.n_trials = 1

# Generate a sequence of observations and corresponding hidden states
dat_length = 1000  # Length of sequence to generate
X, states = model.sample(dat_length)  # X contains observations, states contains true hidden states

if show_plots:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True) 
    ax1.imshow(X.T, **img_kwargs)
    ax1.set_title('Generated sequence (1D)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Category')
    
    ax2.plot(states, linewidth=2)
    ax2.set_title('Sequence of states')
    
    plt.tight_layout()
    plt.show()

# Parameter Estimation Section
# In real applications, we don't know the true parameters and need to estimate them from data
# Here we'll pretend we don't know the true parameters and try to recover them

# Create initial guesses for the model parameters
# For transition matrix: assume high self-transition probabilities (diagonal-dominant)
TRANS_GUESS = np.eye(trans.shape[0]) + np.random.rand(*trans.shape) * 0.05

# For emission matrix: we can try different initialization strategies
USE_RANDOM_EMIS_GUESS = False
if USE_RANDOM_EMIS_GUESS:
    # Strategy 1: Completely random guess
    EMIS_GUESS = np.random.rand(*emis.shape)
    print("Using random guess for emissions")
else:
    # Strategy 2: Use the mean of observed data as a starting point (more informed)
    # This creates identical rows (same emission probabilities for both states) plus some noise
    EMIS_GUESS = np.tile(X.mean(axis=0), (2, 1)) + np.random.rand(*emis.shape) * 0.05 
    print("Using mean of data as guess for emissions")

# Normalize guesses to ensure they are valid probability distributions
EMIS_GUESS = EMIS_GUESS / np.sum(EMIS_GUESS, axis=1)[:, np.newaxis]  # Each row sums to 1
TRANS_GUESS = TRANS_GUESS / np.sum(TRANS_GUESS, axis=1)[:, np.newaxis]  # Each row sums to 1

# Function to calculate Bayesian Information Criterion (BIC) for model selection
def calculate_bic(model, X):
    """
    Calculate BIC for a fitted HMM model.
    
    BIC = -2 * log-likelihood + n_params * log(n_samples)
    
    Lower BIC values indicate better models (good fit with fewer parameters).
    
    Args:
        model: Fitted HMM model
        X: Observation data
        
    Returns:
        bic: BIC value for the model
    """
    # Get log likelihood of the data given the model
    log_likelihood = model.score(X)
    
    # Calculate number of free parameters in the model
    n_components = model.n_components  # Number of hidden states
    n_features = model.emissionprob_.shape[1]  # Number of possible observations
    
    # Parameters: transition probs + emission probs + initial probs - constraints
    # Constraints: each row of transition and emission matrices sums to 1
    n_params = (n_components * (n_components - 1)) + (n_components * (n_features - 1)) + (n_components - 1)
    
    # Calculate BIC: -2 * log_likelihood + n_params * log(n_samples)
    n_samples = X.shape[0]
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    
    return bic

# Function to perform multiple model fits with different initializations
def perform_multiple_fits(X, n_repeats=10):
    """
    Perform multiple repeats of model fitting with different random initializations
    and return all models with their BIC values.
    
    This helps avoid local optima in the parameter estimation process.
    
    Args:
        X: Observation data
        n_repeats: Number of times to fit the model with different initializations
        
    Returns:
        models_with_bic: List of (model, bic_value) tuples
    """
    models_with_bic = []
    
    for i in range(n_repeats):
        # Create a new model with random initialization
        model = hmm.MultinomialHMM(n_components=2, n_iter=1000)
        model.startprob_ = np.array([0.5, 0.5])  # Equal initial probabilities
        
        # Random initialization for transition matrix with high self-transition
        trans_guess = np.eye(2) + np.random.rand(2, 2) * 0.05
        trans_guess = trans_guess / np.sum(trans_guess, axis=1)[:, np.newaxis]
        model.transmat_ = trans_guess
        
        # Random initialization for emission matrix
        if USE_RANDOM_EMIS_GUESS:
            emis_guess = np.random.rand(2, 6)  # Completely random
        else:
            # Use data mean as starting point with some noise
            emis_guess = np.tile(X.mean(axis=0), (2, 1)) + np.random.rand(2, 6) * 0.05
        
        # Normalize to get valid probability distributions
        emis_guess = emis_guess / np.sum(emis_guess, axis=1)[:, np.newaxis]
        model.emissionprob_ = emis_guess
        
        # Fit the model to the data (estimate parameters)
        model.fit(X)
        
        # Calculate BIC to evaluate model quality
        bic_value = calculate_bic(model, X)
        
        # Store model and its BIC value
        models_with_bic.append((model, bic_value))
        
        print(f"Repeat {i+1}/{n_repeats} - BIC: {bic_value:.2f}")
    
    return models_with_bic

# Reshape X for training
X_train = X.copy()

# Perform multiple fits
print(f"Performing {n_repeats} repeats of model fitting...")
models_with_bic = perform_multiple_fits(X_train, n_repeats)

# Sort models by BIC (lower is better)
models_with_bic.sort(key=lambda x: x[1])

# Get best model (lowest BIC)
best_model, best_bic = models_with_bic[0]
worst_model, worst_bic = models_with_bic[-1]

# Use the best model for further analysis
model_est = best_model
TRANS_EST = model_est.transmat_
EMIS_EST = model_est.emissionprob_

# Orient emission matrix to match original
forward_ = np.sum(np.abs(EMIS_EST - emis))
backward_ = np.sum(np.abs(EMIS_EST[1::-1, :] - emis))
val, ind = min((forward_, 0), (backward_, 1))

if ind == 0:
    plot_emis_est = EMIS_EST
else:
    plot_emis_est = EMIS_EST[1::-1, :]

# Plot BIC values across all repeats
if show_plots:
    # Extract BIC values
    bic_values = [bic for _, bic in models_with_bic]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_repeats + 1), bic_values, 'o-')
    plt.xlabel('Repeat Number')
    plt.ylabel('BIC Value')
    plt.title('BIC Values Across All Repeats (Lower is Better)')
    plt.grid(True)
    plt.show()
    
    # Plot best and worst model parameters
    plt.figure(figsize=(15, 10))
    
    # Best model transition matrix
    plt.subplot(2, 3, 1)
    plt.imshow(best_model.transmat_, vmin=0, vmax=1)
    plt.title(f"Best Model Transition Matrix\nBIC: {best_bic:.2f}")
    plt.colorbar()
    
    # Worst model transition matrix
    plt.subplot(2, 3, 2)
    plt.imshow(worst_model.transmat_, vmin=0, vmax=1)
    plt.title(f"Worst Model Transition Matrix\nBIC: {worst_bic:.2f}")
    plt.colorbar()
    
    # Actual transition matrix
    plt.subplot(2, 3, 3)
    plt.imshow(trans, vmin=0, vmax=1)
    plt.title("Actual Transition Matrix")
    plt.colorbar()
    
    # Orient emission matrices to match original
    best_emis = best_model.emissionprob_
    worst_emis = worst_model.emissionprob_
    
    # Check if we need to flip the best model emission matrix
    forward_best = np.sum(np.abs(best_emis - emis))
    backward_best = np.sum(np.abs(best_emis[1::-1, :] - emis))
    if backward_best < forward_best:
        best_emis = best_emis[1::-1, :]
    
    # Check if we need to flip the worst model emission matrix
    forward_worst = np.sum(np.abs(worst_emis - emis))
    backward_worst = np.sum(np.abs(worst_emis[1::-1, :] - emis))
    if backward_worst < forward_worst:
        worst_emis = worst_emis[1::-1, :]
    
    # Best model emission matrix
    plt.subplot(2, 3, 4)
    plt.imshow(best_emis, vmin=0, vmax=1)
    plt.title(f"Best Model Emission Matrix\nBIC: {best_bic:.2f}")
    plt.colorbar()
    
    # Worst model emission matrix
    plt.subplot(2, 3, 5)
    plt.imshow(worst_emis, vmin=0, vmax=1)
    plt.title(f"Worst Model Emission Matrix\nBIC: {worst_bic:.2f}")
    plt.colorbar()
    
    # Actual emission matrix
    plt.subplot(2, 3, 6)
    plt.imshow(emis, vmin=0, vmax=1)
    plt.title("Actual Emission Matrix")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# Generate plots to compare best model with actual parameters
if show_plots:
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(trans, vmin=0, vmax=1)
    plt.title("Actual Transition Matrix")
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(TRANS_EST, vmin=0, vmax=1)
    plt.title(f"Best Model Transition Matrix\nBIC: {best_bic:.2f}")
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.imshow(emis, vmin=0, vmax=1)
    plt.title("Actual Emission Matrix")
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.imshow(plot_emis_est, vmin=0, vmax=1)
    plt.title(f"Best Model Emission Matrix\nBIC: {best_bic:.2f}")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# Estimate State Sequence
# We need to give the decoding algorithm our estimate of the model
# parameters (emission and transition matrices)
# We can check how well it does using the actual parameters vs the
# estimated parameters

# Create models with known and estimated parameters
model_actual = hmm.MultinomialHMM(n_components=2)
model_actual.startprob_ = np.array([0.5, 0.5])
model_actual.transmat_ = trans
model_actual.emissionprob_ = emis
model_actual.n_trials = 1

# Get state probabilities
_, pStates_actual = model_actual.score_samples(X)
_, pStates_est = best_model.score_samples(X)
_, pStates_est_worst = worst_model.score_samples(X)

# HMM decode returns PROBABILITIES for being in each state at each timepoint
# We can just pick the state with the larger probability at each time
state_estim_actual = (pStates_actual[:, 0] > pStates_actual[:, 1]).astype(int)
state_estim_est = (pStates_est[:, 0] > pStates_est[:, 1]).astype(int)
state_estim_est_worst = (pStates_est_worst[:, 0] > pStates_est_worst[:, 1]).astype(int)

# Concatenate estimates for visualization
all_states = np.vstack([states, state_estim_actual, state_estim_est, state_estim_est_worst])

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2, 1, 1)
plt.imshow(X.T, **img_kwargs)
plt.title('Observations')

ax2 = plt.subplot(2, 1, 2)
plt.imshow(all_states, **img_kwargs)
plt.yticks(
        [0, 1, 2, 3], 
        ['Actual', 'Using Actual Emissions', 'Using Estimated Emissions', 'Using Worst Estimated Emissions'])
plt.title('State Sequences')

# Link x-axes
ax1.sharex(ax2)

plt.tight_layout()
plt.show()
