"""
Python translation of hmm_simulated.m
Labelling simulated data using HMM
"""

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import random

# If false, shows only the main plot
show_plots = True

# Create transitions matrix with strong self-transition probabilities
trans = np.array([[0.975, 0.025],
                  [0.025, 0.975]])

# Create very different emissions for easier detection
emis = np.array([[3/6, 2/6, 1/6, 1/6, 1/6, 1/6],  # State1 emissions
                 [2/10, 1/10, 1/10, 3/10, 1/10, 1/2]])  # State2 emissions

# Normalize emission matrix to make sure everything adds up to 1
emis = emis / np.sum(emis, axis=1)[:, np.newaxis]

# Plot the matrices
if show_plots:
    img_kwargs = {'interpolation': 'nearest', 'aspect': 'auto'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(trans, **img_kwargs)
    ax1.set_title('Transition matrix')
    ax1.set_xlabel('To state')
    ax1.set_ylabel('From state')
    im = ax2.imshow(emis.T, **img_kwargs, vmin=0, vmax=1)
    cax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cax, label='Probability')
    ax2.set_title('Emission matrix')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('State')
    plt.show()

# Generate data using hmmlearn
model = hmm.MultinomialHMM(n_components=2, n_iter=100)
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = trans
model.emissionprob_ = emis
# Not sure why this has to be set at all...some internal inconsistency?
model.n_trials = 1

# Generate sequence
dat_length = 1000
X, states = model.sample(dat_length)

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

# Estimate parameters
# First we have to estimate the emission and transition matrices
# We'll assume: no information about the emissions (random values)
# We'll assume: high self-transition probabilities
TRANS_GUESS = np.eye(trans.shape[0]) + np.random.rand(*trans.shape) * 0.05

## TRY RANDOM GUESS FOR EMISSIONS VS. MEAN OF DATA
## AND SEE THE DIFFERENCE
# EMIS_GUESS = np.random.rand(*emis.shape)
EMIS_GUESS = np.tile(X.mean(axis=0), (2, 1)) + np.random.rand(*emis.shape) * 0.05 

# Make sure numbers add up to 1
EMIS_GUESS = EMIS_GUESS / np.sum(EMIS_GUESS, axis=1)[:, np.newaxis]
TRANS_GUESS = TRANS_GUESS / np.sum(TRANS_GUESS, axis=1)[:, np.newaxis]

# Estimate parameters
model_est = hmm.MultinomialHMM(n_components=2, n_iter=1000)
model_est.startprob_ = np.array([0.5, 0.5])
model_est.transmat_ = TRANS_GUESS
model_est.emissionprob_ = EMIS_GUESS

# Reshape X for training
X_train = X.copy()
model_est.fit(X_train)

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

# Generate plots to see how we did
if show_plots:
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(trans, vmin=0, vmax=1)
    plt.title("Actual Transition Matrix")
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(TRANS_EST, vmin=0, vmax=1)
    plt.title("Estimated Transition Matrix")
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.imshow(emis, vmin=0, vmax=1)
    plt.title("Actual Emission Matrix")
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.imshow(plot_emis_est, vmin=0, vmax=1)
    plt.title("Estimated Emission Matrix")
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

model_est_decode = hmm.MultinomialHMM(n_components=2)
model_est_decode.startprob_ = np.array([0.5, 0.5])
model_est_decode.transmat_ = TRANS_EST
model_est_decode.emissionprob_ = EMIS_EST
model_est_decode.n_trials = 1

# Get state probabilities
_, pStates_actual = model_actual.score_samples(X)
_, pStates_est = model_est_decode.score_samples(X)

# HMM decode returns PROBABILITIES for being in each state at each timepoint
# We can just pick the state with the larger probability at each time
state_estim_actual = (pStates_actual[:, 0] > pStates_actual[:, 1]).astype(int)
state_estim_est = (pStates_est[:, 0] > pStates_est[:, 1]).astype(int)

# Concatenate estimates for visualization
all_states = np.vstack([states, state_estim_actual, state_estim_est])

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2, 1, 1)
plt.imshow(X.T, **img_kwargs)
plt.title('Observations')

ax2 = plt.subplot(2, 1, 2)
plt.imshow(all_states, **img_kwargs)
plt.yticks([0, 1, 2], ['Actual', 'Using Actual Emissions', 'Using Estimated Emissions'])
plt.title('State Sequences')

# Link x-axes
ax1.sharex(ax2)

plt.tight_layout()
plt.show()
