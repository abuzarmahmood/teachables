# Gaussian Mixture Models and Hidden Markov Models

This directory contains implementations of Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM) for data analysis in both MATLAB and Python.

## Code Organization

The code is organized into two main directories:

- **MATLAB_code/**: Contains MATLAB implementations
- **python_code/**: Contains Python implementations

Both directories contain equivalent implementations of the same algorithms, allowing you to choose your preferred language.

## Algorithms Implemented

### GMM (Gaussian Mixture Models)
- **2D Simulated Data**: Demonstrates GMM clustering on 2D simulated data, visualizing the mixture components and decision boundaries.
- **Neural Spike Waveforms**: Applies GMM to neural spike waveform data, performing dimensionality reduction with PCA and clustering to identify different neuron types.

### HMM (Hidden Markov Models)
- **Simulated Data**: Simulates a hidden Markov model with two states and multinomial emissions, then demonstrates parameter estimation and state decoding.

## Running the MATLAB Code

To run the MATLAB scripts, you'll need MATLAB with the Statistics and Machine Learning Toolbox. Open MATLAB and navigate to the `MATLAB_code` directory, then run:

```matlab
gmm_2D_simulated  % For 2D GMM demonstration
gmm_spikes        % For spike waveform clustering
hmm_simulated     % For HMM demonstration
```

## Running the Python Code

### Setting Up a Virtual Environment

To run the Python scripts, you'll need to set up a Python virtual environment and install the required dependencies.

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The `requirements.txt` file includes:
- numpy: For numerical operations
- scipy: For scientific computing and data processing
- matplotlib: For data visualization
- scikit-learn: For machine learning algorithms including GMM
- hmmlearn: For hidden Markov model implementations

### Running the Python Scripts

After setting up your environment, navigate to the `python_code` directory and run any of the scripts:

```bash
python gmm_2D_simulated.py
python gmm_spikes.py
python hmm_simulated.py
```

Each script includes visualization components that can be toggled with the `show_plots` variable.

## Data Files

The spike waveform analysis uses the `spike_waveforms.mat` file, which is included in both the MATLAB and Python directories. This file contains standardized neural spike waveforms for clustering.
