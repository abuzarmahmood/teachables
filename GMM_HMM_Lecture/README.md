# Gaussian Mixture Models and Hidden Markov Models

This directory contains Python implementations of Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM) for data analysis.

## Python Scripts

### GMM Scripts
- **gmm_2D_simulated.py**: Demonstrates GMM clustering on 2D simulated data, visualizing the mixture components and decision boundaries.
- **gmm_spikes.py**: Applies GMM to neural spike waveform data, performing dimensionality reduction with PCA and clustering to identify different neuron types.

### HMM Scripts
- **hmm_simulated.py**: Simulates a hidden Markov model with two states and multinomial emissions, then demonstrates parameter estimation and state decoding.

## Setting Up a Virtual Environment

To run these scripts, you'll need to set up a Python virtual environment and install the required dependencies.

### Steps to Set Up

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

## Running the Scripts

After setting up your environment, you can run any of the scripts:

```bash
python gmm_2D_simulated.py
python gmm_spikes.py
python hmm_simulated.py
```

Each script includes visualization components that can be toggled with the `show_plots` variable.
