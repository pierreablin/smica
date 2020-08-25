# Spectral Matching Independent Component Analysis

This repository contains the python code for the Spectral Matching Independent Component Analysis (SMICA) algorithm.

It contains three main functions:

* `smica.CovarianceFit` implements a class to fit a noisy covariance model, that takes a sequence of covariance matrices and decomposes them.
* `smica.SMICA` implements the core SMICA algorithm, that takes signals, computes their spectral covariances, and computes the estimated parameters (mixing matrix, source powers and noise powers). It can then perform Wiener filtering to compute the estimated sources.
* `smica.ICA` implements SMICA in the `mne` framework, to handle easily M/EEG recordings. It emulates some properties of the `mne.preprocessing.ICA` class.


For an example, you can run the `examples/plot_meg_decomposition.py` file.
