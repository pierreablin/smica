"""Spectral matching ICA"""

__version__ = '0.0dev'


from .core_smica import SMICA
from .mne import ICA, transfer_to_ica, transfer_to_mne
from .utils import fourier_sampling, loss, compute_covariances
