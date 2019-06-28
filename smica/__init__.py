"""Spectral matching ICA"""

__version__ = '0.0dev'


from .core_fitter import CovarianceFit
from .core_smica import SMICA
from .mne import ICA, transfer_to_ica, transfer_to_mne, plot_components
from .utils import fourier_sampling, loss, compute_covariances
from .smica_transformer import SMICATransformer
from .smica_predictor import SMICAPredictor
from .sobi import sobi, SOBI, SOBI_mne
from .noiseless_jd import JDIAG, JDIAG_mne
from .dipolarity import dipolarity
