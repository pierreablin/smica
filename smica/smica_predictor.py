import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

from .core_smica import SMICA


class SMICAPredictor(BaseEstimator, ClassifierMixin):
    '''
    SMICA transfomer for epochs processing
    '''
    def __init__(self, n_components, freqs, sfreq, avg_noise=False, rng=None):
        self.n_components = n_components
        self.freqs = freqs
        self.sfreq = sfreq
        self.rng = check_random_state(rng)
        self.avg_noise = avg_noise

    def fit(self, X, y=None, **kwargs):
        '''
        X is of shape (n, p, T)
        '''
        self.classes_ = np.unique(y)
        self.smicas = {}
        for classe in self.classes_:
            smica = SMICA(self.n_components, self.freqs, self.sfreq,
                          self.avg_noise, self.rng)
            smica.fit(np.hstack(X[y == classe]), **kwargs)
            self.smicas[classe] = smica
        return self

    def predict_log_proba(self, X):
        n, _, _ = X.shape
        n_classes = len(self.classes_)
        log_probas = np.zeros((n, n_classes))
        for i, x in enumerate(X):
            for j, classe in enumerate(self.classes_):
                smica = self.smicas[classe]
                log_probas[i, j] = smica.compute_loss(x)
        print(log_probas)
        return -log_probas

    def predict(self, X):
        """Predict class labels for samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        scores = self.predict_log_proba(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]
