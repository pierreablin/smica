.. picard documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SMICA
=====

This is a library to run the Spectral Matching ICA (SMICA) algorithm [1].
This algorithm exploit spectral diversity for source separation and
has an explicity noise model allowing to estimate less sources than channels
without any previous dimensionality reduction (eg. with PCA).

Installation
------------

Then install smica::

	$ pip install python-smica

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import smica'

and it should not give any error message.

Quickstart
----------

The easiest way to get started is to copy the following lines of code
in your script:

.. code:: python

    >>> import numpy as np
    >>> from smica import SMICA
    >>> rng = np.random.RandomState(0)
    >>> n_samples, n_channels = 1000, 5
    >>> X = rng.randn(n_channels, n_samples)
    >>> sfreq = 1
    >>> freqs = np.array([0.1, 0.2, 0.3])
    >>> smica = SMICA(n_components=3, freqs=freqs, sfreq=sfreq).fit(X)
    >>> estimated_sources = smica.compute_sources()  # doctest:+ELLIPSIS

Bug reports
-----------

Use the `github issue tracker <https://github.com/pierreablin/smica/issues>`_ to report bugs.

Cite
----

   [1] Ablin, Pierre, Jean-François Cardoso, and Alexandre Gramfort.
   "Spectral independent component analysis with noise modeling for M/EEG source separation."
   Journal of Neuroscience Methods (2021): 109144.


Arxiv
-----
https://arxiv.org/abs/2008.09693


API
---

.. toctree::
    :maxdepth: 1

    api.rst
