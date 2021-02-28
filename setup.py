#! /usr/bin/env python
"""Setup SMICA."""
import os
from setuptools import setup, find_packages

# get the version
version = None
with open(os.path.join('smica', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

descr = 'SMICA: Spectral Matching Independent Component Analysis'

DISTNAME = 'smica'
DESCRIPTION = descr
MAINTAINER = 'Pierre Ablin'
MAINTAINER_EMAIL = 'pierreablin@gmail.com'
URL = 'https://github.com/pierreablin/smica'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/pierreablin/smica.git'
VERSION = version

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          long_description_content_type='text/markdown',
          python_requires='~=3.6',
          install_requires=[
              'numpy >=1.14',
              'scipy >=0.18.1',
              'joblib',
              'mne',
              'qndiag',
              'scikit-learn',
              'numba',
              'matplotlib',
          ],
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
          ],
          platforms='any',
          packages=find_packages(),
          project_urls={
              'Documentation': 'https://pierreablin.github.io/smica/',
              'Bug Reports': 'https://github.com/pierreablin/smica/issues',
              'Source': 'https://github.com/pierreablin/smica',
          },
          )
