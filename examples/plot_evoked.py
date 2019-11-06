# We compare smica with other denoising techniques to obtain clain erps
import os
import numpy as np
import matplotlib.pyplot as plt
from smica import ICA, transfer_to_ica
import mne
from mne.datasets import sample
from mne.preprocessing import ICA as ICA_mne
from mne.filter import filter_data


fontsize = 10
rc = {"pdf.fonttype": 42, 'text.usetex': False, 'font.size': fontsize,
      'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize,
      'text.latex.preview': True}


plt.rcParams.update(rc)
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')
raw.set_eeg_reference('average', projection=True)  # set EEG average reference
events = mne.find_events(raw)
tmin, tmax = -0.2, 0.5
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4}

reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
# Fit smica
freqs = np.linspace(1, 70, 41)
smica = ICA(n_components=20,
            freqs=freqs, rng=0).fit(raw, picks)


colors = ['k', 'darkred', 'forestgreen', 'royalblue', 'orange']

# Plot the clusters

# smica.plot_extended(sort=False)

bad_sources = [17, 18, 19]

X_filtered = smica.filter(raw._data[picks], bad_sources=bad_sources, method='wiener')
raw_filtered = raw.copy()
raw_filtered._data[picks] = X_filtered

raw.filter(1, 70)
ica = ICA_mne(n_components=20, method='fastica', random_state=0)
ica.fit(raw, picks=picks)

sources = ica.get_sources(raw).get_data()
ica_mne = transfer_to_ica(raw, picks, freqs,
                          ica.get_sources(raw).get_data(),
                          ica.get_components())

# ica_mne.plot_extended(sources, sort=False)
bads_infomax = [0, 1, 2]
X_ifmx = ica_mne.filter(raw._data[picks], bad_sources=bads_infomax, method='pinv')
raw_ifmx = raw.copy()
raw_ifmx._data[picks] = X_ifmx
# We identify that clusters 6, 7, 8, 9 correspond to noise

max_raw = raw.copy()
max_raw = mne.preprocessing.maxwell_filter(max_raw)
setups = []
setups.append({'raw': raw, 'proj': False, 'name': 'Unfiltered'})
setups.append({'raw': raw, 'proj': True, 'name': 'SSP'})
setups.append({'raw': max_raw, 'proj': False, 'name': 'Maxwell'})
setups.append({'raw': raw_ifmx, 'proj': False, 'name': 'Infomax'})
setups.append({'raw': raw_filtered, 'proj': False, 'name': 'SMICA'})
# Epochs the signal and observe the average auditory response

f, axes = plt.subplots(4, 3, figsize=(8, 6),
                       gridspec_kw=dict(width_ratios=[2, 1, .7],
                                        top=0.91,
                                        bottom=0.075,
                                        left=0.04,
                                        right=0.98,
                                        hspace=0.215,
                                        wspace=0.03))

expes = ['Auditory/Right', 'Auditory/Left', 'Visual/Right', 'Visual/Left']
peaks = [(160, 200), (150, 200), (155, 210), (190, 260)]
for ee, (expe, peak) in enumerate(zip(expes, peaks)):
    ax = axes[ee]
    evokeds = []
    for setup in setups:
        epochs = mne.Epochs(setup['raw'], events=events, event_id=event_id, tmin=tmin,
                            tmax=tmax, baseline=(None, 0.), proj=setup['proj'],
                            reject=reject)
        evokeds.append(epochs[expe].average(picks=picks))

    # Plot average

    sfreq = smica.sfreq
    n_samples = evokeds[0].data.shape[1]
    t_large = np.linspace(-.2, n_samples / sfreq - .2, n_samples)
    markers = [',', 'o', 's', 'P', '*']
    markevery = .2

    for i, (name, evoked, marker) in enumerate(zip(['Raw', 'SSP', 'Maxwell',
                                                    'Infomax', 'SMICA'],
                                                   evokeds, markers)):
        # if ee == 1:
        #     evoked.plot(titles=name)
        to_plot = np.sqrt(np.mean(evoked.data ** 2, axis=0))
        y_ = ax[0].set_ylabel(expe)
        ax[0].plot(t_large, to_plot, color=colors[i])
                   # marker=marker, markevery=markevery)
        to_plot_ = np.sqrt(np.mean(evoked.copy().crop(None, 0).data ** 2,
                                   axis=0))
        # to_plot_ = filter_data(to_plot_, 100, None, 5)
        t_zoom = np.linspace(-.2, len(to_plot_) / sfreq - .2, len(to_plot_))
        # ax[1].plot(t_zoom, to_plot_, color=colors[i], alpha=0.5)
        # avg = np.mean(to_plot_)
        # ax[1].axhline(avg, color=colors[i], linewidth=2)
        ax[1].plot(t_zoom, to_plot_, color=colors[i], linestyle='--', alpha=.3)
        ax[1].plot(t_zoom, np.mean(to_plot_, axis=0) * np.ones(len(t_zoom)),
                   color=colors[i], marker=marker, markevery=markevery)
        # ax[0].set_yscale('log')
        # sl = range(180, 260)
        sl = range(peak[0], peak[1])
        if ee == 3:
            ax[2].plot(t_large[sl], to_plot[sl], label=name, color=colors[i])
                       # marker=marker, markevery=markevery)
        else:
            ax[2].plot(t_large[sl], to_plot[sl], color=colors[i])
                       # marker=marker, markevery=markevery)
        if ee == 0:
            ax[0].set_title('Total evoked response')
            ax[1].set_title('Zoom on t<0')
            ax[2].set_title('Zoom on the peak')
        for u in range(3):
            ax[u].set_yticks([])
            ax[u].set_yticklabels([])
            if ee == 3:
                t_ = ax[u].set_xlabel('Time (s.)')

        # to_plot = np.std(evoked.data, axis=0)
        # ax[1, 0].plot(t_large, to_plot)
        # ax[1, 1].plot(t_large[sl], to_plot[sl], label=name)
l_ = f.legend(loc='upper center', ncol=len(evokeds))
plt.savefig('figures/evoked.pdf', bbox_inches='tight',
            bbox_extra_artists=[l_, t_, y_])
plt.show()
