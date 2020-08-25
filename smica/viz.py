import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from mne.viz.topomap import _plot_ica_topomap
from mne.filter import filter_data
from mne.time_frequency import psd_array_welch
from scipy.stats import kurtosis


def plot_extended(S, sfreq, f_scale, powers, ica_mne, sort=True,
                  plot_env=True, number=False, h_freq_filt=2., save=None,
                  t_min=1, y_lim=None, t_max=10, pdf_from_sources=True,
                  extension='pdf'):
    n_plots = 10
    n_s = S.shape[0]
    if sort is True:
        order = np.argsort(kurtosis(S, axis=1))[::-1]
    elif type(sort) is list:
        order = sort
        n_s = len(order)
    else:
        order = np.arange(n_s)
    if plot_env:
        n_col = 3
        width_ratios = [1.5, 1, .7]
        figsize = (3, 8)
    else:
        n_col = 2
        width_ratios = [1, .3]
        figsize = (3, 8)
    n_f = max(n_s // n_plots, 1)
    for i in range(n_f):
        f, ax = plt.subplots(n_plots, n_col, figsize=figsize,
                             gridspec_kw=dict(width_ratios=width_ratios,
                                              top=0.95,
                                              bottom=0.11,
                                              left=0.11,
                                              right=0.9,
                                              hspace=0.18,
                                              wspace=0.04))
        for j in range(n_plots):
            if i * n_plots + j >= n_s:
                break
            idx = order[i * n_plots + j]
            ax_idx = 0
            axe = ax[j]
            if plot_env:
                s = S[idx]
                s *= np.sign(np.mean(s ** 3))
                # amp = np.abs(hilbert(s))
                #
                # to_plot = amp[100:-100]
                # to_plot = filter_data(to_plot, sfreq=sfreq, l_freq=None,
                #                       h_freq=h_freq_filt)
                n_s = 3000
                idx_min = int(sfreq * t_min)
                idx_max = int(sfreq * t_max)
                slice_len = int((idx_max - idx_min) / n_s) + 1
                to_plot = s[idx_min:idx_max:slice_len]
                # loc = np.argmax(to_plot)
                # n_points = 3000
                # idx_loc = np.arange(max(0, loc - n_points),
                #                     min(len(to_plot), loc + n_points))
                # to_plot = to_plot[idx_loc]
                n_s = len(to_plot)
                t = np.linspace(t_min, t_max, n_s)
                axe[0].plot(t, to_plot, linewidth=0.4, color='k')
                # axe[0].set_xlim([t_min, t_max])
                axe[0].set_yticklabels([])
                if j == n_plots - 1:
                    times = np.arange(t_min, t_max + 1, dtype=int)
                    axe[0].set_xticks(times)
                    axe[0].set_xticklabels(['%d' % time for time in times])
                    axe[0].set_xlabel('time (sec.)')
                else:
                    axe[0].set_xticklabels([])
                axe[0].set_yticks([])
                for axis in ['top', 'left', 'right']:
                    axe[0].spines[axis].set_visible(False)
                ax_idx = 1
            # axe[0].set_xlim([0, n_s])
            freqs = np.arange(0, max(f_scale), 20, dtype=int)
            axe[ax_idx].set_xticks(freqs)
            if j == n_plots - 1:
                axe[ax_idx].set_xlabel('f (Hz)')

                axe[ax_idx].set_xticklabels(['%d' % freq for freq in freqs])
            else:
                axe[ax_idx].set_xticklabels([])
            if pdf_from_sources:
                s_ = s / np.sqrt(np.mean(s ** 2))
                fmin = min(f_scale)
                fmax = max(f_scale)
                pows, f_ = psd_array_welch(s_, sfreq, fmin, fmax)
                print(min(pows), max(pows))
            else:
                pows = powers[:, idx]
                f_ = f_scale
            axe[ax_idx].semilogy(f_, pows, color='k',
                                 linewidth=1)
            if y_lim is None:
                y_lim = np.min(powers), np.max(powers)
            axe[ax_idx].set_ylim(*y_lim)
            axe[ax_idx].grid()
            axe[ax_idx].set_yticklabels([])
            axe[ax_idx].minorticks_off()
            ica_mne._ica_names[idx] = ''
            _plot_ica_topomap(ica_mne, idx=idx, axes=axe[ax_idx + 1], title='',
                              sensors=False)
            if number is not False:
                if type(number) is np.ndarray:
                    string = '%d, %.2f' % (i * n_plots + j,
                                           number[i * n_plots + j])
                else:
                    string = '%d' % (i * n_plots + j + 1)
                axe[0].set_ylabel(string,
                                  rotation=0)
            if j == 0:
                if plot_env:
                    axe[0].set_title('Source')
                axe[ax_idx].set_title('Spectrum')
                axe[ax_idx + 1].set_title('Topo')
        if save is not None:
            plt.savefig('%s_%d.%s' % (save, i, extension))
    plt.show()
    return None
