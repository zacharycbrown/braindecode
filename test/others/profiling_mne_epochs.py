"""This script is used to profile the different ways of extracting epochs from
mne.Epochs.
"""

import os
import time

import mne
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt


mne.set_log_level('CRITICAL')
plt.ion()


def create_mne_raw(n_channels, n_times, sfreq, savedir=None):
    """Create an mne.io.RawArray with fake data, and save it as .fif and .hdf5.
    """
    data = np.random.rand(n_channels, n_times)
    ch_names = [f'ch{i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)

    if isinstance(savedir, str):
        raw_fname = os.path.join(savedir, 'fake_eeg_raw.fif')
        raw.save(raw_fname, overwrite=True)
        h5_fname = raw_fname.replace('_raw.fif', '.hdf5')
        with h5py.File(h5_fname, 'w') as f:
            dset = f.create_dataset('fake_raw', dtype='f16', data=raw.get_data())
    else:
        raw_fname, h5_fname = '', ''

    return raw, raw_fname, h5_fname


def raw_to_epochs(raw, win_len_s, win_overlap_s, preload=False):
    """Extract epochs from mne.io.Raw.
    """
    events = mne.make_fixed_length_events(
        raw, id=1, start=0, stop=None, duration=win_len_s, first_samp=True, 
        overlap=win_overlap_s)

    md_columns = ['subject', 'session', 'run', 'age', 'label']
    metadata = pd.DataFrame(
        np.zeros((events.shape[0], len(md_columns))), columns=md_columns)

    tmax = win_len_s - 1. / raw.info['sfreq']
    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=0, tmax=tmax, baseline=None,
        preload=preload, metadata=metadata)
    
    start_end_inds = np.vstack(
        (events[:, 0], events[:, 0] + int(win_len_s * raw.info['sfreq']))).T

    return epochs, start_end_inds


def create_mne_epochs(n_epochs, n_channels, n_times, sfreq=100, savedir=None):
    """Create an mne.Epochs object with fake data.
    """
    ch_names = [f'ch{i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels

    # Create random epoch data
    epochs_data = np.random.rand(n_epochs, n_channels, n_times)

    # Create events
    event_id = 1
    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 0] = sfreq * n_times * np.arange(n_epochs)
    events[:, -1] = event_id

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    epochs = mne.EpochsArray(
        epochs_data, info=info, events=events, event_id={'arbitrary': 1})
    # picks = mne.pick_types(info, meg=False, eeg=True, misc=False)

    if isinstance(savedir, str):
        fname = os.path.join(savedir, 'fake_epochs_epo.fif')
        epochs.save(fname, overwrite=True)
    else:
        fname = ''

    return epochs, fname


def time_statement(statement, n, verbose=False):
    """Measure execution time of a statement n times.

    This works, however it adds a little variable overhead (due to eval?)...
    """
    start = time.time()
    for i in range(n):
        eval(statement)
    duration = (time.time() - start) * 1e3 / n

    if verbose:
        print('\nPreload=False')
        print(f'--- {duration} ms ---')

    return duration


if __name__ == '__main__':
    """
    To run on a single CPU, use:
    >>> taskset -c 0 python profiling_mne_epochs.py
    """

    n_loops = 500
    verbose = False
    savedir = os.path.dirname(os.path.realpath(__file__))
    # savedir = '/storage/store/work/hjacobba/data/tests'

    # Create fake data and save it on disk
    n_channels = 32
    sfreq = 128
    n_times = np.linspace(10 * sfreq, 8 * 60 * 60 * sfreq, 5, dtype=int)

    durations = dict()
    for i, n in enumerate(n_times):
        print(f'Timing case where n_times={n} ({i}/{len(n_times)})...')

        raw, raw_fname, hdf5_fname = create_mne_raw(
            n_channels, n, sfreq, savedir=savedir)
        hf = h5py.File(hdf5_fname, 'r')

        raw = mne.io.read_raw_fif(raw_fname, preload=False, verbose=None)
        epochs, start_end_inds = raw_to_epochs(raw, 2, 0.5, preload=False)

        ### CASE 1 ###
        start = time.time()
        for i in range(n_loops):
            x1 = epochs[0].get_data()[0]
        duration1 = (time.time() - start) * 1e3 / n_loops

        epochs.load_data()

        ### CASE 2 ###
        start = time.time()
        for i in range(n_loops):
            x2 = epochs[0].get_data()[0]
        duration2 = (time.time() - start) * 1e3 / n_loops

        ### CASE 3 ###
        start = time.time()
        for i in range(n_loops):
            x3 = epochs._data[0]
        duration3 = (time.time() - start) * 1e3 / n_loops

        ### CASE 4 ###
        start = time.time()
        for i in range(n_loops):
            x4 = hf['fake_raw'][:, start_end_inds[0, 0]:start_end_inds[0, 1]]
        duration4 = (time.time() - start) * 1e3 / n_loops

        durations[n] = {
            'mne: preload=False': duration1,
            'mne: get_data()': duration2,
            'mne: _data[index]': duration3,
            'hdf5': duration4}
    
        hf.close()
        
        # Make sure all methods return the same thing
        # tol = 1e-10
        # assert (np.allclose(x1, x2, atol=tol) & 
        #         np.allclose(x2, x3, atol=tol) & 
        #         np.allclose(x3, x4, atol=tol))

    durations_df = pd.DataFrame(durations).T
    ax = durations_df.plot(marker='o')
    ax.set_ylabel('Execution time per loop (ms)')
    ax.set_xlabel('Raw length (#samples)')
    ax.set_title('Time taken to get epochs with various methods')
    plt.savefig(os.path.join(savedir, 'timing_results.png'))
