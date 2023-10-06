import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations, set_log_level
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

from my_CSP import MyCSP

set_log_level(verbose=False)

# Load data.
raw_fnames = eegbci.load_data(subject=1, runs=[6, 10, 14])                  # -Get paths to edf files.
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames]) # -Load(read_raw_edf) in memomry(preloard=True) recprdings and convcatenate them.
eegbci.standardize(raw)  # set channel names                                # -Not sure what this does but it's necessary...
raw.set_montage(make_standard_montage("standard_1005"))                     # -Specify to MNE what montage/setup was used during the recording.
                                                                            #  "standard_1005" refers to a standardized way of placing electrodes on the testee.
                                                                            #  See https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG).
raw.filter(7.0, 30.0, skip_by_annotation="edge")                            # -Apply bandpass filter, only keep frequecies in the range 7-30 Hz.
events, _ = events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))   # -Make events from annotaions, only use T1 and T2 annotations. Mark them as 0 and 1 respectively.
                                                                            #  According to the Physionet EEG-MI dataset web page(https://physionet.org/content/eegmmidb/1.0.0/),
                                                                            #  T0 corresponds to the motion of both fists and T1 to the motion of both feet.
picks = pick_types(raw.info, eeg=True)                                      # -Specify that we only want to listen to the EEG channels, the other channels are set to False by default.
# Testing will be done with a running classifier
epochs = Epochs(
    raw,
    events,
    tmin=-1.0,
    tmax=4.0,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)

# Construct and fit model.
epochs_data_train = epochs.get_data(tmin=1.0, tmax=2.0)
labels = epochs.events[:, 2]
csp = MyCSP(n_components=2)
lda = LinearDiscriminantAnalysis()
clf = Pipeline([("csp", csp), ("lda", lda)])
clf.fit(epochs_data_train, labels)

# Plot data in realtime.
raw_data, samples_times = raw.get_data(['eeg'], return_times=True, verbose=True)
raw_data = raw_data.T


plt.ioff()
plt.show()

# print(epochs.get_data().shape[2] / raw.info['sfreq'])
fig = plt.figure(figsize=(12,12))
SLICE_LEN = epochs.get_data().shape[2]
raw_data_slice = raw_data[:SLICE_LEN]
samples_times_slice = samples_times[:SLICE_LEN]

# define axis1, labels, and legend
ah1 = fig.add_subplot(211)
ah1.set_ylabel("Voltage [\u03BCV]", fontsize=14)
for i in range(raw_data_slice.shape[1]):
    ah1.plot(samples_times_slice, raw_data_slice[:, i], alpha=0.5)
ah1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)



# for 