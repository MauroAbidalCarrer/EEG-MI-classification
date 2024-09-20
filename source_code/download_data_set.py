from rich.progress import track
from mne.datasets import eegbci

from constants import *
from my_bci import load_epochs


for subject_idx in track(ALL_SUBJECTS):
    eegbci.load_data(
        subject=subject_idx,
        runs=ALL_TASKS,
        verbose=False,
        path="/mnt/nfs/homes/maabidal/sgoinfre/physionet",
        update_path=True,

    )