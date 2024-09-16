from time import time
import sys
import argparse

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from mne import Epochs, pick_types, events_from_annotations, set_log_level
from mne.channels import make_standard_montage
from mne.io import Raw, concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne import Info
import pandas as pd
from pandas import DataFrame as DF
from pandas import Series
from rich import print
from rich.traceback import install as install_rich_traceback
from rich.progress import track

from my_CSP import MyCSP
from my_PCA import MyPCA
from constants import *

def main():
    parser = argparse.ArgumentParser(description=MY_BCI_ARGPARSE_DESCRIPTION)

    # Add optional positional arguments
    parser.add_argument('subject_idx', type=int, nargs='?', help="An integer argument.")
    parser.add_argument('task_idx', type=int, nargs='?', help="task index between 3 and 14.")
    parser.add_argument('mode', type=str, nargs='?', help="The subject index, between 1 and 109.")


    # Parse the command-line arguments
    args = parser.parse_args()
    # Logic to call the correct function
    if args.subject_idx is None and args.task_idx is None and args.mode is None:
        scores = test_model_on_all_runs(dim_red_class=MyPCA)
    elif args.subject_idx is not None and args.task_idx is not None and args.mode is not None:
        if args.subject_idx < 1 or args.subject_idx > 109:
            print("[red]Error: The subject index is not in bound [1, 109]")
        elif args.task_idx < 3 or args.task_idx > 14:
            print("[red]Error: The task index is not in bound [3, 14]")
        elif args.mode == "predict":
            raw, events, epochs = load_epochs(args.task_idx, args.subject_idx)
            train_and_test_model(epochs, print_predictions=True)
        elif args.mode == "train":
            raw, events, epochs = load_epochs(args.task_idx, args.subject_idx)
            train_and_test_model(epochs, print_train_accuracy=True)
        else:
            print("Error: The third argument must be either 'predict' or 'train'")
    else:
        print("Error: The program expects either 0 or 3 arguments.")


def test_model_on_all_runs(dim_red_class=MyCSP) -> DF:
    scores = {}
    for subject_idx in track(ALL_SUBJECTS):
        task_accuracies = {}
        for task_name, task_idx in TASKS_IDX.items():
            raw, events, epochs = load_epochs(task_idx, subject_idx)
            bci_clf, accuracies = train_and_test_model(epochs, dim_red_class=dim_red_class)
            task_accuracies[task_name] = accuracies
        task_accuracies = DF.from_records(task_accuracies).T
        scores[subject_idx] = task_accuracies
    scores = pd.concat(scores, axis="index", keys=scores.keys(), names=["subject", "task"])
    print(scores)
    print(scores["test"].groupby(level=1).describe())
    print(scores["test"].describe())
        
    return scores

def load_epochs(taks_idx:list[int], subject_idx:int) -> Epochs:
    raw_fnames = eegbci.load_data(subject=subject_idx, runs=taks_idx, verbose=False)# - Get paths to edf files.
    raw:Raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames]) # - Load(read_raw_edf) in memomry(preloard=True) recprdings and convcatenate them.
    eegbci.standardize(raw)  # set channel names                                    # - Not sure what this does but it's necessary...
    raw.set_montage(make_standard_montage("standard_1005"))                         # - Specify to MNE what montage/setup was used during the recording.
                                                                                    #   "standard_1005" refers to a standardized way of placing electrodes on the testee.
                                                                                    #   See https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG).
    raw.filter(9, 30.0, skip_by_annotation="edge")                                  # - Apply bandpass filter, only keep frequecies in the range 7-30 Hz.
    events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))             # - Make events from annotaions, only use T1 and T2 annotations. Mark them as 0 and 1 respectively.
                                                                                    #   According to the Physionet EEG-MI dataset web page(https://physionet.org/content/eegmmidb/1.0.0/),
                                                                                    #   T0 corresponds to the motion of both fists and T1 to the motion of both feet.
    picks = pick_types(raw.info, eeg=True)                                          # - Specify that we only want to listen to the EEG channels, the other channels are set to False by default.
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
    return raw, events, epochs

def train_and_test_model(epochs: Epochs, dim_red_class=MyCSP, print_predictions=False, print_train_accuracy=False) -> tuple[Pipeline, Series]:
    x = epochs.get_data(tmin=1.0, tmax=2.0)
    y = epochs.events[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)
    
    cv = ShuffleSplit(10, test_size=0.2, random_state=21)

    # Assemble the classifier
    bci_pipeline = Pipeline([
        ("dim_reduction", dim_red_class(n_components=2)),
        ("clf", LinearDiscriminantAnalysis())
    ])
    # Use scikit-learn Pipeline with cross_val_score function
    training_accuracies = cross_val_score(bci_pipeline, x_train, y_train, cv=cv)
    if print_train_accuracy:
        print("cross validation accuracies:", training_accuracies)
        print("mean:", training_accuracies.mean())
    training_accuracies = {f"train_fold_{fold_idx}": accuracy for fold_idx, accuracy in enumerate(training_accuracies)}
    bci_clf = bci_pipeline.fit(x_train, y_train)
    test_accuracy = bci_clf.score(x_test, y_test)
    if print_predictions:
        y_pred = bci_clf.predict(x)
        print(DF({"prediction": y_pred, "truth": y_pred}).rename_axis("epoch").eval("correct = prediction == truth"))
        print("test accurracy:", test_accuracy)
    scores = training_accuracies | {"test": test_accuracy}
    # scores = Series(training_accuracies | {"test": test_accuracy})
    
    return bci_clf, scores



if __name__ == "__main__":
    set_log_level(verbose=False)
    install_rich_traceback(extra_lines=0, width=130)
    try: 
        main()
    except KeyboardInterrupt as _:
        print("[blue]exiting...")