import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh, norm, inv, eig
from mne.viz import plot_topomap
import matplotlib.pyplot as plt

from mne import Epochs, pick_types, events_from_annotations, set_log_level
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

def pinv(a, rtol=None):
    """Compute a pseudo-inverse of a matrix."""
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    del a
    maxS = np.max(s)
    if rtol is None:
        rtol = max(vh.shape + u.shape) * np.finfo(u.dtype).eps
    rank = np.sum(s > maxS * rtol)
    u = u[:, :rank]
    u /= s[:rank]
    return (u @ vh[:rank]).conj().T

def CSP(a_class_epochs, b_class_epochs):
    mean_cov_over_epochs = lambda epochs : np.mean([covarianceMatrix(task) for task in epochs], axis=0)
    return spatialFilter(mean_cov_over_epochs(b_class_epochs), mean_cov_over_epochs(a_class_epochs))

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
    cov_mat = A @ A.T
    return cov_mat / np.trace(cov_mat) 


def spatialFilter(Ra,Rb):
    R = Ra + Rb
    E,U = eig(R)

    # CSP requires the eigenvalues E and eigenvector U be sorted in descending order
    ord = np.argsort(E)
    ord = ord[::-1] # argsort gives ascending order, flip to get descending
    E = E[ord]
    U = U[:,ord]
    U = U.real

    # Find the whitening transformation matrix
    P = np.dot(np.sqrt(inv(np.diag(E))),np.transpose(U)).real

    # The mean covariance matrices may now be transformed
    Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
    Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

    # Find and sort the generalized eigenvalues and eigenvector
    E1,U1 = eig(Sa,Sb)
    ord1 = np.argsort(E1)
    ord1 = ord1[::-1]
    E1 = E1[ord1]
    U1 = U1[:,ord1]
    U1 = U1.real

    # The projection matrix (the spatial filter) may now be obtained
    SFa = np.dot(np.transpose(U1),P)
    return SFa.astype(np.float32)

class MyCSP(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=3):
        """Set the number of components/features. Can be changed later.

        Args:
            n_components (int, optional): Number of filters to use when transforming signal. Defaults to 1.
        """
        self.n_components = n_components
        

    def fit(self, x, y):
        """
        Args:
            x ndarray, shape(n_epochs, n_channels, n_samples/times): epochs
            y ndarray, shape(n_epochs): event labels there must be 2 unique values
        """
        unique_labels = np.unique(y)
        n_unique_labels = len(unique_labels)
        if n_unique_labels != 2: 
            raise ValueError(f"Could not fit CSP, there are {n_unique_labels} unique labels. Two were expected.")

        epochs_per_class = [] # n_classes, n_channels, n_samples
        for class_index in range(2):
            class_label = unique_labels[class_index]
            class_epochs_mask = np.where(y==class_label)
            class_epochs = x[class_epochs_mask]
            # epochs_per_class.append(np.transpose(class_epochs, (0, 2, 1)))
            epochs_per_class.append(class_epochs)

        # print(samples_per_class[0].T.shape)
        self.filters_ = CSP(epochs_per_class[0], epochs_per_class[1])

        self.patterns_ = pinv(self.filters_)
        return self
    
    def transform(self, x):
        """
        Extracts features from signal.

        Args:
            x (ndarray, shape(n_epochs, n_channels, n_times/samples)): signal epochs
        Returns:
            ndarray, shape(n_epochs, n_components, n_times/samples): extracted features
        """
        tmp_filters = self.filters_[:self.n_components]
        filtered_epochs = np.asarray([tmp_filters @ epoch for epoch in x]) #(n_epochs, n_features, n_samples)
        final_output = (filtered_epochs**2).mean(axis=2) #(n_epochs, n_features)
        final_output = np.log(final_output)
        # print('final_output.shape:', final_output.shape, '\n')
        return final_output
    

    def plot_patterns(self, raw_info, ch_type='eeg'):
        for i, component in enumerate(self.filters_.T[:self.n_components]):
            plot_topomap(
                component,
                raw_info,
                size=1.5,
                ch_type=ch_type
            )
        print('normal filters:')
        for component in self.filters_[:self.n_components]:
             plot_topomap(
                component,
                raw_info,
                size=1.5,
                ch_type=ch_type
            )