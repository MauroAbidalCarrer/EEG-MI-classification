import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh, norm, inv, eig
from mne.viz import plot_topomap
import matplotlib.pyplot as plt



class MyCSP(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=4):
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
        Returns:
            self
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
            epochs_per_class.append(class_epochs)

        mean_covariance = lambda epochs : np.mean([covarianceMatrix(task) for task in epochs], axis=0)
        mean_cov_b = mean_covariance(epochs_per_class[1])
        mean_cov_a = mean_covariance(epochs_per_class[0])
        self.filters_ = spatialFilter(mean_cov_b, mean_cov_a)
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
        # final_output = np.log(final_output) #Removing the log seems to increase classification accuracy.
        return final_output
    

    def plot_patterns(self, raw_info, ch_type='eeg'):
        for component in self.patterns_.T[:self.n_components]:
            plot_topomap(
                component,
                raw_info,
                size=1.5,
                ch_type=ch_type
            )
    

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
    cov_mat = A @ A.T
    return cov_mat / np.trace(cov_mat) 

def spatialFilter(Ra,Rb):
    def sorted_eig(vals, vecs):
        ord = np.argsort(vals)[::-1]
        return vals[ord].real, vecs[:, ord].real

    # Compute whitening matrix.
    E, U = sorted_eig(*eig(Ra + Rb))
    white_mat = np.sqrt(inv(np.diag(E))) @ U.T
    white_mat = white_mat.real

    # The mean covariance matrices may now be transformed
    Sa = white_mat @ Ra @ white_mat.T
    Sb = white_mat @ Rb @ white_mat.T

    # Find and sort the generalized eigenvalues and eigenvector
    _, U1 = sorted_eig(*eig(Sa, Sb))

    # The projection matrix (the spatial filter) may now be obtained
    SFa = U1.T @ white_mat
    return SFa.astype(np.float32)

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