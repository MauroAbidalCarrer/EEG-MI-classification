import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh, norm
from mne.viz import plot_topomap

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

        samples_per_class = []
        for class_index in range(2):
            class_label = unique_labels[class_index]
            class_epochs_indices = np.where(y==class_label)
            class_epochs = x[class_epochs_indices]
            class_samples = np.concatenate(class_epochs, axis=1)
            samples_per_class.append(class_samples)

        # print('class_samples.shape: ', samples_per_class[0].shape)
        covariance_matrices = np.asarray([np.cov(class_samples @ class_samples.T) for class_samples in samples_per_class])
        gen_eig_vals, gen_eig_vecs = eigh(covariance_matrices[0], covariance_matrices.sum(0))
        gen_eig_vals = gen_eig_vals.real
        gen_eig_vecs = gen_eig_vecs.real
        normed_gen_eig_vecs = gen_eig_vecs / norm(gen_eig_vecs, axis=1, keepdims=True)
        descending_order_indices = np.argsort(abs(gen_eig_vals))[::-1]
        self.filters_ = normed_gen_eig_vecs[:, descending_order_indices].T
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
        return plot_topomap(
            self.patterns_[:, 0],
            raw_info,
            # units="Patterns (AU)",
            size=1.5,
            ch_type=ch_type
        )