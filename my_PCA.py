import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import norm, inv, eig
from mne.viz import plot_topomap



class MyPCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, x, y):
        """
        Args:
            x ndarray, shape(n_epochs, n_channels, n_samples/times): epochs
        Returns:
            self
        """
        unique_labels = np.unique(y)
        n_unique_labels = len(unique_labels)
        if n_unique_labels != 2: 
            raise ValueError(f"Could not fit CSP, there are {n_unique_labels} unique labels. Two were expected.")
    
        def PCA_of_class(class_index):
            class_label = unique_labels[class_index]
            class_epochs_mask = np.where(y==class_label)
            class_epochs = x[class_epochs_mask]
            mean_cov = np.mean([covarianceMatrix(epoch) for epoch in class_epochs], axis=0)
            eig_vals, eig_vecs = eig(mean_cov)
            eig_vals = eig_vals.real
            eig_vecs = eig_vecs.real
            return eig_vals, eig_vecs

        eig_vals_0, eig_vecs_0 = PCA_of_class(0)
        eig_vals_1, eig_vecs_1 = PCA_of_class(1)
        eig_vals = np.concatenate((eig_vals_0, eig_vals_1))
        eig_vecs = np.concatenate((eig_vecs_0, eig_vecs_1), axis=1)
        descending_order = np.argsort(eig_vals)[::-1]

        self.filters_ = eig_vecs.T[descending_order]
        self.patterns_ = pinv(self.filters_)

        return self

    def transform(self, x):
        tmp_filters = self.filters_[:self.n_components]
        filtered_epochs = np.asarray([tmp_filters @ epoch for epoch in x]) #(n_epochs, n_features, n_samples)
        final_output = (filtered_epochs**2).mean(axis=2) #(n_epochs, n_features)
        final_output = np.log(final_output) #Removing the log seems to increase classification accuracy.
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