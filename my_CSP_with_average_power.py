import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import norm, inv, eig
from mne.viz import plot_topomap



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
            y ndarray, shape(n_epochs): labels of epochs
        Returns:
            self
        """
        unique_labels, n_unique_counts = np.unique(y, return_counts=True)
        n_unique_labels = len(unique_labels)
        if n_unique_labels < 2: 
            raise ValueError(f"Could not fit CSP, there are {n_unique_labels} unique labels. At least two are expected.")
        
        # print('n_unique_counts:', n_unique_counts)

        mean_centered_epochs = x - np.mean(x, axis=2, keepdims=True)

        # epochs_avg_sig_pow = (np.abs(mean_centered_epochs)).mean(axis=2)
        # epochs_avg_sig_pow = np.log(epochs_avg_sig_pow)

        variance_epochs = np.sqrt(np.var(mean_centered_epochs, axis=2))
        # print('epochs_avg_sig_pow.shape:', epochs_avg_sig_pow.shape)

        def avg_sig_pow_cov_of_class(class_index):
            class_label = unique_labels[class_index]
            class_epochs_indices = np.where(y==class_label)
            class_avg_sig_pow_epochs = variance_epochs[class_epochs_indices]
            concat_class_avg_sig_pow_epochs = class_avg_sig_pow_epochs.T
            cov = covarianceMatrix(concat_class_avg_sig_pow_epochs)
            # print('cov has nans:', np.isnan(cov).any())
            # print('cov mean:', np.mean(cov))
            # print('cov diagonal:', np.diag(cov))
            return cov

        avg_sig_pow_cov_per_class = [avg_sig_pow_cov_of_class(class_i) for class_i in range(n_unique_labels)]

        def mean_avg_sig_pow_cov_of_other_classes(class_i):
            return np.mean([cov for i, cov in enumerate(avg_sig_pow_cov_per_class) if i != class_i], axis=0)
        filters_per_class = [self.spatialFilter(avg_sig_pow_cov_per_class[i], mean_avg_sig_pow_cov_of_other_classes(i)) for i in range(n_unique_labels)]
        self.filters_ = np.concatenate(filters_per_class, axis=0)
        self.patterns_ = pseudo_inverse(self.filters_)
        
        return self

    def transform(self, x):
        """
        Extracts features from signal.

        Args:
            x (ndarray, shape(n_epochs, n_channels, n_times/samples)): signal epochs
        Returns:
            ndarray, shape(n_epochs, n_components, n_times/samples): extracted features
        """
        filtered_epochs = np.asarray([self.filters_ @ epoch for epoch in x]) #(n_epochs, n_features, n_samples)
        final_output = (filtered_epochs**2).mean(axis=2) #(n_epochs, n_features)
        final_output = np.log(final_output) #Removing the log seems to increase classification accuracy.
        return final_output
    

    def plot_patterns(self, raw_info, ch_type='eeg'):
        for component in self.patterns_.T:
            plot_topomap(
                component,
                raw_info,
                size=1.5,
                ch_type=ch_type
            )

    def spatialFilter(self, Ra,Rb):
        def sorted_eig(vals, vecs):
            ord = np.argsort(vals.real)[::-1]
            return vals[ord].real, vecs[:, ord].real
        
        REGULARIZATION_PARAM = 0.0001
        regularization_mat = np.eye(64) * REGULARIZATION_PARAM
        Ra += regularization_mat
        Rb += regularization_mat
        
        # Tests
        has_negatives = lambda array : len(np.where(array<0)) > 0
        Ra_eig_vals, _ = eig(Ra)
        Rb_eig_vals, _ = eig(Rb)
        # print('Ra_eig_vals has negatives:', has_negatives(Ra_eig_vals))
        # print('Rb_eig_vals has negatives:', has_negatives(Rb_eig_vals))
        
        # Compute PCA whitening matrix.
        E, U = sorted_eig(*eig(Ra + Rb))

        # print("E has_negatives:", has_negatives(E))
        

        white_mat = np.sqrt(inv(np.diag(E))) @ U.T
        white_mat = white_mat.real

        # The mean covariance matrices may now be transformed.
        Sa = white_mat @ Ra @ white_mat.T
        Sb = white_mat @ Rb @ white_mat.T

        # print('Sa contains nans:', np.isnan(Sa).any())
        # print('Sb contains nans:', np.isnan(Sb).any())
        # Find and sort the generalized eigenvalues and eigenvector
        _, U1 = sorted_eig(*eig(Sa, Sb))

        # The projection matrix (the spatial filter) may now be obtained
        SFa = U1.T @ white_mat
        filters = SFa.astype(np.float32)[:self.n_components]
        return filters
    
# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
    cov_mat = A @ A.T
    return cov_mat / np.trace(cov_mat) 

def pseudo_inverse(a, rtol=None):
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