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

def CSP(*tasks):
	if len(tasks) < 2:
		raise ValueError("Must have at least 2 tasks for filtering.")
		return (None,) * len(tasks)
	else:
		filters = ()
		# CSP algorithm
		# For each task x, find the mean variances Rx and not_Rx, which will be used to compute spatial filter SFx
		iterator = range(0,len(tasks))
		for x in iterator:
			# Find Rx
			Rx = covarianceMatrix(tasks[x][0])
			for t in range(1,len(tasks[x])):
				Rx += covarianceMatrix(tasks[x][t])
			Rx = Rx / len(tasks[x])

			# Find not_Rx
			count = 0
			not_Rx = Rx * 0
			for not_x in [element for element in iterator if element != x]:
				for t in range(0,len(tasks[not_x])):
					not_Rx += covarianceMatrix(tasks[not_x][t])
					count += 1
			not_Rx = not_Rx / count

			# Find the spatial filter SFx
			SFx = spatialFilter(Rx,not_Rx)
			filters += (SFx,)

			# Special case: only two tasks, no need to compute any more mean variances
			if len(tasks) == 2:
				filters += (spatialFilter(not_Rx,Rx),)
				break
		return filters

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
    # print('A.shape:', A.shape)
    C = A @ A.T
    # print('C.shape:', C.shape)
    trace = np.trace(C) 
	# Ca = np.trace(np.dot(A,np.transpose(A)))
    return C / trace


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
    P = np.dot(np.sqrt(inv(np.diag(E))),np.transpose(U))
    P = P.real

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

        # # transform X into an array "samples_per_class" of shape (n_classes(2), n_channels, n_times/samples)
        # # samples_per_class[0] is all the concatenated samples of class 0, same goes for samples_per_class[]
        # samples_per_class = []
        # for class_index in range(2):
        #     class_label = unique_labels[class_index]
        #     class_epochs_mask = np.where(y==class_label)
        #     class_epochs = x[class_epochs_mask]
        #     class_samples = np.concatenate(class_epochs, axis=1)
        #     samples_per_class.append(class_samples)
        # transform X into an array "samples_per_class" of shape (n_classes(2), n_channels, n_times/samples)
        # samples_per_class[0] is all the concatenated samples of class 0, same goes for samples_per_class[]
        epochs_per_class = [] # n_classes, n_channels, n_samples
        for class_index in range(2):
            class_label = unique_labels[class_index]
            class_epochs_mask = np.where(y==class_label)
            class_epochs = x[class_epochs_mask]
            # epochs_per_class.append(np.transpose(class_epochs, (0, 2, 1)))
            epochs_per_class.append(class_epochs)

        # print(samples_per_class[0].T.shape)
        self.filters_ = CSP(epochs_per_class[0], epochs_per_class[1])[1]
        # print(self.filters_.shape)
        
        # # samples_per_class = np.asarray(samples_per_class)

        # #mean center samples
        # # self.channels_means_per_class = [np.mean(class_samples, axis=1, keepdims=True) for class_samples in samples_per_class]
        # # print('means:\n', self.channels_means_per_class)
        # # samples_per_class[0] -= self.channels_means_per_class[0]
        # # samples_per_class[1] -= self.channels_means_per_class[1]

        # # print('class_samples.shape: ', samples_per_class[0].shape)
        # covariance_matrices = np.asarray([np.cov(class_samples @ class_samples.T) / class_samples.shape[1] for class_samples in samples_per_class])
        # covariance_matrices = np.asarray([cov_mat / np.trace(cov_mat) for cov_mat in covariance_matrices])
        # # print('my_covs:\n', covariance_matrices)
        
        # # # cov_mats_weights = np.asarray([float(class_samples.shape[1]) for class_samples in samples_per_class])
        # # # gen_eig_vals, gen_eig_vecs = eigh(covariance_matrices[0], covariance_matrices.sum(0))
        # # gen_eig_vals, gen_eig_vecs = eig(covariance_matrices[0], covariance_matrices.sum(0))
        # # gen_eig_vals = gen_eig_vals.real
        # # gen_eig_vecs = gen_eig_vecs.real
        # # # print('my_CSP eig vals:\n', gen_eig_vals)
        # # # normed_gen_eig_vecs = self._normalize_eigenvectors(gen_eig_vecs, covariance_matrices, cov_mats_weights)

        # # # normed_gen_eig_vecs = gen_eig_vecs / norm(gen_eig_vecs, axis=1, keepdims=True)
        # # descending_order_indices = np.argsort(np.abs(gen_eig_vals))[::-1]
        # # self.filters_ = gen_eig_vecs[descending_order_indices]
        
        # self.filters_ = spatialFilter(covariance_matrices[0], covariance_matrices[1])

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