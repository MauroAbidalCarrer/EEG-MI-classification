import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

def generate_synthetic_data(num_samples, variances):
    cov_matrix = np.diag(variances)  # Create a covariance matrix with the specified variances
    
    # Generate random samples using the specified mean and covariance matrix
    synthetic_data = np.random.multivariate_normal((0, 0), cov_matrix, num_samples)
    
    return synthetic_data


def rotation_matrix_2d(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Create the 2D rotation matrix
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])
    
    return rotation_matrix

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero
    return vector / norm


data = generate_synthetic_data(1000, (8, 3))
cov = data.T @ data
# print(data.shape)
eig_vecs, eig_vals = la.eig(cov)
print(eig_vecs.shape)

plt.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.4)
for vec in cov:
    vec = normalize_vector(vec)
    plt.arrow(0, 0, vec[0], vec[1], color='red')

# for eig_vec in eig_vecs:
plt.arrow(0, 0, eig_vecs[0], eig_vecs[1], color='green')


plt.axis('square')
plt.show()