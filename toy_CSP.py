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

data = generate_synthetic_data(1000, (8, 3))

# Calculate the covariance matrix
cov_matrix = np.cov(data, rowvar=False)
print('covariance matrices:')
print(cov_matrix)
print(np.dot(data.T, data))

# Calculate the eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(eigenvectors)

# Plot the data points
plt.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.4)

# Plot the covariance column vectors in green
plt.quiver(0, 0, cov_matrix[0, 0], cov_matrix[1, 0], angles='xy', scale_units='xy', scale=1, color='green', label='Covariance Vector 1')
plt.quiver(0, 0, cov_matrix[0, 1], cov_matrix[1, 1], angles='xy', scale_units='xy', scale=1, color='green', label='Covariance Vector 2')

# Plot the eigenvectors as red arrows
for i in range(len(eigenvalues)):
    eigen_vector = eigenvectors[:, i]
    scaled_eigen_vector = eigenvalues[i] * eigen_vector
    plt.quiver(0, 0, scaled_eigen_vector[0], scaled_eigen_vector[1], angles='xy', scale_units='xy', scale=1, color='red', label=f'Eigenvector {i+1}')

plt.axis('equal')
plt.show()
